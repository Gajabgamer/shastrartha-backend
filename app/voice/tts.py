"""Fish Audio text-to-speech for the four playable gurus (https://docs.fish.audio, POST /v1/tts).

- Voices are AI character voices chosen by the team (TTS_VOICE_<PERSONA> env vars). None of the four
  historical figures was ever recorded, so the UI labels every clip "AI voice · interpretation".
- Only verified guru text is spoken; citation markers are stripped (the chips stay on screen).
- Fish streams the mp3 as it generates (first bytes in ~1-2 s, a full turn in ~15 s). A `_Job` keeps the
  bytes in memory so the browser can start playing immediately while the rest arrives; the prefetch and the
  browser's request share one job. Finished clips are cached on disk by (model, voice, text), so a replay or
  rematch costs no credits and the demo keeps working offline once the lines have been spoken once.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Iterator
import queue
from pathlib import Path

import requests

from ..config import get_settings, tts_voice

log = logging.getLogger(__name__)

FISH_TTS_URL = "https://api.fish.audio/v1/tts"
MAX_CHARS = 900  # one guru turn is ~110 words; anything longer is cut at a sentence
FIRST_BYTE_TIMEOUT = 25
STALL_TIMEOUT = 30


class TTSUnavailable(RuntimeError):
    pass


def cache_dir() -> Path:
    d = get_settings().data_dir / "voice_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def speakable(text: str) -> str:
    """Remove [src:...]/[inferred] markers and tidy spacing; cap length at a sentence boundary."""
    text = re.sub(r"\s*\[(src:[^\]]+|inferred)\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        text = cut[: max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! ")) + 1] or cut
    return text


class _Job:
    """One in-flight Fish request. Readers follow `buf` as it grows."""

    def __init__(self, persona_id: str, line: str, voice: str, path: Path) -> None:
        self.persona_id, self.line, self.voice, self.path = persona_id, line, voice, path
        self.buf = bytearray()
        self.done = False
        self.error: str | None = None
        self.cond = threading.Condition()

    def run(self) -> None:
        s = get_settings()
        try:
            r = requests.post(
                FISH_TTS_URL,
                headers={"Authorization": f"Bearer {s.fish_api_key}", "model": s.fish_tts_model},
                json={
                    "text": self.line,
                    "reference_id": self.voice,
                    "format": "mp3",
                    "mp3_bitrate": 128,
                    "latency": s.fish_tts_latency,  # balanced/low: first audio in ~1 s; normal: ~8 s
                    "normalize": True,
                    "language": "hi" if re.search(r"[\u0900-\u097F]", self.line) else "en",
                    "prosody": {"speed": 0.95, "normalize_loudness": True},
                },
                stream=True,
                timeout=(10, STALL_TIMEOUT),
            )
            if r.status_code != 200:
                raise TTSUnavailable(f"Fish Audio {r.status_code}: {r.text[:160]}")
            for chunk in r.iter_content(8192):
                if chunk:
                    with self.cond:
                        self.buf += chunk
                        self.cond.notify_all()
            if len(self.buf) < 1000:
                raise TTSUnavailable("Fish Audio returned no audio")
            tmp = self.path.with_suffix(".part")
            tmp.write_bytes(bytes(self.buf))
            tmp.replace(self.path)
            log.info("TTS %s: %d chars -> %s (%d KB)", self.persona_id, len(self.line), self.path.name, len(self.buf) // 1024)
        except (requests.RequestException, TTSUnavailable) as exc:
            self.error = str(exc)
            log.warning("TTS %s failed: %s", self.persona_id, exc)
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()
            with _jobs_guard:
                _jobs.pop(self.path.name, None)

    def wait_first_bytes(self) -> None:
        """Block until audio starts arriving; raise if the request failed first."""
        with self.cond:
            self.cond.wait_for(lambda: self.buf or self.done, timeout=FIRST_BYTE_TIMEOUT)
        if not self.buf:
            raise TTSUnavailable(self.error or "Fish Audio did not answer in time")

    def stream(self) -> Iterator[bytes]:
        sent = 0
        while True:
            with self.cond:
                self.cond.wait_for(lambda: len(self.buf) > sent or self.done, timeout=STALL_TIMEOUT)
                chunk = bytes(self.buf[sent:])
                finished = self.done
            if not chunk:
                return  # finished, or Fish stalled for STALL_TIMEOUT
            sent += len(chunk)
            yield chunk
            if finished and sent >= len(self.buf):
                return


_jobs: dict[str, _Job] = {}
_jobs_guard = threading.Lock()


def _clip(persona_id: str, text: str) -> tuple[Path, str, str]:
    s = get_settings()
    voice = tts_voice(persona_id)
    if not (s.tts_enabled and voice):
        raise TTSUnavailable(f"No voice configured for {persona_id}")
    line = speakable(text)
    if not line:
        raise TTSUnavailable("Nothing to speak")
    key = hashlib.sha256(f"{s.fish_tts_model}|{voice}|{line}".encode()).hexdigest()[:24]
    return cache_dir() / f"{persona_id}-{key}.mp3", line, voice


def _cached(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1000


def start(persona_id: str, text: str) -> Path | _Job:
    """The cached mp3, or the (new or already running) streaming job for it. Never blocks on Fish."""
    path, line, voice = _clip(persona_id, text)
    if _cached(path):
        return path
    with _jobs_guard:
        job = _jobs.get(path.name)
        if job is None:
            if _cached(path):  # finished between the two checks
                return path
            job = _jobs[path.name] = _Job(persona_id, line, voice, path)
            _pool.submit(job.run)  # FIFO: parts queued in order are generated in order, 3 at a time
    return job


# ---------------------------------------------------------------- sentence parts (play part 1 while 2 generates)

class _Workers:
    """FIFO queue served by 3 daemon threads (a ThreadPoolExecutor would block process exit on a slow Fish call)."""

    def __init__(self, n: int) -> None:
        self.q: queue.Queue = queue.Queue()
        for i in range(n):
            threading.Thread(target=self._loop, name=f"tts-{i}", daemon=True).start()

    def _loop(self) -> None:
        while True:
            fn = self.q.get()
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("TTS worker failed")

    def submit(self, fn) -> None:  # noqa: ANN001
        self.q.put(fn)


_pool = _Workers(3)
FIRST_PART_CHARS = 140   # a short first part comes back fastest
PART_CHARS = 260


def parts(text: str) -> list[str]:
    """A guru turn split at sentence ends into speakable parts: a short first one, then ~2-3 sentences each."""
    line = speakable(text)
    sents = [s for s in re.split(r"(?<=[.!?।])\s+", line) if s.strip()]
    out: list[str] = []
    cur = ""
    for s in sents:
        limit = FIRST_PART_CHARS if not out else PART_CHARS
        if cur and len(cur) + 1 + len(s) > limit:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    if len(out) > 1 and len(out[-1]) < 40:  # no tiny trailing part
        out[-2] = f"{out[-2]} {out.pop()}"
    return out


def start_parts(persona_id: str, text: str) -> list[Path | _Job]:
    """Queue every part of a turn (in order). Returns each part's cached path or running job."""
    return [start(persona_id, p) for p in parts(text)]


def synthesize(persona_id: str, text: str) -> Path:
    """Blocking: path to the finished mp3 (scripts and checks)."""
    got = start(persona_id, text)
    if isinstance(got, Path):
        return got
    with got.cond:
        got.cond.wait_for(lambda: got.done, timeout=120)
    if got.error or not _cached(got.path):
        raise TTSUnavailable(got.error or "Fish Audio timed out")
    return got.path
