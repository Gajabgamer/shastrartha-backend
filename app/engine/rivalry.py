"""Rivalry Sabha: two gurus argue one question from their own texts; the audience judges.

Order: A states his thesis (the topic's guru, purvapaksha), B answers from his own texts, A replies, B closes.
Every turn goes through the same pipeline as a debate: retrieval from the speaker's corpus (+ TODAY passages for
modern topics), cited generation, claim verification, dropped sentences removed. The audience votes with the
Live Sabha (one vote per phone, changeable until the host closes it). The app reports, per guru, how many of
his claims verified against his own text, which is the objective half of the judgement.
Needs the LLM: the scripted fallback can only restate a topic's own thesis, so it cannot play guru B.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from ..agents.persona import generate_reply
from ..config import get_settings
from ..knowledge import PLAYABLE, get_knowledge
from ..rag.retrieve import Hit, retrieve
from ..rag.verify import verify_claim
from ..schemas import Persona, Topic

log = logging.getLogger(__name__)
ORDER = [("a", "purvapaksha"), ("b", "uttarapaksha"), ("a", "uttarapaksha"), ("b", "siddhanta")]
# Default opponent per guru: the doctrine that clashes most usefully.
RIVAL = {"chanakya": "sun_tzu", "sun_tzu": "chanakya", "napoleon": "sun_tzu", "marcus_aurelius": "chanakya"}


class RivalryError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class Bout:
    def __init__(self, topic: Topic, a: Persona, b: Persona, language: str = "en") -> None:
        self.id = str(uuid.uuid4())
        self.topic, self.a, self.b, self.language = topic, a, b, language
        self.turns: list[dict[str, Any]] = []
        self.status = "speaking"  # speaking | done | closed | error
        self.speaking: str | None = None
        self.error: str | None = None
        self.verdict: dict[str, Any] | None = None
        self.jury: dict[str, Any] | None = None  # jury conclusion, set when the Sabha is closed
        self.created_at = time.time()
        self._task: asyncio.Task | None = None

    def _hits(self, p: Persona, query: str) -> list[Hit]:
        hits = retrieve(p.id, f"{self.topic.question} {query}", k=5)
        if p.id == self.topic.persona_id:
            k = get_knowledge()
            stance = k.chunk(self.topic.stance.source_id)
            if stance and all(h.chunk.chunk_id != stance.chunk_id for h in hits):
                hits = [Hit(stance, 99.0, 0), *hits][:5]
        if self.topic.era == "modern":
            from ..today import today_hits

            hits += today_hits(self.topic, query)
        return hits

    def _transcript(self) -> str:
        names = {self.a.id: self.a.name, self.b.id: self.b.name}
        return "\n".join(f"{names[t['persona_id']].upper()} ({t['stage']}): {t['text']}" for t in self.turns)

    async def run(self) -> None:
        try:
            for i, (side, stage) in enumerate(ORDER):
                me, other = (self.a, self.b) if side == "a" else (self.b, self.a)
                last = self.turns[-1]["text"] if self.turns else ""
                self.speaking = me.id
                hits = self._hits(me, last)
                reply = await generate_reply(
                    me, self.topic, stage, hits, f"rv{i}_{me.id[:4]}", self._transcript(), user_text=last,
                    language=self.language, use_llm=True, opponent=other.name,
                )
                verified = await asyncio.to_thread(lambda: [verify_claim(c) for c in reply.claims])
                text = reply.text
                from .debate import _remove_sentence

                for v in verified:
                    if v.status == "dropped":
                        text = _remove_sentence(text, v.text)
                kept = [v.model_dump() for v in verified if v.status != "dropped"]
                self.turns.append({"n": i, "persona_id": me.id, "stage": stage, "text": text, "method": reply.method,
                                   "claims": [c.model_dump() for c in reply.claims if any(k["claim_id"] == c.claim_id for k in kept)],
                                   "verified": kept,
                                   "dropped": [v.model_dump() for v in verified if v.status == "dropped"],
                                   "passages": [{"chunk_id": h.chunk.chunk_id, "work": h.chunk.work, "section": h.chunk.section} for h in hits]})
                if get_settings().tts_enabled:
                    from ..voice.tts import TTSUnavailable, start

                    try:
                        start(me.id, text)  # voice prefetch while the next guru thinks
                    except TTSUnavailable:
                        pass
            self.status = "done"
        except Exception as exc:  # noqa: BLE001
            log.exception("Rivalry %s failed", self.id)
            self.status, self.error = "error", str(exc)
        finally:
            self.speaking = None

    def integrity(self) -> dict[str, dict[str, int]]:
        out = {}
        for p in (self.a, self.b):
            vs = [v for t in self.turns if t["persona_id"] == p.id for v in t["verified"]]
            out[p.id] = {"verified": sum(v["status"] == "verified" for v in vs), "claims": len(vs)}
        return out

    def close(self) -> dict[str, Any]:
        from .sabha import sabhas

        if self.verdict is not None:
            return self.verdict
        s = sabhas.for_target(self.id)
        total = s.tally()["total"] if s else {}
        n = sum(total.values())
        winner = None
        if n and total.get("a", 0) != total.get("b", 0):
            winner = self.a.id if total.get("a", 0) > total.get("b", 0) else self.b.id
        self.verdict = {"votes": total, "voters": n, "winner": winner, "integrity": self.integrity()}
        self.status = "closed"
        sabhas.close(self.id)
        self._conclude_jury()
        return self.verdict

    def _conclude_jury(self) -> None:
        """Rules jury at once (the snapshot always has one), then the LLM's in the background when available."""
        from . import jury

        try:
            inp = jury.rivalry_input(self)
            self.jury = jury.fallback(inp)
        except Exception:  # noqa: BLE001 - closing must never fail because of the jury
            log.exception("Jury for rivalry %s failed", self.id)
            return
        if get_settings().llm_available:
            jury.run_later(self.id, lambda: jury.conclude(inp, True), lambda r: setattr(self, "jury", r))

    def snapshot(self) -> dict[str, Any]:
        from .sabha import sabhas

        s = sabhas.for_target(self.id)
        return {"bout_id": self.id, "topic": self.topic.model_dump(), "a": self.a.id, "b": self.b.id, "language": self.language,
                "turns": self.turns, "status": self.status, "speaking": self.speaking, "error": self.error,
                "total_turns": len(ORDER), "integrity": self.integrity(), "verdict": self.verdict, "jury": self.jury,
                "sabha": s.public() if s else None, "created_at": self.created_at}


class Bouts:
    def __init__(self) -> None:
        self.items: dict[str, Bout] = {}

    def create(self, topic_id: str, rival_id: str | None, language: str = "en") -> Bout:
        k = get_knowledge()
        topic = k.topics.get(topic_id)
        if topic is None:
            raise RivalryError("NOT_FOUND", "Unknown topic", 404)
        if not get_settings().llm_available:
            raise RivalryError("LLM_UNAVAILABLE", "A rivalry needs the LLM: the scripted guru cannot argue another guru.", 503)
        rival_id = rival_id or RIVAL.get(topic.persona_id)
        if rival_id not in PLAYABLE or rival_id == topic.persona_id:
            raise RivalryError("VALIDATION_ERROR", "Choose a different guru as the rival.", 422)
        b = Bout(topic, k.personas[topic.persona_id], k.personas[rival_id], language)
        self.items[b.id] = b
        from .sabha import sabhas

        sabhas.open("rivalry", b.id, [("a", b.a.name), ("b", b.b.name)])
        b._task = asyncio.create_task(b.run())
        return b

    def get(self, bout_id: str) -> Bout:
        b = self.items.get(bout_id)
        if b is None:
            from .sabha import SabhaError

            raise SabhaError("NOT_FOUND", "No such rivalry", 404)
        return b


bouts = Bouts()
