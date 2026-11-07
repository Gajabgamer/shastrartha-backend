"""Speech-to-text for the player's spoken argument, via Gemini audio understanding.

The browser's built-in recogniser handles Indian English poorly, so the Composer records a short clip
(webm/opus) and posts it here. Gemini goes through the same key pool as the agents (rotation on 429,
model fallback on 503). Audio costs ~32 tokens per second, so a one-minute argument is ~2k input tokens.
The clip is not stored.
"""

from __future__ import annotations

import re

from ..agents.llm import LLMUnavailable, gemini_pool
from ..config import get_settings

MAX_BYTES = 6 * 1024 * 1024  # ~3 min of opus; the Composer stops at 2 min
MIME_OK = re.compile(r"^audio/(webm|ogg|mp4|mpeg|wav|x-wav|aac|flac)$")

STT_SYSTEM = """You transcribe a student's spoken debate argument. The speaker is Indian and speaks English
with an Indian accent; they may mix in Hindi words or Sanskrit terms.
Rules:
- Write exactly what was said (in English unless told the debate is in Hindi), with normal punctuation and sentence capitals. Do not summarise,
  improve, answer or comment on the argument.
- Hindi or Sanskrit words stay as spoken, written in Roman script (e.g. "dharma", "rajya", "matlab").
- Drop filler sounds (um, uh, hmm) and false starts. Keep every real word.
- Likely terms: Chanakya, Kautilya, Arthashastra, Chandragupta, Nanda, Magadha, Sun Tzu, Napoleon,
  Marcus Aurelius, Meditations, shastrartha, purvapaksha, khandana, siddhanta, pramana, pratyaksha,
  anumana, upamana, shabda, hetu, pratijna, udaharana, upanaya, nigamana, hetvabhasa, mandala, vijigishu,
  madhyama, udasina, matsya nyaya, yogakshema, danda, amatya, mantri.
- Indian policy terms: Lok Sabha, Rajya Sabha, Election Commission, Supreme Court, freebies, revdi, welfare,
  lateral entry, UPSC, IAS, reservation, Atmanirbhar, DRDO, HAL, Chief of Defence Staff, theatre commands,
  Quad, BRICS, strategic autonomy, non-alignment, farm laws, MSP, GST, fiscal deficit, NITI Aayog.
- If there is no intelligible speech, return an empty string.
- A DEBATE CONTEXT may follow. The student is answering that debate, so when a word could be heard two ways,
  choose the one that fits the debate (e.g. "spies" not "spice", "ally" not "alley", "war" not "wore",
  "realm" not "Rome", "Nanda" not "Nandu"). Use the context only to choose between sound-alikes; never add
  words the student did not say.
Return only the transcript text."""


class STTUnavailable(RuntimeError):
    pass


def debate_context(persona: str, question: str, stance: str = "", briefing: str = "", guru_last: str = "") -> str:
    """Short text block that tells the transcriber what the student is arguing about."""
    guru_last = re.sub(r"\s*\[(src:[^\]]+|inferred)\]", "", guru_last)
    parts = [f"Debating {persona}. Question: {question}"]
    if stance:
        parts.append(f"{persona}'s position: {stance}")
    if briefing:
        parts.append(f"Briefing: {briefing}")
    if guru_last:
        parts.append(f"{persona} just said: {guru_last[-700:]}")
    return "\n".join(parts)


def transcribe(audio: bytes, mime: str, context: str = "") -> str:
    """Blocking; call via asyncio.to_thread. `context` comes from debate_context()."""
    s = get_settings()
    if s.llm_provider != "gemini" or not s.gemini_api_keys:
        raise STTUnavailable("Speech-to-text needs Gemini (LLM_PROVIDER=gemini)")
    mime = mime.split(";")[0].strip().lower()
    if not MIME_OK.match(mime):
        raise STTUnavailable(f"Unsupported audio type {mime}")
    if not 500 <= len(audio) <= MAX_BYTES:
        raise STTUnavailable("Recording is empty or too long")
    try:
        prompt = f"DEBATE CONTEXT:\n{context}\n\nTranscribe this recording." if context else "Transcribe this recording."
        if "LANGUAGE: Hindi" in context:
            prompt += (" This debate is in Hindi: if the student speaks Hindi, write it in Devanagari exactly as spoken, "
                       "keeping English words in Latin script.")
        text = gemini_pool().generate(STT_SYSTEM, prompt, 1000,
                                      audio=(audio, mime), json_mode=False, temperature=0)
    except LLMUnavailable as exc:
        raise STTUnavailable(str(exc)) from exc
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    return text[:2000]
