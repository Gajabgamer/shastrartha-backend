"""Provider-agnostic LLM calls. Provider and model come from env only (CLAUDE.md rule 3)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..config import get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

SAFETY = (
    "Never claim to be the real person. Never give religious rulings. Do not disparage any community, faith, "
    "nation or ethnic group. Do not name or judge living people; speak of states, roles and structures. "
    "On present-day conflicts, analyse structure (alliances, balance of power, costs, deterrence) and do not "
    "take a side, endorse violence, or advise operations against real targets. Do not invent current events, "
    "figures or dates."
)

MODERN_RULES = """This is a question about the world TODAY. Answer it the way you reasoned in your own time:
- Map today's actors onto your own categories (e.g. Chanakya: vijigishu, ari, mitra, madhyama, udasina and the six measures; Sun Tzu: cost, deception, terrain, foreknowledge; Napoleon: the object of war, the offensive, lines of operation; Marcus: what is in our power, opinion, the common good).
- Speak as you would to your student, with a concrete parallel from your own experience or text (as Chanakya taught Chandragupta against the Nandas).
- Your doctrine must be cited [src:<chunk_id>] from the PASSAGES. Every sentence about the modern world is your interpretation: mark it [inferred].
- Use only the general facts in the BRIEFING and in TODAY passages; if you need a fact they do not give, reason conditionally ("if a power does X...").
- PASSAGES whose id starts with "today:" are present-day background (Wikipedia), not your own text. Cite one [src:today:...] only for a plain fact it states about today, never as your doctrine. Your judgement about that fact is still [inferred].
- On Indian politics, argue the policy and its structure (treasury, ministers, alliances, command, consent). Never name, praise or blame a party, leader or government; speak of "a government", "the state", "the ruling side".
"""


class LLMUnavailable(RuntimeError):
    pass


def _call_sync(system: str, prompt: str, max_tokens: int, cap: bool = True) -> str:
    s = get_settings()
    if not s.llm_available:
        raise LLMUnavailable("No LLM provider configured")
    if s.llm_provider == "anthropic":
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=s.anthropic_api_key)
        msg = client.messages.create(
            model=s.llm_model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content)
    if s.llm_provider == "openai":
        import openai  # type: ignore

        client = openai.OpenAI(api_key=s.openai_api_key)
        resp = client.chat.completions.create(
            model=s.llm_model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    if s.llm_provider == "gemini":
        return gemini_pool().generate(system, prompt, min(max_tokens, s.llm_max_tokens) if cap else max_tokens)
    raise LLMUnavailable(f"Unsupported LLM_PROVIDER: {s.llm_provider}")


# ---------------------------------------------------------------- Gemini key pool

QUOTA_CODES = {429}            # rate limit / daily quota: rotate to the next key
KEY_CODES = {401, 403}         # bad or unauthorised key: bench it for longer
OVERLOAD_CODES = {500, 503, 504}  # model overloaded: try the next model, keys are fine
COOLDOWN_S = 65                # Gemini per-minute quotas reset within a minute
BAD_KEY_COOLDOWN_S = 3600


class GeminiPool:
    """Round-robin over the team's keys; a key that hits its quota sits out a cooldown.

    Keys and models come from env only (GEMINI_API_KEYS, LLM_MODEL, LLM_FALLBACK_MODELS).
    Logs show key positions (key#2), never key values.
    """

    def __init__(self, keys: list[str], models: list[str]) -> None:
        if not keys:
            raise LLMUnavailable("No Gemini keys configured")
        self.keys = keys
        self.models = models
        self.next = 0
        self.bench: dict[int, float] = {}  # key index -> usable again at (monotonic)
        self.clients: dict[int, object] = {}
        self.unsupported: set[tuple[int, str]] = set()  # (key index, model) pairs that returned 404
        self.overloaded: dict[str, float] = {}  # model -> skip until (monotonic)
        self.lock = threading.Lock()

    def _client(self, i: int):
        if i not in self.clients:
            from google import genai  # type: ignore

            self.clients[i] = genai.Client(api_key=self.keys[i])
        return self.clients[i]

    def _order(self) -> list[int]:
        """Usable keys, starting from the round-robin pointer."""
        now = time.monotonic()
        with self.lock:
            start = self.next
            self.next = (self.next + 1) % len(self.keys)
        ring = [(start + j) % len(self.keys) for j in range(len(self.keys))]
        return [i for i in ring if self.bench.get(i, 0) <= now]

    def _sit_out(self, i: int, seconds: float) -> None:
        with self.lock:
            self.bench[i] = time.monotonic() + seconds

    def generate(self, system: str, prompt: str, max_tokens: int,
                 audio: tuple[bytes, str] | None = None, json_mode: bool = True, temperature: float = 0.6) -> str:
        """`audio` = (bytes, mime type) is sent before the prompt (speech-to-text)."""
        from google.genai import types  # type: ignore

        contents: object = prompt
        if audio is not None:
            contents = [types.Part.from_bytes(data=audio[0], mime_type=audio[1]), prompt]
        last = "no usable key"
        now = time.monotonic()
        for i in self._order():
            for model in self.models:
                if (i, model) in self.unsupported or self.overloaded.get(model, 0) > now:
                    continue
                config = types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    **({"response_mime_type": "application/json"} if json_mode else {}),  # agents return Pydantic schemas
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    **({"thinking_config": types.ThinkingConfig(thinking_budget=0)} if "2.5" in model else {}),
                )
                try:
                    resp = self._client(i).models.generate_content(model=model, contents=contents, config=config)
                    return resp.text or ""
                except Exception as exc:  # noqa: BLE001
                    code = getattr(exc, "code", None)
                    last = f"{model} key#{i + 1}: {code} {str(exc)[:120]}"
                    if code == 404:  # model not offered to this key's project (e.g. 2.5 for new accounts)
                        log.info("Gemini key#%d cannot use %s; trying its next model", i + 1, model)
                        self.unsupported.add((i, model))
                        continue
                    if code in OVERLOAD_CODES:
                        log.warning("%s overloaded (%s); trying the next model", model, code)
                        self.overloaded[model] = time.monotonic() + 20
                        continue
                    if code in QUOTA_CODES:
                        log.warning("Gemini key#%d hit its quota on %s; rotating", i + 1, model)
                        self._sit_out(i, COOLDOWN_S)
                        break
                    if code in KEY_CODES:
                        log.warning("Gemini key#%d rejected (%s); benched for an hour", i + 1, code)
                        self._sit_out(i, BAD_KEY_COOLDOWN_S)
                        break
                    raise LLMUnavailable(last) from exc
        raise LLMUnavailable(f"All Gemini keys/models exhausted ({last})")

    def status(self) -> dict:
        now = time.monotonic()
        return {"keys": len(self.keys), "benched": sum(1 for t in self.bench.values() if t > now), "models": self.models,
                "unsupported": sorted(f"key#{i + 1}:{m}" for i, m in self.unsupported)}


_pool: GeminiPool | None = None


def gemini_pool() -> GeminiPool:
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = GeminiPool(s.gemini_api_keys, [s.llm_model, *s.llm_fallback_models])
    return _pool


def _extract_json(raw: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fenced:
        return fenced.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start : end + 1] if start != -1 and end > start else raw


async def complete_json(system: str, prompt: str, model: type[T], max_tokens: int = 1200, cap: bool = True) -> T:
    """Call the LLM and validate against `model`. Retries once with the validation error appended.
    `cap=False` lets one call exceed LLM_MAX_TOKENS (Hindi replies need the room)."""
    attempt_prompt = prompt
    for attempt in range(2):
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(asyncio.to_thread(_call_sync, system, attempt_prompt, max_tokens, cap), timeout=60)
            log.info("LLM %s ok in %.1fs (%d chars)", model.__name__, time.monotonic() - t0, len(raw))
        except LLMUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(str(exc)) from exc
        try:
            return model.model_validate(json.loads(_extract_json(raw)))
        except (ValidationError, json.JSONDecodeError) as exc:
            log.warning("LLM JSON invalid (attempt %d): %s", attempt + 1, exc)
            attempt_prompt = f"{prompt}\n\nYour previous output was invalid: {exc}. Return only valid JSON."
    raise LLMUnavailable("LLM returned invalid JSON twice")
