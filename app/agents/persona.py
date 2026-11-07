"""Persona agent: produces cited replies from retrieved passages only.

LLM path: prompt 1 in docs/PROMPTS.md. Scripted path: composes the reply from the topic stance and
verbatim short quotes of retrieved passages, so every factual sentence still carries [src:id].
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from ..rag.retrieve import Hit
from ..rag.text import short_quote
from ..schemas import Analysis, Claim, Persona, PersonaReply, SupportPoint, Topic
from .llm import MODERN_RULES, SAFETY, LLMUnavailable, complete_json

log = logging.getLogger(__name__)

PERSONA_SYSTEM = """You are an AI interpretation of {name} ({era}), speaking in a structured shastrartha debate.
Rules:
1. Make factual or doctrinal claims ONLY from the PASSAGES provided. Append [src:<chunk_id>] after each such claim.
2. If the sources are silent, you may reason from the persona's documented positions, but mark those sentences [inferred].
3. Never invent quotes. Quote at most 15 words from any passage; paraphrase otherwise.
4. Stay in the persona's documented reasoning style: {style}.
5. In the purvapaksha nobody has argued yet: state your position and end with a challenge to the student.
   In later stages, respond directly to the student's last argument and name its weakest link. Do not ramble.
6. Respect the debate stage: {stage}. Keep to {max_words} words.
7. If asked for something the sources do not cover, say so in character rather than guessing.
8. Language: {language}.
{opponent}{modern}{safety}
Return JSON only: {{"text": "...", "claims": [{{"text": "...", "source_ids": ["..."], "inferred": false}}]}}"""

PERSONA_PROMPT = """TOPIC: {question}
{briefing}PASSAGES:
{passages}
DEBATE SO FAR:
{transcript}
Write the {stage} now."""


LANGUAGES = {
    "en": "English",
    "hi": ("Hindi, in Devanagari script, simple and spoken (as a guru speaks to a student). Keep every [src:...] and "
           "[inferred] marker exactly as written after its sentence. Write Sanskrit terms in Devanagari. In \"claims\", "
           "write each claim's \"text\" in ENGLISH: a faithful translation of that Hindi sentence, in the same order as "
           "they appear, so it can be checked against the English passage"),
}


class _LLMClaim(BaseModel):
    text: str
    source_ids: list[str] = []
    inferred: bool = False


class _LLMReply(BaseModel):
    text: str
    claims: list[_LLMClaim]


def _passages_block(hits: list[Hit]) -> str:
    return "\n".join(f"[{h.chunk.chunk_id}] ({h.chunk.section}) {h.chunk.text}" for h in hits)


_BRACKET = re.compile(r"\s*\[([^\[\]]{1,300})\]")
_SRC_PREFIX = re.compile(r"^\s*src\s*:\s*", re.I)


def _resolve_id(raw: str, allowed: dict[str, str]) -> str | None:
    """Map an LLM-written id ("ARTH_1_12_13", " src: arth_1_12_13 ", "'x'") to a real chunk id."""
    s = _SRC_PREFIX.sub("", raw).strip().strip("\"'`").strip()
    return allowed.get(s) or allowed.get(s.lower())


def normalize_citations(text: str, allowed_ids: set[str]) -> str:
    """Rewrite every citation marker to the canonical form the UI and verifier expect.

    Handles "[src: ID]", "[SRC:a, src:b]", "[src:a; b]", bare "[ARTH_1_12_13]" and "[Inferred]".
    A src marker whose ids are not among the retrieved passages becomes [inferred] (no source, no claim).
    Brackets that are not citations are left untouched.
    """
    allowed = {i: i for i in allowed_ids} | {i.lower(): i for i in allowed_ids}

    def fix(m: re.Match[str]) -> str:
        body = m.group(1).strip()
        if body.lower() == "inferred":
            return " [inferred]"
        is_src = bool(_SRC_PREFIX.match(body))
        ids = [r for p in re.split(r"[,;]|\s+(?=src\s*:)", body, flags=re.I) if (r := _resolve_id(p, allowed))]
        if ids:
            return " " + " ".join(f"[src:{i}]" for i in dict.fromkeys(ids))
        return " [inferred]" if is_src else m.group(0)

    return _BRACKET.sub(fix, text)


def _claim_id(turn_key: str, i: int) -> str:
    return f"{turn_key}_c{i}"


def _weakest_link(analysis: Analysis) -> str:
    s = analysis.structure
    if analysis.fallacies:
        f = analysis.fallacies[0]
        names = {
            "asiddha": "Your reason is assumed, not established.",
            "savyabhichara": "Your reason would serve the opposite conclusion just as well.",
            "viruddha": "Your reason, examined, argues against you.",
            "satpratipaksha": "An equal reason stands against yours.",
            "badhita": "Stronger evidence contradicts your conclusion.",
        }
        return names[f.type]
    if not s.hetu and (s.udaharana or s.upanaya):
        return "Your example carries the weight, but you never state the reason it proves."
    if not s.hetu:
        return "You state a conclusion but give no reason for it."
    if not s.udaharana:
        return "You give a reason but no example that shows it holds."
    return "Your argument is well formed, so let us test its premise."


def _chunk_text(hits: list[Hit], chunk_id: str) -> str:
    hit = next((h for h in hits if h.chunk.chunk_id == chunk_id), None)
    if hit:
        return hit.chunk.text
    from ..knowledge import get_knowledge

    c = get_knowledge().chunk(chunk_id)
    return c.text if c else ""


def scripted_reply(
    persona: Persona,
    topic: Topic,
    stage: str,
    hits: list[Hit],
    turn_key: str,
    user_text: str = "",
    analysis: Analysis | None = None,
    support: SupportPoint | None = None,
) -> PersonaReply:
    claims: list[Claim] = []
    parts: list[str] = []

    def add(text: str, source_ids: list[str], inferred: bool = False, lead: str = "") -> None:
        """`lead` is rhetorical framing shown to the user but not part of the verified claim."""
        tag = "[inferred]" if inferred else " ".join(f"[src:{s}]" for s in source_ids)
        parts.append(f"{lead}{text} {tag}")
        claims.append(Claim(claim_id=_claim_id(turn_key, len(claims) + 1), text=text, source_ids=source_ids, inferred=inferred))

    def lower_first(s: str) -> str:
        return s[0].lower() + s[1:]

    stance = topic.stance
    query = f"{topic.question} {user_text}"

    if stage == "purvapaksha" and topic.era == "modern":
        add("Your question about today is an old question in new dress, and my text already answers it.", [], inferred=True)
    if stage == "purvapaksha":
        add(stance.statement, [stance.source_id])
        primary = next((h for h in hits if h.chunk.chunk_id == stance.source_id), None)
        if primary:
            add(f"\u201c{short_quote(primary.chunk.text, stance.statement)}\u201d", [stance.source_id], lead="As the text has it: ")
        if support is not None:
            add(support.angle, [support.chunk_id], lead="And more: ")
        else:
            extra = next((h for h in hits if h.chunk.chunk_id != stance.source_id), None)
            if extra:
                add(f"\u201c{short_quote(extra.chunk.text, query)}\u201d", [extra.chunk.chunk_id], lead="And again: ")
        parts.append("Refute this, if you can.")
    elif stage == "uttarapaksha":
        parts.append(_weakest_link(analysis) if analysis else "Let us examine your claim.")
        if support is not None:
            add(support.angle, [support.chunk_id])
            add(f"\u201c{short_quote(_chunk_text(hits, support.chunk_id), support.angle)}\u201d",
                [support.chunk_id], lead="In the words of the text: ")
        else:
            fresh = [h for h in hits if h.chunk.chunk_id != stance.source_id][:1] or hits[:1]
            for h in fresh:
                add(f"\u201c{short_quote(h.chunk.text, query)}\u201d", [h.chunk.chunk_id], lead="The text answers you: ")
        add(lower_first(stance.statement), [stance.source_id], lead="So I hold: ")
        parts.append("Your move.")
    else:  # siddhanta
        best_user = _best_user_point(user_text)
        if best_user:
            parts.append(f"Your strongest point was this: \u201c{best_user}\u201d")
        add(lower_first(stance.statement), [stance.source_id], lead="Mine remains: ")
        parts.append("Let the judgement be given.")

    return PersonaReply(text=" ".join(parts), claims=claims, method="scripted")


def _best_user_point(user_text: str) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", user_text) if 6 <= len(s.split()) <= 30]
    if not sents:
        return ""
    cue = re.compile(r"\b(because|therefore|for example|for instance|since)\b", re.I)
    return next((s for s in sents if cue.search(s)), sents[0])


async def generate_reply(
    persona: Persona,
    topic: Topic,
    stage: str,
    hits: list[Hit],
    turn_key: str,
    transcript: str,
    user_text: str = "",
    analysis: Analysis | None = None,
    language: str = "en",
    use_llm: bool = True,
    max_words: int = 110,
    support: SupportPoint | None = None,
    opponent: str | None = None,
) -> PersonaReply:
    if use_llm:
        try:
            modern = topic.era == "modern"
            hindi = language == "hi"
            system = PERSONA_SYSTEM.format(
                name=persona.name, era=persona.era, style=persona.style_notes, stage=stage,
                max_words=max_words + 50 if modern else max_words, language=LANGUAGES.get(language, language), safety=SAFETY,
                modern=MODERN_RULES if modern else "",
                opponent=(f"Your opponent in this Sabha is {opponent}, another classical thinker, not a student; the "
                          f"audience judges you both. Address {opponent} by name"
                          + (" and state your thesis as a challenge to him.\n" if stage == "purvapaksha"
                             else ", answer the argument just made, and argue from your own texts.\n"))
                if opponent else "",
            )
            prompt = PERSONA_PROMPT.format(
                question=topic.question, passages=_passages_block(hits), transcript=transcript[-6000:], stage=stage,
                briefing=f"BRIEFING (today, neutral): {topic.context}\n" if modern and topic.context else "",
            )
            # Devanagari needs ~3x the tokens of English, and the claims are repeated in English for the verifier.
            out = await complete_json(system, prompt, _LLMReply, max_tokens=1800 if hindi else 1200, cap=not hindi)
            allowed = {h.chunk.chunk_id for h in hits}
            lookup = {i: i for i in allowed} | {i.lower(): i for i in allowed}
            claims = []
            for i, c in enumerate(out.claims):
                ids = list(dict.fromkeys(r for s in c.source_ids if (r := _resolve_id(s, lookup))))
                text = re.sub(r"\s*\[(src:[^\]]+|inferred)\]", "", normalize_citations(c.text, allowed)).strip()
                claims.append(Claim(claim_id=_claim_id(turn_key, i + 1), text=text, source_ids=ids,
                                    inferred=c.inferred or not ids))
            return PersonaReply(text=normalize_citations(out.text, allowed), claims=claims, method="llm")
        except LLMUnavailable as exc:
            log.info("Persona falling back to scripted reply: %s", exc)
    return scripted_reply(persona, topic, stage, hits, turn_key, user_text, analysis, support)
