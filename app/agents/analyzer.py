"""Stance, Nyaya structure, pramana and fallacy analysis of the user's argument.

LLM path: prompts 2 and 3 in docs/PROMPTS.md. Rules path: deterministic cue-word analysis used when no
LLM is configured or the call fails. The rules path only flags the fallacies it can detect reliably
(asiddha, savyabhichara); the other three need the LLM or NLI.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from ..rag.text import sentences
from ..schemas import Analysis, Fallacy, PramanaTag, Structure
from .llm import LLMUnavailable, complete_json

log = logging.getLogger(__name__)

CUES = {
    "hetu": r"\b(because|since|for the reason that|as a result of|due to|given that|owing to)\b",
    "udaharana": r"\b(for example|for instance|such as|e\.g\.|consider|as when|take the case|in the case of)\b",
    "upanaya": r"\b(similarly|likewise|in the same way|so too|this case|here too|the same applies|just so)\b",
    "nigamana": r"\b(therefore|thus|hence|so it follows|it follows that|in conclusion|that is why|which is why)\b",
}
PROPOSITION = r"\b(should|must|ought|need to|is better|is wrong|is right|cannot|can't|never|always)\b"

PRAMANA_CUES = {
    "pratyaksha": r"\b(\d{3,4}\s?(bce|bc|ce|ad)?|\d+(\.\d+)?\s?%|observed|we saw|i saw|witnessed|recorded|documented|records show|history shows|in practice|happened|evidence|data|census|survey)\b",
    "anumana": r"\b(because|therefore|thus|hence|it follows|if .{3,60} then|implies|must mean|so it)\b",
    "upamana": r"\b(just as|like a|like the|similar to|as a .{2,30} (does|would)|analogy|compare|resembles|in the same way)\b",
    "shabda": r"\b(according to|wrote|writes|says|said|states|quoted|arthashastra|art of war|edict|scripture|scholars?|historians?|the text)\b",
}

ASSERTION = r"\b(obviously|clearly|everyone knows|everybody knows|it is well known|no one can deny|undeniably|of course|it is obvious)\b"
ABSOLUTE = r"\b(all|every|always|never|no one|nobody|none)\b"

TENSIONS = {
    "power_vs_ethics": r"\b(ethic|moral|right|wrong|virtue|power|ruthless|deceit|deception)\w*",
    "means_vs_ends": r"\b(means|ends|justify|justif|cost|price)\w*",
    "individual_vs_collective": r"\b(privacy|individual|citizen|freedom|liberty|people|society|state|surveil|spy|spies|informer|trust)\w*",
    "war_vs_peace": r"\b(war|peace|battle|fight|conquer|army|enemy)\w*",
    "duty_vs_liberty": r"\b(duty|duties|obligation|liberty|rights?)\b",
}


def _find(pattern: str, sents: list[str], exclude: set[str]) -> str | None:
    for s in sents:
        if s not in exclude and re.search(pattern, s, re.I):
            return s
    return None


def analyze_rules(text: str) -> Analysis:
    sents = sentences(text) or [text.strip()]
    used: set[str] = set()

    nigamana = _find(CUES["nigamana"], sents[1:], used) if len(sents) > 1 else None
    if nigamana:
        used.add(nigamana)
    hetu = _find(CUES["hetu"], sents, used)
    if hetu:
        used.add(hetu)
    udaharana = _find(CUES["udaharana"], sents, used)
    if udaharana:
        used.add(udaharana)
    upanaya = _find(CUES["upanaya"], sents, used)
    if upanaya:
        used.add(upanaya)
    pratijna = _find(PROPOSITION, sents, used) or next((s for s in sents if s not in used), None)

    # A single sentence like "X should Y because Z" carries both the pratijna and the hetu.
    combined = pratijna if pratijna and not hetu else hetu if hetu and not pratijna else None
    if combined:
        m = re.search(CUES["hetu"], combined, re.I)
        if m and m.start() > 0 and re.search(PROPOSITION, combined[: m.start()], re.I):
            pratijna, hetu = combined[: m.start()].strip(" ,;"), combined[m.start():].strip()

    structure = Structure(pratijna=pratijna, hetu=hetu, udaharana=udaharana, upanaya=upanaya, nigamana=nigamana)

    pramanas: list[PramanaTag] = []
    for s in sents:
        for name, pat in PRAMANA_CUES.items():
            if re.search(pat, s, re.I) and not any(p.span == s and p.pramana == name for p in pramanas):
                pramanas.append(PramanaTag(span=s, pramana=name))  # type: ignore[arg-type]

    has_support = any(p.pramana in ("pratyaksha", "shabda") for p in pramanas)
    fallacies: list[Fallacy] = []
    reason_text = hetu or pratijna or ""
    assertion = re.search(ASSERTION, text, re.I)
    if assertion and not has_support:
        span = next((s for s in sents if re.search(ASSERTION, s, re.I)), assertion.group(0))
        fallacies.append(Fallacy(
            type="asiddha",
            span=span,
            explanation="The reason is asserted as obvious but never established; nothing shared or cited supports it.",
            fix="Back the reason with a documented case (pratyaksha) or a reliable source (shabda).",
        ))
    elif hetu and re.search(ABSOLUTE, reason_text, re.I) and not has_support:
        fallacies.append(Fallacy(
            type="savyabhichara",
            span=reason_text,
            explanation="The reason is stated as a universal rule, so a single counter-case would also fit the opposite conclusion.",
            fix="Narrow the claim or show that no counter-case exists.",
        ))

    lowered = text.lower()
    tensions = [k for k, pat in TENSIONS.items() if re.search(pat, lowered)] or ["other"]
    return Analysis(structure=structure, fallacies=fallacies, pramanas=pramanas, tensions=tensions[:3], method="rules")


class _LLMAnalysis(BaseModel):
    structure: Structure
    fallacies: list[Fallacy] = []
    pramanas: list[PramanaTag] = []


ANALYZER_SYSTEM = "You are a Nyaya logic examiner. Return JSON only."
ANALYZER_PROMPT = """Parse the user's argument into the five members
(pratijna, hetu, udaharana, upanaya, nigamana). Mark any missing member as null.
Then check the hetu for hetvabhasa:
- asiddha: reason not established
- savyabhichara: reason also fits the opposite conclusion
- viruddha: reason proves the opposite
- satpratipaksha: an equally strong opposing reason exists
- badhita: conclusion contradicted by stronger evidence (use EVIDENCE below)
The user is meant to argue AGAINST the guru. Disagreeing with the guru's doctrine, or with a passage in EVIDENCE,
is never by itself a fallacy. Flag badhita only when the user states a fact that the EVIDENCE plainly contradicts.
Flag asiddha when the reason is a bare assertion offered as obvious (e.g. "everyone knows", "every rupee returns")
with nothing given to establish it.
Only flag a fallacy if you are confident. Quote the exact span. Give a one-sentence plain explanation and a one-sentence fix.
Also tag each reasoning step with its pramana: pratyaksha, anumana, upamana, shabda.
Return JSON:
{{"structure": {{"pratijna": "...", "hetu": "...", "udaharana": null, "upanaya": null, "nigamana": "..."}},
 "fallacies": [{{"type": "asiddha", "span": "...", "explanation": "...", "fix": "..."}}],
 "pramanas": [{{"span": "...", "pramana": "anumana"}}]}}
USER ARGUMENT: {user_text}
EVIDENCE: {evidence}"""


async def analyze(text: str, evidence: str = "", use_llm: bool = True) -> Analysis:
    rules = analyze_rules(text)
    if not use_llm:
        return rules
    try:
        out = await complete_json(ANALYZER_SYSTEM, ANALYZER_PROMPT.format(user_text=text, evidence=evidence[:4000]), _LLMAnalysis)
        return Analysis(structure=out.structure, fallacies=out.fallacies, pramanas=out.pramanas, tensions=rules.tensions, method="llm")
    except LLMUnavailable as exc:
        log.info("Analyzer falling back to rules: %s", exc)
        return rules
