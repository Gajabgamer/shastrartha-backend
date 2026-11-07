"""Referee agent: scores the user's turn on the five rubric components (docs/PROMPTS.md prompt 5)."""

from __future__ import annotations

import logging
import re

from ..rag.text import overlap, sentences
from ..schemas import Analysis, Persona, RefereeResult, Scores
from .llm import LLMUnavailable, complete_json

log = logging.getLogger(__name__)


def score_rules(user_text: str, persona_last: str, analysis: Analysis, persona: Persona) -> RefereeResult:
    s = analysis.structure
    present = s.present()
    coherent = bool(s.pratijna and s.hetu)
    structure = min(1.0, present / 5 + (0.1 if coherent else 0.0))

    kinds = {p.pramana for p in analysis.pramanas}
    support_steps = sum(1 for p in analysis.pramanas if p.pramana in ("pratyaksha", "shabda"))
    evidence = min(1.0, 0.25 * support_steps + (0.15 if s.udaharana else 0.0))

    weak = persona.weak_points[0].effective_pramana if persona.weak_points else None
    if weak and weak in kinds:
        pramana_fit = 1.0
    elif kinds:
        pramana_fit = 0.45
    else:
        pramana_fit = 0.1

    resp_raw = overlap(persona_last, user_text) if persona_last else 0.0
    responsiveness = min(1.0, 0.25 + resp_raw * 2.0) if user_text.strip() else 0.0

    sents = sentences(user_text) or [user_text]
    avg_len = sum(len(x.split()) for x in sents) / len(sents)
    total = len(user_text.split())
    clarity = 1.0 if 8 <= avg_len <= 28 else 0.7 if avg_len <= 40 else 0.4
    if total < 12:
        clarity = min(clarity, 0.5)

    reasons = {
        "structure": f"{present} of 5 members present" + ("; proposition and reason connect." if coherent else "; reason missing."),
        "evidence": f"{support_steps} step(s) backed by observation or testimony" + ("; example given." if s.udaharana else "."),
        "pramana_fit": (
            f"Used {weak}, this guru's weak point." if pramana_fit == 1.0
            else f"Did not use {weak}, the most effective pramana here." if weak else "No pramana detected."
        ),
        "responsiveness": "Engages the guru's last point." if resp_raw >= 0.2 else "Barely engages the guru's last point.",
        "clarity": f"Average sentence length {avg_len:.0f} words.",
    }
    return RefereeResult(
        scores=Scores(structure=round(structure, 2), evidence=round(evidence, 2), pramana_fit=pramana_fit,
                      responsiveness=round(responsiveness, 2), clarity=clarity),
        reasons=reasons, method="rules",
    )


REFEREE_SYSTEM = "You are an impartial referee in a classical Indian debate. Return JSON only."
REFEREE_PROMPT = """Score the USER's latest turn.
Rubric (each 0.0 to 1.0):
- structure: completeness and coherence of the five-member argument
- evidence: use of verifiable, cited support (reward real citations, not assertions)
- pramana_fit: did the user use the pramana most effective against this persona's weak point ({weak_points})?
- responsiveness: engages the persona's actual last point, not a strawman
- clarity: concise, understandable language
Also confirm: did the PERSONA stay in character and within sources? (boolean + note)
Be strict but fair. Give a one-sentence reason per score.
Return JSON:
{{"scores": {{"structure": 0.0, "evidence": 0.0, "pramana_fit": 0.0, "responsiveness": 0.0, "clarity": 0.0}},
 "reasons": {{"structure": "...", "evidence": "...", "pramana_fit": "...", "responsiveness": "...", "clarity": "..."}},
 "persona_in_character": true, "persona_note": "..."}}
USER TURN: {user_text}
PERSONA LAST TURN: {persona_text}
FALLACIES DETECTED: {fallacies}"""


async def score_turn(user_text: str, persona_last: str, analysis: Analysis, persona: Persona, use_llm: bool = True) -> RefereeResult:
    if use_llm:
        try:
            weak = ", ".join(f"{w.label or w.topic} -> {w.effective_pramana}" for w in persona.weak_points)
            fallacies = ", ".join(f.type for f in analysis.fallacies) or "none"
            clean_persona = re.sub(r"\[(src:[^\]]+|inferred)\]", "", persona_last)
            out = await complete_json(
                REFEREE_SYSTEM,
                REFEREE_PROMPT.format(weak_points=weak, user_text=user_text, persona_text=clean_persona, fallacies=fallacies),
                RefereeResult,
            )
            return out.model_copy(update={"method": "llm"})
        except LLMUnavailable as exc:
            log.info("Referee falling back to rules: %s", exc)
    return score_rules(user_text, persona_last, analysis, persona)
