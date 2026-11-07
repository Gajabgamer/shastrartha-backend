"""Round and debate scoring (docs/DEBATE_ENGINE_SPEC.md section 6). Pure functions."""

from __future__ import annotations

from typing import Mapping

WEIGHTS: dict[str, float] = {
    "structure": 0.30,
    "evidence": 0.25,
    "pramana_fit": 0.20,
    "responsiveness": 0.15,
    "clarity": 0.10,
}
FALLACY_PENALTY = 10.0
FALLACY_PENALTY_CAP = 30.0
DEFAULT_THRESHOLD = 55.0
AI_WEIGHT = 0.7
CROWD_WEIGHT = 0.3

# Clan playstyle bonuses (section 9): +5% on one component.
CLAN_BONUS: dict[str, str] = {
    "nyaya": "structure",
    "vedanta": "pramana_fit",
    "charvaka": "evidence",
    "arthashastra": "responsiveness",
}
CLAN_BONUS_FACTOR = 1.05
# A shishya "Tested" on a guru (every module of his Read-tab course passed) gets +5% evidence against him.
TESTED_BONUS_FACTOR = 1.05


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def fallacy_penalty(fallacy_count: int) -> float:
    return min(FALLACY_PENALTY * max(0, fallacy_count), FALLACY_PENALTY_CAP)


def apply_clan_bonus(components: Mapping[str, float], clan_id: str | None, uses_shabda: bool = False) -> dict[str, float]:
    """Return a copy of the components with the clan bonus applied (each stays within 0..1).

    Vedanta's bonus only applies when the turn actually uses shabda (testimony).
    """
    out = {k: float(components.get(k, 0.0)) for k in WEIGHTS}
    key = CLAN_BONUS.get(clan_id or "")
    if key and not (clan_id == "vedanta" and not uses_shabda):
        out[key] = min(1.0, out[key] * CLAN_BONUS_FACTOR)
    return out


def round_score(components: Mapping[str, float], fallacy_count: int) -> float:
    """100 * weighted sum of 0..1 components, minus the fallacy penalty, clamped to 0..100."""
    weighted = sum(WEIGHTS[k] * clamp(float(components.get(k, 0.0)), 0.0, 1.0) for k in WEIGHTS)
    return round(clamp(100.0 * weighted - fallacy_penalty(fallacy_count)), 1)


def debate_score(round_scores: list[float]) -> float:
    if not round_scores:
        return 0.0
    return round(clamp(sum(round_scores) / len(round_scores)), 1)


def crowd_score(votes_for_user: int, total_votes: int) -> float | None:
    if total_votes <= 0:
        return None
    return round(100.0 * votes_for_user / total_votes, 1)


def final_score(ai_score: float, crowd: float | None) -> float:
    """Blend AI and crowd scores. Without a Live Sabha the AI score stands alone."""
    if crowd is None:
        return round(clamp(ai_score), 1)
    return round(clamp(AI_WEIGHT * ai_score + CROWD_WEIGHT * crowd), 1)


def user_won(final: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return final >= threshold
