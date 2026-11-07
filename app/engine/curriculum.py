"""Next-challenge recommendation (docs/DEBATE_ENGINE_SPEC.md section 7). Pure functions."""

from __future__ import annotations

from typing import Iterable, Mapping

RATING_MIN = 1000.0
RATING_MAX = 1600.0


def weakness(rating: float) -> float:
    normalized = (rating - RATING_MIN) / (RATING_MAX - RATING_MIN)
    return 1.0 - max(0.0, min(1.0, normalized))


def recommend(
    candidates: Iterable[tuple[str, str, str]],
    p_known: Mapping[str, float],
    ratings: Mapping[str, float],
    default_p: float = 0.2,
) -> tuple[str, str, str] | None:
    """candidates: (persona_id, concept_id, weak_pramana). Maximise (1 - p_known) * weakness(pramana)."""
    best, best_val = None, -1.0
    for persona_id, concept_id, pramana in candidates:
        val = (1 - p_known.get(concept_id, default_p)) * weakness(ratings.get(pramana, RATING_MIN))
        if val > best_val:
            best, best_val = (persona_id, concept_id, pramana), val
    return best
