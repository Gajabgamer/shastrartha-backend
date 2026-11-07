"""Bayesian Knowledge Tracing per concept (docs/DEBATE_ENGINE_SPEC.md section 7). Pure functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.2
    p_transit: float = 0.15
    p_guess: float = 0.2
    p_slip: float = 0.1


DEFAULT = BKTParams()
MASTERY = 0.95


def update(p_known: float, correct: bool, params: BKTParams = DEFAULT) -> float:
    if correct:
        num = p_known * (1 - params.p_slip)
        den = num + (1 - p_known) * params.p_guess
    else:
        num = p_known * params.p_slip
        den = num + (1 - p_known) * (1 - params.p_guess)
    posterior = num / den if den else p_known
    return posterior + (1 - posterior) * params.p_transit


def mastered(p_known: float) -> bool:
    return p_known >= MASTERY
