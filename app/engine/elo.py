"""Per-pramana Elo (docs/DEBATE_ENGINE_SPEC.md section 7). Pure functions."""

from __future__ import annotations

INITIAL_RATING = 1000.0
K_NEW = 32.0
K_ESTABLISHED = 16.0
NEW_PLAYER_ROUNDS = 20


def expected(r_user: float, r_persona: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_persona - r_user) / 400.0))


def k_factor(rounds_played: int) -> float:
    return K_NEW if rounds_played < NEW_PLAYER_ROUNDS else K_ESTABLISHED


def update(r_user: float, r_persona: float, outcome: float, rounds_played: int) -> float:
    """outcome: 1 win, 0.5 draw, 0 loss."""
    if outcome not in (0.0, 0.5, 1.0):
        raise ValueError("outcome must be 0, 0.5 or 1")
    return round(r_user + k_factor(rounds_played) * (outcome - expected(r_user, r_persona)), 1)
