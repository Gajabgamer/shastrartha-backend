"""Territory capture rules (docs/DEBATE_ENGINE_SPEC.md section 8). Pure functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionState:
    region_id: str
    guardian_persona_id: str | None
    owner_clan_id: str | None
    top_score: float


@dataclass(frozen=True)
class CaptureResult:
    changed: bool
    region: RegionState
    old_owner: str | None
    reason: str


def resolve(
    region: RegionState,
    *,
    persona_id: str,
    clan_id: str | None,
    mode: str,
    won: bool,
    final_score: float,
) -> CaptureResult:
    """A ranked (jalpa) win against the guardian captures the region if it beats the recorded top score."""
    old = region.owner_clan_id
    if mode != "jalpa":
        return CaptureResult(False, region, old, "Learning mode (vada) does not put territory at stake.")
    if region.guardian_persona_id != persona_id:
        return CaptureResult(False, region, old, "This persona does not guard the region.")
    if not won:
        return CaptureResult(False, region, old, "The debate was not won.")
    if not clan_id:
        return CaptureResult(False, region, old, "Join a gurukul house to hold territory.")
    if old == clan_id:
        new = RegionState(region.region_id, region.guardian_persona_id, clan_id, max(region.top_score, final_score))
        return CaptureResult(False, new, old, "Your house already holds this region.")
    if old is not None and final_score <= region.top_score:
        return CaptureResult(False, region, old, f"Score {final_score} did not beat the holder's {region.top_score}.")
    new = RegionState(region.region_id, region.guardian_persona_id, clan_id, final_score)
    return CaptureResult(True, new, old, "Region captured.")
