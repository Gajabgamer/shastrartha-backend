"""Learner and territory state.

In memory for guests (user_id "demo"); write-through to Supabase for signed-in users (user_id = auth uuid)
when SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are set. Tables: supabase/schema.sql. The debate engine only
calls the methods below, so it does not know which backend is active.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..db import supabase as db
from ..knowledge import get_knowledge
from . import bkt, elo
from .territory import RegionState

PRAMANAS = ("pratyaksha", "anumana", "upamana", "shabda")


def is_account(user_id: str) -> bool:
    """True for a Supabase auth user id (persisted); False for guests like "demo"."""
    try:
        uuid.UUID(user_id)
        return db.enabled()
    except ValueError:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Learner:
    user_id: str
    ratings: dict[str, float] = field(default_factory=lambda: {p: elo.INITIAL_RATING for p in PRAMANAS})
    rounds_played: dict[str, int] = field(default_factory=lambda: {p: 0 for p in PRAMANAS})
    mastery: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


class Store:
    def __init__(self) -> None:
        self.learners: dict[str, Learner] = {}
        self.reset_regions()

    def reset_regions(self) -> None:
        self.regions: dict[str, dict] = {r["id"]: copy.deepcopy(r) for r in get_knowledge().regions}
        if db.enabled():
            for row in db.select("regions", "select=id,owner_clan_id,top_score,locked,guardian_persona_id"):
                if row["id"] in self.regions:
                    self.regions[row["id"]].update(owner_clan_id=row["owner_clan_id"], top_score=float(row["top_score"]),
                                                   locked=row["locked"], guardian_persona_id=row["guardian_persona_id"])

    def learner(self, user_id: str) -> Learner:
        if user_id not in self.learners:
            lr = Learner(user_id)
            if is_account(user_id):
                self._load(lr)
            self.learners[user_id] = lr
        return self.learners[user_id]

    def _load(self, lr: Learner) -> None:
        uid = lr.user_id
        for r in db.select("learner_ratings", f"user_id=eq.{uid}&select=pramana,rating,rounds_played"):
            lr.ratings[r["pramana"]] = float(r["rating"])
            lr.rounds_played[r["pramana"]] = int(r["rounds_played"])
        for m in db.select("concept_mastery", f"user_id=eq.{uid}&select=concept_id,p_known"):
            lr.mastery[m["concept_id"]] = float(m["p_known"])
        rows = db.select("debates", f"user_id=eq.{uid}&status=eq.completed&select=id,persona_id,final_score,user_won"
                                    "&order=created_at.asc&limit=50")
        lr.history = [{"debate_id": d["id"], "persona_id": d["persona_id"], "final_score": float(d["final_score"] or 0),
                       "user_won": bool(d["user_won"])} for d in rows]

    def apply_round(self, user_id: str, *, pramanas_used: set[str], persona_difficulty: dict[str, int],
                    won: bool, concept_id: str) -> None:
        lr = self.learner(user_id)
        outcome = 1.0 if won else 0.0
        for p in pramanas_used:
            lr.ratings[p] = elo.update(lr.ratings[p], float(persona_difficulty[p]), outcome, lr.rounds_played[p])
            lr.rounds_played[p] += 1
        prior = lr.mastery.get(concept_id, bkt.DEFAULT.p_init)
        lr.mastery[concept_id] = round(bkt.update(prior, won), 4)
        if is_account(user_id):
            db.upsert("learner_ratings", [
                {"user_id": user_id, "pramana": p, "rating": round(lr.ratings[p], 2), "rounds_played": lr.rounds_played[p],
                 "updated_at": _now()} for p in PRAMANAS
            ], on_conflict="user_id,pramana")
            db.upsert("concept_mastery", {"user_id": user_id, "concept_id": concept_id, "p_known": lr.mastery[concept_id],
                                          "updated_at": _now()}, on_conflict="user_id,concept_id")

    def region_for_guardian(self, persona_id: str) -> RegionState | None:
        for r in self.regions.values():
            if r.get("guardian_persona_id") == persona_id and not r.get("locked"):
                return RegionState(r["id"], r["guardian_persona_id"], r["owner_clan_id"], float(r["top_score"]))
        return None

    def save_region(self, state: RegionState, user_id: str | None = None, captured: bool = False) -> None:
        r = self.regions[state.region_id]
        r["owner_clan_id"] = state.owner_clan_id
        r["top_score"] = state.top_score
        if db.enabled():
            values: dict = {"owner_clan_id": state.owner_clan_id, "top_score": state.top_score}
            if captured:
                values["captured_at"] = _now()
                values["held_by"] = user_id if user_id and is_account(user_id) else None
            db.update("regions", f"id=eq.{state.region_id}", values)


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
