"""Online 1v1: two shishyas debate a topic, a guru moderates, the same judge scores both.

Flow: queue (Elo window widens while you wait) -> match -> guru opening (cited thesis, LLM or scripted)
-> round 1..N: both players submit within the timer; arguments stay hidden until both are in -> judge
scores each argument (analyzer + referee) -> verdict -> Elo moves between the players
(store.apply_round with the opponent's ratings as the "difficulty").

Side "for" defends the guru's thesis; side "against" attacks it. Each player gets a secret at queue time;
every submission must carry it. Events use the same {event, data, seq} envelope as solo debates.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..agents.analyzer import analyze
from ..agents.persona import generate_reply
from ..agents.referee import score_turn
from ..config import get_settings
from ..db import supabase as db
from ..knowledge import get_knowledge
from ..rag.retrieve import Hit, retrieve
from ..rag.verify import verify_claim
from . import scoring
from .store import PRAMANAS, get_store, is_account

log = logging.getLogger(__name__)

ROUNDS = 2
BASE_WINDOW = 100      # Elo half-width at t=0
WINDOW_GROWTH = 40     # +Elo per second waited
MAX_WINDOW = 800


class PvpError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass
class Ticket:
    id: str
    player_id: str
    handle: str
    elo: float
    secret: str
    joined: float = field(default_factory=time.time)
    match_id: str | None = None
    side: str | None = None

    def window(self) -> float:
        return min(MAX_WINDOW, BASE_WINDOW + WINDOW_GROWTH * (time.time() - self.joined))


@dataclass
class Player:
    player_id: str
    handle: str
    elo: float
    secret: str
    side: str  # "for" | "against"
    texts: dict[int, str] = field(default_factory=dict)
    scores: list[float] = field(default_factory=list)
    pramanas: list[set[str]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {"player_id": self.player_id, "handle": self.handle, "elo": round(self.elo), "side": self.side}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:2000]


class PvpMatch:
    def __init__(self, a: Ticket, b: Ticket) -> None:
        k = get_knowledge()
        # Online duels argue present-day questions; Indian policy first (the demo audience), then the world.
        modern = ([t for t in k.topics.values() if t.scope == "india" and not t.id.startswith("news-")]
                  or [t for t in k.topics.values() if t.era == "modern"] or list(k.topics.values()))
        self.topic = random.choice(modern)
        self.persona = k.personas[self.topic.persona_id]
        self.id = str(uuid.uuid4())
        self.created_at = time.time()
        sides = ["for", "against"]
        random.shuffle(sides)
        self.players = {
            t.player_id: Player(t.player_id, t.handle, t.elo, t.secret, s) for t, s in zip((a, b), sides)
        }
        a.match_id = b.match_id = self.id
        a.side, b.side = self.players[a.player_id].side, self.players[b.player_id].side
        self.round_seconds = get_settings().pvp_round_seconds
        self.use_llm = get_settings().llm_available
        self.stage = "opening"
        self.round_no = 0
        self.ends_at: float | None = None
        self.status = "active"
        self.opening = ""
        self.verdict: dict[str, Any] | None = None
        self.jury: dict[str, Any] | None = None  # jury conclusion (report), also emitted as jury.conclusion
        self.reasons: dict[tuple[str, int], dict[str, str]] = {}  # (player_id, round) -> referee reasons per component
        self.events: list[dict[str, Any]] = []
        self._subs: set[asyncio.Queue] = set()
        self._round_done = asyncio.Event()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------ events

    def emit(self, event: str, data: dict[str, Any]) -> None:
        msg = {"event": event, "data": data, "seq": len(self.events) + 1}
        self.events.append(msg)
        for q in list(self._subs):
            q.put_nowait(msg)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def snapshot(self) -> dict[str, Any]:
        return {
            "match_id": self.id, "topic": self.topic.model_dump(), "persona_id": self.persona.id,
            "players": [p.public() for p in self.players.values()], "rounds": ROUNDS, "round_no": self.round_no,
            "stage": self.stage, "ends_at": self.ends_at, "status": self.status, "verdict": self.verdict,
            "round_seconds": self.round_seconds, "created_at": self.created_at, "last_seq": len(self.events),
        }

    # ------------------------------------------------------------ flow

    def start(self) -> None:
        db.insert("pvp_matches", {
            "id": self.id, "topic_id": self.topic.id, "persona_id": self.persona.id, "question": self.topic.question,
            "players": [p.public() | {"user_id": p.player_id if is_account(p.player_id) else None} for p in self.players.values()],
        })
        self._task = asyncio.create_task(self._run())

    def submit(self, player_id: str, secret: str, text: str) -> None:
        p = self.players.get(player_id)
        if not p or not secrets.compare_digest(p.secret, secret):
            raise PvpError("FORBIDDEN", "Not a player in this match.", 403)
        if self.stage != "round":
            raise PvpError("VALIDATION_ERROR", "No round is open.", 409)
        if self.round_no in p.texts:
            raise PvpError("VALIDATION_ERROR", "You already submitted this round.", 409)
        text = _clean(text)
        if not text:
            raise PvpError("VALIDATION_ERROR", "Empty argument.", 422)
        p.texts[self.round_no] = text
        self.emit("player.submitted", {"player_id": player_id, "round_no": self.round_no})
        if all(self.round_no in q.texts for q in self.players.values()):
            self._round_done.set()

    async def _run(self) -> None:
        try:
            await self._opening()
            for r in range(1, ROUNDS + 1):
                await self._round(r)
            self._finish()
            self._conclude_jury()
        except Exception as exc:  # noqa: BLE001
            log.exception("PvP match %s failed", self.id)
            self.status = "abandoned"
            self.emit("match.error", {"message": str(exc)})

    async def _opening(self) -> None:
        self.emit("stage.changed", {"stage": "opening", "round_no": 0})
        k = get_knowledge()
        hits = retrieve(self.persona.id, f"{self.topic.question} {self.topic.stance.statement}", k=5)
        stance = k.chunk(self.topic.stance.source_id)
        if stance and all(h.chunk.chunk_id != stance.chunk_id for h in hits):
            hits = [Hit(stance, 99.0, 0), *hits][:5]
        from ..today import today_hits

        hits = hits + today_hits(self.topic, self.topic.question)
        self.emit("retrieval.passages", {"round_no": 0, "stage": "purvapaksha", "passages": [
            {"chunk_id": h.chunk.chunk_id, "work": h.chunk.work, "section": h.chunk.section, "rank": i + 1, "snippet": h.chunk.text[:160]}
            for i, h in enumerate(hits)]})
        support = self.topic.support[0] if self.topic.support else None
        reply = await generate_reply(self.persona, self.topic, "purvapaksha", hits, f"pvp_{self.id[:6]}_open", "",
                                     use_llm=self.use_llm, support=support)
        verified = await asyncio.to_thread(lambda: [verify_claim(c) for c in reply.claims])
        kept = {v.claim_id for v in verified if v.status != "dropped"}
        self.opening = reply.text
        self.emit("persona.completed", {"stage": "purvapaksha", "round_no": 0, "text": reply.text, "method": reply.method,
                                        "claims": [c.model_dump() for c in reply.claims if c.claim_id in kept]})
        for v in verified:
            if v.claim_id in kept:
                self.emit("claim.verified", v.model_dump(exclude={"text"}))
        for v in verified:
            if v.claim_id not in kept:
                self.emit("claim.dropped", v.model_dump() | {"round_no": 0, "stage": "purvapaksha"})

    async def _round(self, r: int) -> None:
        self.round_no, self.stage = r, "round"
        self._round_done = asyncio.Event()
        self.ends_at = time.time() + self.round_seconds
        self.emit("stage.changed", {"stage": "round", "round_no": r, "ends_at": self.ends_at})
        try:
            await asyncio.wait_for(self._round_done.wait(), timeout=self.round_seconds)
        except asyncio.TimeoutError:
            pass
        self.stage, self.ends_at = "judging", None
        self.emit("stage.changed", {"stage": "judging", "round_no": r})
        a, b = list(self.players.values())
        results = await asyncio.gather(self._judge(a, b, r), self._judge(b, a, r))
        self.emit("round.scored", {"round_no": r, "results": results})

    async def _judge(self, me: Player, other: Player, r: int) -> dict[str, Any]:
        text = me.texts.get(r, "")
        if not text:
            me.scores.append(0.0)
            me.pramanas.append(set())
            return {"player_id": me.player_id, "text": "", "round_score": 0.0, "breakdown": {}, "fallacies": [],
                    "structure": {}, "pramanas": [], "reason": "No argument submitted in time."}
        target = other.texts.get(r - 1) or other.texts.get(r) or self.opening
        hits = retrieve(self.persona.id, text, k=3)
        analysis = await analyze(text, "\n".join(f"[{h.chunk.chunk_id}] {h.chunk.text}" for h in hits), use_llm=self.use_llm)
        ref = await score_turn(text, target, analysis, self.persona, use_llm=self.use_llm)
        comps = {k_: round(v, 3) for k_, v in ref.scores.model_dump().items()}
        self.reasons[(me.player_id, r)] = dict(ref.reasons)
        rs = scoring.round_score(comps, len(analysis.fallacies))
        me.scores.append(rs)
        used = {p.pramana for p in analysis.pramanas}
        me.pramanas.append(used)
        return {"player_id": me.player_id, "text": text, "round_score": rs, "breakdown": comps,
                "fallacies": [f.model_dump() for f in analysis.fallacies], "structure": analysis.structure.model_dump(),
                "pramanas": sorted(used), "reason": next(iter(ref.reasons.values()), ""), "method": ref.method}

    def _finish(self) -> None:
        store = get_store()
        a, b = list(self.players.values())
        avg = {p.player_id: scoring.debate_score(p.scores) for p in (a, b)}
        winner = None if avg[a.player_id] == avg[b.player_id] else max(avg, key=avg.get)
        deltas: dict[str, dict[str, float]] = {}
        for me, other in ((a, b), (b, a)):
            before = dict(store.learner(me.player_id).ratings)
            opp = {k_: int(round(v)) for k_, v in store.learner(other.player_id).ratings.items()}
            for used in me.pramanas:
                if used:
                    store.apply_round(me.player_id, pramanas_used=used, persona_difficulty=opp,
                                      won=winner == me.player_id, concept_id=self.topic.concept_id)
            after = store.learner(me.player_id).ratings
            deltas[me.player_id] = {k_: round(after[k_] - before[k_], 1) for k_ in PRAMANAS}
        self.stage, self.status = "verdict", "completed"
        self.verdict = {"scores": avg, "winner": winner, "deltas": deltas}
        self.emit("stage.changed", {"stage": "verdict", "round_no": self.round_no})
        self.emit("match.verdict", self.verdict)
        db.update("pvp_matches", f"id=eq.{self.id}", {
            "status": "completed", "scores": avg, "winner": winner if winner and is_account(winner) else None,
            "winner_handle": self.players[winner].handle if winner else None,
            "transcript": [{"player_id": p.player_id, "side": p.side, "texts": p.texts, "scores": p.scores} for p in (a, b)],
            "finished_at": _iso(),
        })

    def _conclude_jury(self) -> None:
        from . import jury

        def done(result: dict[str, Any]) -> None:
            self.jury = result
            self.emit("jury.conclusion", result)

        try:
            inp = jury.duel_input(jury.fold_events(self.events), [p.public() for p in self.players.values()],
                                  self.persona.name, self.topic.question, self.reasons)
        except Exception:  # noqa: BLE001 - the verdict stands even if the jury cannot be built
            log.exception("Jury input for match %s failed", self.id)
            return
        jury.run_later(self.id, lambda: jury.conclude(inp, self.use_llm), done)


def _iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class Lobby:
    def __init__(self) -> None:
        self.tickets: dict[str, Ticket] = {}
        self.matches: dict[str, PvpMatch] = {}
        self._lock = asyncio.Lock()

    def waiting(self) -> list[Ticket]:
        return [t for t in self.tickets.values() if t.match_id is None]

    async def join(self, player_id: str, handle: str, elo: float) -> Ticket:
        async with self._lock:
            for t in self.waiting():  # one ticket per player
                if t.player_id == player_id:
                    return t
            t = Ticket(secrets.token_urlsafe(9), player_id, handle, elo, secrets.token_urlsafe(18))
            self.tickets[t.id] = t
        await self.try_match()
        return t

    async def leave(self, ticket_id: str) -> None:
        async with self._lock:
            t = self.tickets.get(ticket_id)
            if t and t.match_id is None:
                del self.tickets[ticket_id]

    async def try_match(self) -> None:
        async with self._lock:
            pool = sorted(self.waiting(), key=lambda t: t.joined)
            used: set[str] = set()
            for i, a in enumerate(pool):
                if a.id in used:
                    continue
                for b in pool[i + 1:]:
                    if b.id in used or b.player_id == a.player_id:
                        continue
                    if abs(a.elo - b.elo) <= min(a.window(), b.window()):
                        m = PvpMatch(a, b)
                        self.matches[m.id] = m
                        used |= {a.id, b.id}
                        m.start()
                        break

    def get(self, match_id: str) -> PvpMatch:
        m = self.matches.get(match_id)
        if not m:
            raise PvpError("NOT_FOUND", "Match not found", 404)
        return m

    async def ticker(self) -> None:
        """Re-check the queue every second (windows widen over time) and drop stale tickets."""
        while True:
            await asyncio.sleep(1)
            now = time.time()
            for tid, t in list(self.tickets.items()):
                if t.match_id is None and now - t.joined > 600:
                    self.tickets.pop(tid, None)
            if len(self.waiting()) >= 2:
                await self.try_match()


lobby = Lobby()
