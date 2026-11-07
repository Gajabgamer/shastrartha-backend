"""Debate orchestrator: the shastrartha state machine.

Nodes mirror the LangGraph diagram in docs/ARCHITECTURE.md:
Purvapaksha -> UserKhandana -> Analyze -> Retrieve -> PersonaReply <-> Verify -> Referee
-> (next round | Siddhanta) -> Verdict -> UpdateLearner -> TerritoryResolve.
Implemented as a hand-rolled async state machine (TECH_STACK.md fallback) so the demo has no extra
runtime dependency; each node is a method and can be lifted into a LangGraph StateGraph unchanged.

Every event is appended to an ordered log with a sequence number, so a reconnecting client can
replay with ?last_seq=N (docs/API_SPEC.md).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..agents.analyzer import analyze
from ..agents.persona import generate_reply
from ..agents.referee import score_turn
from ..config import get_settings
from ..knowledge import get_knowledge
from ..rag.retrieve import Hit, retrieve
from ..rag.text import overlap, short_quote
from ..rag.verify import accuracy, verify_claim
from ..schemas import Analysis, CreateDebate, Persona, PersonaReply, SupportPoint, Topic, VerifiedClaim
from ..db import supabase as db
from . import scoring, territory
from .store import get_store, is_account


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()

log = logging.getLogger(__name__)


class DebateError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass
class Turn:
    speaker: str
    stage: str
    round_no: int
    text: str
    extra: dict[str, Any] = field(default_factory=dict)


class DebateSession:
    def __init__(self, req: CreateDebate) -> None:
        k = get_knowledge()
        if req.persona_id not in k.personas:
            raise DebateError("NOT_FOUND", f"Unknown persona {req.persona_id}", 404)
        topic = k.topics.get(req.topic_id)
        if topic is None or topic.persona_id != req.persona_id:
            raise DebateError("NOT_FOUND", f"Unknown topic {req.topic_id} for {req.persona_id}", 404)

        s = get_settings()
        self.id = str(uuid.uuid4())
        self.created_at = time.time()
        self.req = req
        self.persona: Persona = k.personas[req.persona_id]
        self.topic: Topic = topic
        self.rounds = s.rounds_demo if s.demo_mode else s.rounds_ranked
        self.use_llm = s.llm_available
        self.token_delay = s.token_delay_ms / 1000

        self.stage = "purvapaksha"
        self.round_no = 0
        self.status = "active"
        self.busy = False
        self.turns: list[Turn] = []
        self.round_scores: list[float] = []
        self.round_results: list[dict[str, Any]] = []
        self.all_claims: list[VerifiedClaim] = []
        self.used_chunks: set[str] = set()
        self.used_support: set[str] = set()
        self.verdict: dict[str, Any] | None = None

        self.events: list[dict[str, Any]] = []
        self.seq = 0  # last sequence number; a restored log has gaps where persona.token events were not stored
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._resume: str | None = None  # node to re-run after a restore interrupted it mid-way
        self._last_graph_node: str | None = None

    # ------------------------------------------------------------ events

    def emit(self, event: str, data: dict[str, Any]) -> None:
        self.seq += 1
        msg = {"event": event, "data": data, "seq": self.seq}
        self.events.append(msg)
        if event != "persona.token":  # tokens are rebuilt from persona.completed; skip them to keep the log small
            db.insert("debate_events", {"debate_id": self.id, **msg})
        for q in list(self._subscribers):
            q.put_nowait(msg)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def snapshot(self) -> dict[str, Any]:
        return {
            "debate_id": self.id,
            "persona_id": self.persona.id,
            "topic": self.topic.model_dump(),
            "mode": self.req.mode,
            "clan_id": self.req.clan_id,
            "rounds": self.rounds,
            "round_no": self.round_no,
            "stage": self.stage,
            "status": self.status,
            "method": "llm" if self.use_llm else "scripted",
            "transcript": [t.__dict__ for t in self.turns],
            "round_scores": self.round_scores,
            "verdict": self.verdict,
            "last_seq": self.seq,
            "created_at": self.created_at,
            "study": self.study(),
        }

    def study(self) -> dict[str, Any]:
        """The shishya's study record for this guru: {level, max, tested} (Read-tab course)."""
        from .study import certification

        try:
            return certification(self.req.user_id, self.persona.id)
        except Exception:  # noqa: BLE001 - a missing curriculum must never break a debate
            return {"level": 0, "max": 0, "tested": False}

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._persist_start()  # queued before any event, and the writer is ordered, so the FK holds
        self._task = asyncio.create_task(self._guard(self._purvapaksha()))

    def submit(self, text: str) -> None:
        if self.status != "active":
            raise DebateError("VALIDATION_ERROR", "This debate is finished.", 409)
        if self.busy or self.stage != "khandana":
            raise DebateError("VALIDATION_ERROR", "Wait for the guru to finish speaking.", 409)
        self.busy = True
        self._task = asyncio.create_task(self._guard(self._user_round(text.strip())))

    async def _guard(self, coro) -> None:
        try:
            await coro
        except Exception as exc:  # noqa: BLE001
            log.exception("Debate %s failed", self.id)
            self.emit("debate.error", {"code": "LLM_UNAVAILABLE", "message": str(exc)})
        finally:
            self.busy = False

    async def wait_idle(self) -> None:
        if self._task:
            await self._task

    # ------------------------------------------------------------ nodes

    def _set_stage(self, stage: str) -> None:
        self.stage = stage
        self.emit("stage.changed", {"stage": stage, "round_no": self.round_no})

    def _transcript_text(self) -> str:
        return "\n".join(f"{t.speaker.upper()} ({t.stage}): {t.text}" for t in self.turns)

    def _retrieve(self, query: str, k: int = 5, pinned: str | None = None) -> list[Hit]:
        """BM25 candidates re-ranked for topic relevance, so generic words ("king") don't pull in off-topic passages."""
        k_ = get_knowledge()
        stance_id = self.topic.stance.source_id
        stance_chunk = k_.chunks[stance_id]
        topic_text = f"{self.topic.question} {self.topic.stance.statement}"
        stance_title = stance_chunk.section.split(":", 1)[-1].lower().replace("the ", "").strip()

        candidates = retrieve(self.persona.id, f"{topic_text} {query}", k=20, exclude=self.used_chunks - {stance_id})
        top = max((h.score for h in candidates), default=1.0) or 1.0

        def rank(h: Hit) -> float:
            same_theme = stance_title in h.chunk.section.lower().replace("the ", "")
            return h.score / top + 1.5 * overlap(topic_text, h.chunk.text) + 0.6 * overlap(query, h.chunk.text) + (0.5 if same_theme else 0)

        on_topic = [h for h in candidates if overlap(topic_text, h.chunk.text) > 0.15] or candidates
        hits = sorted(on_topic, key=rank, reverse=True)[:k]
        if self.stage == "purvapaksha" and all(h.chunk.chunk_id != stance_id for h in hits):
            hits = [Hit(stance_chunk, 99.0, 0), *hits][:k]
        if pinned and all(h.chunk.chunk_id != pinned for h in hits):
            hits = [Hit(k_.chunks[pinned], 99.0, 0), *hits][:k]
        elif pinned:
            hits = sorted(hits, key=lambda h: h.chunk.chunk_id != pinned)
        if self.topic.era == "modern":  # plus up to two cited facts about today (Wikipedia background)
            from ..today import today_hits

            hits = hits + today_hits(self.topic, query)
        self.emit("retrieval.passages", {
            "round_no": self.round_no,
            "stage": self.stage,
            "passages": [
                {"chunk_id": h.chunk.chunk_id, "work": h.chunk.work, "section": h.chunk.section, "rank": i + 1,
                 "snippet": short_quote(h.chunk.text, query, max_words=22)}
                for i, h in enumerate(hits)
            ],
        })
        return hits

    def _pick_support(self, user_text: str) -> SupportPoint | None:
        """The unused curated counter-point that best answers the user's argument."""
        unused = [s for s in self.topic.support if s.chunk_id not in self.used_support]
        if not unused:
            return None
        chunks = get_knowledge().chunks
        best = max(unused, key=lambda s: overlap(user_text, f"{s.angle} {chunks[s.chunk_id].text}"))
        self.used_support.add(best.chunk_id)
        return best

    async def _speak(self, user_text: str = "", analysis: Analysis | None = None) -> PersonaReply:
        query = f"{self.topic.question} {user_text}".strip()
        if self.stage == "uttarapaksha":
            support = self._pick_support(user_text)
        elif self.stage == "purvapaksha":
            support = self._pick_support(self.topic.question)
        else:
            support = None
        hits = self._retrieve(query, pinned=support.chunk_id if support else None)
        if self.token_delay:  # let the "consulting the texts" cards land before the guru speaks
            await asyncio.sleep(min(1.6, self.token_delay * 45))
        turn_key = f"r{self.round_no}_{self.stage[:4]}"

        reply = await generate_reply(
            self.persona, self.topic, self.stage, hits, turn_key, self._transcript_text(),
            user_text=user_text, analysis=analysis, language=self.req.language, use_llm=self.use_llm,
            support=support,
        )
        verified = await _verify_all(reply.claims)

        # Verify loop: regenerate failing LLM replies (max MAX_VERIFY_RETRIES), then drop what still fails.
        retries = 0
        while self.use_llm and any(v.status == "dropped" for v in verified) and retries < get_settings().max_verify_retries:
            retries += 1
            reply = await generate_reply(
                self.persona, self.topic, self.stage, hits, turn_key, self._transcript_text(),
                user_text=user_text, analysis=analysis, language=self.req.language, use_llm=True,
            )
            verified = await _verify_all(reply.claims)
        text = reply.text
        for v in verified:
            if v.status == "dropped":
                text = _remove_sentence(text, v.text)
        kept = [v for v in verified if v.status != "dropped"]

        for h in hits:
            if any(h.chunk.chunk_id in v.source_ids for v in kept):
                self.used_chunks.add(h.chunk.chunk_id)

        _prefetch_voice(self.persona.id, text)  # start the voice now, while the text types out on screen
        for piece in _token_groups(text):
            self.emit("persona.token", {"text": piece})
            if self.token_delay:
                await asyncio.sleep(self.token_delay)

        self.emit("persona.completed", {
            "text": text, "method": reply.method, "stage": self.stage, "round_no": self.round_no,
            "claims": [c.model_dump() for c in reply.claims if any(k.claim_id == c.claim_id for k in kept)],
        })
        self.turns.append(Turn("persona", self.stage, self.round_no, text, {"method": reply.method}))

        for v in kept:
            self.emit("claim.verified", {
                "claim_id": v.claim_id, "status": v.status, "source_ids": v.source_ids,
                "nli_prob": v.nli_prob, "nli_label": v.nli_label, "support": v.support, "method": v.method,
            })
            node_id = f"n_{v.claim_id}"
            self.emit("graph.node", {
                "id": node_id, "label": _short(v.text), "kind": "claim" if self.stage == "purvapaksha" else "rebuttal",
                "pramana": "shabda", "parent_id": self._last_graph_node, "speaker": "persona", "status": v.status,
            })
            if self.token_delay:
                await asyncio.sleep(self.token_delay * 4)
        for v in verified:  # removed from the reply; logged so the report's truth ledger shows them too
            if v.status == "dropped":
                self.emit("claim.dropped", v.model_dump() | {"round_no": self.round_no, "stage": self.stage})
        if kept:
            self._last_graph_node = f"n_{kept[-1].claim_id}"
        self.all_claims.extend(kept)
        self.emit("accuracy.updated", accuracy(self.all_claims))
        return reply

    async def _purvapaksha(self) -> None:
        self._set_stage("purvapaksha")
        await self._speak()
        self.round_no = 1
        self._set_stage("khandana")

    async def _user_round(self, text: str) -> None:
        self.turns.append(Turn("user", "khandana", self.round_no, text))
        self.emit("user.submitted", {"round_no": self.round_no, "text": text})  # lets a restart rebuild the transcript
        persona_last = next((t.text for t in reversed(self.turns) if t.speaker == "persona"), "")

        # Analyze
        evidence_hits = retrieve(self.persona.id, text, k=3)
        evidence = "\n".join(f"[{h.chunk.chunk_id}] {h.chunk.text}" for h in evidence_hits)
        analysis = await analyze(text, evidence, use_llm=self.use_llm)
        self.emit("analysis.completed", {
            "round_no": self.round_no, "structure": analysis.structure.model_dump(),
            "pramanas": [p.model_dump() for p in analysis.pramanas], "tensions": analysis.tensions,
            "method": analysis.method,
        })
        user_node = f"n_user_{self.round_no}"
        self.emit("graph.node", {
            "id": user_node, "label": _short(analysis.structure.pratijna or text),
            "kind": "rebuttal", "pramana": analysis.pramanas[0].pramana if analysis.pramanas else None,
            "parent_id": self._last_graph_node, "speaker": "user",
            "fallacy": analysis.fallacies[0].type if analysis.fallacies else None,
        })
        for i, p in enumerate(analysis.pramanas[:4]):
            if p.pramana in ("pratyaksha", "shabda", "upamana"):
                self.emit("graph.node", {
                    "id": f"{user_node}_e{i}", "label": _short(p.span), "kind": "evidence",
                    "pramana": p.pramana, "parent_id": user_node, "speaker": "user",
                })
        self._last_graph_node = user_node
        for f in analysis.fallacies:
            self.emit("fallacy.flagged", f.model_dump() | {"round_no": self.round_no})

        # Referee
        ref = await score_turn(text, persona_last, analysis, self.persona, use_llm=self.use_llm)
        uses_shabda = any(p.pramana == "shabda" for p in analysis.pramanas)
        components = scoring.apply_clan_bonus(ref.scores.model_dump(), self.req.clan_id, uses_shabda)
        tested = self.study()["tested"]
        if tested:  # brownie points: passed every module of this guru's course -> +5% evidence against him
            components["evidence"] = min(1.0, components["evidence"] * scoring.TESTED_BONUS_FACTOR)
        components = {k: round(v, 3) for k, v in components.items()}
        rs = scoring.round_score(components, len(analysis.fallacies))
        self.round_scores.append(rs)
        self.round_results.append({
            "round_no": self.round_no, "score": rs,
            "pramanas": sorted({p.pramana for p in analysis.pramanas}),
            "won": rs >= self.persona.threshold,
        })
        self.emit("score.updated", {
            "round_no": self.round_no, "breakdown": components, "reasons": ref.reasons,
            "fallacy_penalty": scoring.fallacy_penalty(len(analysis.fallacies)), "round_score": rs,
            "method": ref.method, "tested_bonus": tested,
        })

        # Persona responds
        self._set_stage("uttarapaksha")
        await self._speak(user_text=text, analysis=analysis)
        await self._advance(analysis)

    async def _advance(self, analysis: Analysis | None) -> None:
        """After the guru's reply: next round, or the closing siddhanta and verdict."""
        if self.round_no < self.rounds:
            self.round_no += 1
            self._set_stage("khandana")
            return
        await self._siddhanta(analysis)

    async def _siddhanta(self, analysis: Analysis | None) -> None:
        self._set_stage("siddhanta")
        best_round = max(self.round_results, key=lambda r: r["score"])["round_no"] if self.round_results else self.round_no
        best_text = next((t.text for t in self.turns if t.speaker == "user" and t.round_no == best_round), "")
        await self._speak(user_text=best_text, analysis=analysis)
        from .sabha import sabhas

        sabha = sabhas.for_target(self.id)
        if sabha and not sabha.closed:
            # Give the audience time to vote on the last round; the host ends it with POST /finish.
            wait = get_settings().sabha_final_seconds
            self.emit("sabha.awaiting", {"round_no": self.round_no, "seconds": wait, "code": sabha.code})
            self.busy = False
            t0 = time.monotonic()
            while self.verdict is None and time.monotonic() - t0 < wait:
                await asyncio.sleep(0.25)
        self.finish()

    async def _resume_node(self, node: str) -> None:
        if node == "purvapaksha":
            await self._purvapaksha()
        elif node == "uttarapaksha":
            last_user = next((t.text for t in reversed(self.turns) if t.speaker == "user"), "")
            self._set_stage("uttarapaksha")
            await self._speak(user_text=last_user)
            await self._advance(None)
        elif node == "siddhanta":
            await self._siddhanta(None)

    def ensure_running(self) -> None:
        """Re-run the node a server restart interrupted. Needs the event loop, so callers are async routes."""
        if self._resume and self.status == "active" and (self._task is None or self._task.done()):
            node, self._resume = self._resume, None
            self.busy = True
            self._task = asyncio.create_task(self._guard(self._resume_node(node)))

    # ------------------------------------------------------------ restore

    @classmethod
    def restore(cls, debate_id: str) -> "DebateSession | None":
        """Rebuild an unfinished debate from Supabase (debates row + event log) after a server restart."""
        if not db.enabled() or not re.fullmatch(r"[0-9a-f-]{36}", debate_id):
            return None
        rows = db.select("debates", f"id=eq.{debate_id}&select=*")
        if not rows or rows[0]["status"] != "active":
            return None
        row = rows[0]
        events = db.select("debate_events", f"debate_id=eq.{debate_id}&select=seq,event,data&order=seq.asc")
        try:
            d = cls(CreateDebate(persona_id=row["persona_id"], topic_id=row["topic_id"], mode=row["mode"],
                                 clan_id=row["clan_id"], user_id=row["user_id"] or "demo"))
        except DebateError:
            return None
        from datetime import datetime

        d.id = row["id"]
        d.created_at = datetime.fromisoformat(row["created_at"]).timestamp()
        d.rounds = row["rounds"]
        d.events = events
        d.seq = events[-1]["seq"] if events else 0

        claim_text: dict[str, str] = {}
        support_ids = {s.chunk_id for s in d.topic.support}
        pramanas: dict[int, list[str]] = {}
        for e in events:
            ev, data = e["event"], e["data"]
            if ev == "stage.changed":
                d.stage, d.round_no = data["stage"], data["round_no"]
            elif ev == "persona.completed":
                d.turns.append(Turn("persona", data["stage"], data["round_no"], data["text"], {"method": data.get("method")}))
                for c in data.get("claims", []):
                    claim_text[c["claim_id"]] = c["text"]
                    d.used_support.update(s for s in c.get("source_ids", []) if s in support_ids)
            elif ev == "user.submitted":
                d.turns.append(Turn("user", "khandana", data["round_no"], data["text"]))
            elif ev == "claim.verified":
                d.all_claims.append(VerifiedClaim(**{k: data.get(k) for k in VerifiedClaim.model_fields if k != "text"},
                                                  text=claim_text.get(data["claim_id"], "")))
                d.used_chunks.update(data.get("source_ids", []))
                d._last_graph_node = f"n_{data['claim_id']}"
            elif ev == "analysis.completed":
                pramanas[data["round_no"]] = sorted({p["pramana"] for p in data.get("pramanas", [])})
            elif ev == "score.updated":
                r = data["round_no"]
                d.round_scores.append(data["round_score"])
                d.round_results.append({"round_no": r, "score": data["round_score"], "pramanas": pramanas.get(r, []),
                                        "won": data["round_score"] >= d.persona.threshold})
            elif ev == "graph.node" and data.get("speaker") == "user":
                d._last_graph_node = data["id"]

        if d.stage == "khandana":
            # a submission that was still being judged is dropped; the player simply sends it again
            if d.turns and d.turns[-1].speaker == "user" and d.turns[-1].round_no == d.round_no:
                d.turns.pop()
        elif d.stage in ("purvapaksha", "uttarapaksha", "siddhanta"):
            d._resume = d.stage
        elif d.stage == "verdict":
            return None
        log.info("Restored debate %s at %s round %s (%d events)", d.id, d.stage, d.round_no, len(events))
        return d

    # ------------------------------------------------------------ verdict

    def finish(self) -> dict[str, Any]:
        if self.verdict is not None:
            return self.verdict
        from .sabha import sabhas

        ai = scoring.debate_score(self.round_scores)
        sabha = sabhas.for_target(self.id)
        crowd = sabha.crowd_score() if sabha else None  # Live Sabha: % of audience votes for the shishya
        votes = sum(sabha.tally()["total"].values()) if sabha else 0
        final = scoring.final_score(ai, crowd)
        won = scoring.user_won(final, self.persona.threshold)
        self.verdict = {"ai_score": ai, "crowd_score": crowd, "crowd_votes": votes, "final_score": final, "user_won": won,
                        "threshold": self.persona.threshold, "round_scores": self.round_scores}
        sabhas.close(self.id)
        self.stage, self.status = "verdict", "completed"
        self.emit("stage.changed", {"stage": "verdict", "round_no": self.round_no})
        self.emit("debate.verdict", self.verdict)

        store = get_store()
        for rr in self.round_results:
            store.apply_round(self.req.user_id, pramanas_used=set(rr["pramanas"]), persona_difficulty=self.persona.difficulty,
                              won=rr["won"], concept_id=self.topic.concept_id)
        learner = store.learner(self.req.user_id)
        learner.history.append({"debate_id": self.id, "persona_id": self.persona.id, "final_score": final, "user_won": won})
        self.emit("learner.updated", {"ratings": learner.ratings, "mastery": learner.mastery})

        region = store.region_for_guardian(self.persona.id)
        if region:
            res = territory.resolve(region, persona_id=self.persona.id, clan_id=self.req.clan_id, mode=self.req.mode,
                                    won=won, final_score=final)
            store.save_region(res.region, user_id=self.req.user_id, captured=res.changed)
            if res.changed:
                self.emit("territory.changed", {"region_id": region.region_id, "old_owner": res.old_owner,
                                                "new_owner": res.region.owner_clan_id, "top_score": res.region.top_score})
            else:
                self.emit("territory.unchanged", {"region_id": region.region_id, "reason": res.reason})
        db.update("debates", f"id=eq.{self.id}", {
            "status": "completed", "round_scores": self.round_scores, "ai_score": ai, "crowd_score": crowd,
            "final_score": final, "threshold": self.persona.threshold, "user_won": won,
            "region_id": region.region_id if region else None,
            "territory_changed": bool(region and res.changed), "finished_at": _iso_now(),
        })
        self._conclude_jury()
        return self.verdict

    def _conclude_jury(self) -> None:
        """Jury conclusion in the background (one LLM call or the rules fallback), emitted as jury.conclusion."""
        from . import jury

        try:
            inp = jury.debate_input(jury.fold_events(self.events), self.persona.name, self.topic.question, self.persona.threshold)
        except Exception:  # noqa: BLE001 - the verdict stands even if the jury cannot be built
            log.exception("Jury input for debate %s failed", self.id)
            return
        jury.run_later(self.id, lambda: jury.conclude(inp, self.use_llm), lambda r: self.emit("jury.conclusion", r))

    # ------------------------------------------------------------ persistence

    def _persist_start(self) -> None:
        db.insert("debates", {
            "id": self.id, "user_id": self.req.user_id if is_account(self.req.user_id) else None,
            "persona_id": self.persona.id, "topic_id": self.topic.id, "question": self.topic.question,
            "mode": self.req.mode, "clan_id": self.req.clan_id, "rounds": self.rounds,
            "method": "llm" if self.use_llm else "scripted",
        })


def _prefetch_voice(persona_id: str, text: str) -> None:
    """Start the Fish Audio clip in the background so the browser's request hits the cache."""
    if not get_settings().tts_enabled:
        return
    from ..voice.tts import TTSUnavailable, start_parts

    try:
        start_parts(persona_id, text)  # every sentence part queued in order; the browser streams from the same jobs
    except TTSUnavailable as exc:
        log.info("Voice prefetch skipped: %s", exc)


async def _verify_all(claims: list) -> list[VerifiedClaim]:
    """NLI inference is CPU-bound; run it off the event loop so streams and sockets stay live."""
    return await asyncio.to_thread(lambda: [verify_claim(c) for c in claims])


def _token_groups(text: str, size: int = 3) -> list[str]:
    words = re.findall(r"\S+\s*", text)
    return ["".join(words[i : i + size]) for i in range(0, len(words), size)]


def _short(text: str, n: int = 90) -> str:
    text = re.sub(r"\s*\[(src:[^\]]+|inferred)\]", "", text).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _remove_sentence(text: str, claim_text: str) -> str:
    idx = text.find(claim_text)
    if idx == -1:
        return text
    end = text.find("]", idx + len(claim_text))
    end = end + 1 if end != -1 and end - (idx + len(claim_text)) < 60 else idx + len(claim_text)
    return (text[:idx] + text[end:]).replace("  ", " ").strip()


class DebateRegistry:
    def __init__(self) -> None:
        self.debates: dict[str, DebateSession] = {}
        self._restore_lock = threading.Lock()

    def create(self, req: CreateDebate) -> DebateSession:
        d = DebateSession(req)
        self.debates[d.id] = d
        return d

    def get(self, debate_id: str) -> DebateSession:
        """Live debate, or an unfinished one rebuilt from Supabase after a server restart."""
        d = self.debates.get(debate_id)
        if d is None:
            with self._restore_lock:
                d = self.debates.get(debate_id)
                if d is None:
                    try:
                        d = DebateSession.restore(debate_id)
                    except Exception:  # noqa: BLE001 - a bad stored log must not break the route
                        log.exception("Could not restore debate %s", debate_id)
                        d = None
                    if d is not None:
                        self.debates[d.id] = d
        if d is None:
            raise DebateError("NOT_FOUND", "Debate not found", 404)
        return d


registry = DebateRegistry()
