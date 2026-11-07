"""REST and WebSocket routes (docs/API_SPEC.md)."""

from __future__ import annotations

import asyncio
import re
import time

from pydantic import BaseModel, Field
from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..config import get_settings
from ..db import supabase as db
from ..engine import curriculum
from ..engine.debate import DebateError, registry
from ..engine.pvp import PvpError, lobby
from ..engine.store import get_store
from ..knowledge import get_knowledge
from ..schemas import CreateDebate, SubmitTurn

api = APIRouter(prefix="/api")
ws = APIRouter()


def error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def current_user(request: Request) -> str:
    """Supabase user id from `Authorization: Bearer <access token>`, else the shared guest "demo"."""
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    return db.user_id_from_token(token) or "demo"


def player_identity(request: Request) -> tuple[str, str]:
    """(player_id, handle) for online play: the Supabase user, else a per-browser guest id."""
    uid = current_user(request)
    if uid != "demo":
        rows = db.select("profiles", f"id=eq.{uid}&select=handle")
        return uid, (rows[0]["handle"] if rows else "shishya")
    gid = request.headers.get("x-guest-id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,40}", gid):
        raise PvpError("VALIDATION_ERROR", "Missing guest id", 422)
    return f"guest_{gid}", f"guest_{gid[:4]}"


def resolve_learner(request: Request, user_id: str) -> str:
    # Only your own learner model is readable: "me" (or "demo") maps to the caller. Other ids are refused.
    return current_user(request) if user_id in ("me", "demo") else ""


def _stored_snapshot(debate_id: str) -> dict | None:
    rows = db.select("debates", f"id=eq.{debate_id}&select=*") if db.enabled() else []
    if not rows:
        return None
    d = rows[0]
    k = get_knowledge()
    topic = k.topics.get(d["topic_id"])
    verdict = None
    if d["status"] == "completed":
        verdict = {"ai_score": d["ai_score"], "crowd_score": d["crowd_score"], "final_score": d["final_score"],
                   "user_won": d["user_won"], "threshold": d["threshold"], "round_scores": d["round_scores"]}
    from datetime import datetime

    return {
        "debate_id": d["id"], "persona_id": d["persona_id"], "topic": topic.model_dump() if topic else {"id": d["topic_id"], "question": d["question"]},
        "mode": d["mode"], "clan_id": d["clan_id"], "rounds": d["rounds"], "round_no": len(d["round_scores"] or []),
        "stage": "verdict" if verdict else "khandana", "status": d["status"], "method": d["method"], "transcript": [],
        "round_scores": d["round_scores"], "verdict": verdict, "last_seq": 0,
        "created_at": datetime.fromisoformat(d["created_at"]).timestamp(), "stored": True,
    }


# ---------------------------------------------------------------- personas and sources

@api.get("/personas")
def list_personas():
    k = get_knowledge()
    return [
        {"id": p.id, "name": p.name, "era": p.era, "role": p.role, "region_id": p.region_id}
        for p in k.personas.values()
    ]


@api.get("/personas/{persona_id}")
def persona_detail(persona_id: str):
    k = get_knowledge()
    p = k.personas.get(persona_id)
    if not p:
        return error("NOT_FOUND", "Unknown persona", 404)
    return p.model_dump() | {"topics": [t.model_dump() for t in k.topics_for(persona_id)]}


@api.get("/personas/{persona_id}/topics")
def persona_topics(persona_id: str):
    k = get_knowledge()
    if persona_id not in k.personas:
        return error("NOT_FOUND", "Unknown persona", 404)
    return [t.model_dump() for t in k.topics_for(persona_id)]


@api.get("/sources/{chunk_id}")
def source(chunk_id: str):
    k = get_knowledge()
    cid = chunk_id.strip()
    c = k.chunk(cid) or (k.chunk(cid.lower()) if cid != cid.lower() else None)
    if not c:
        return error("NOT_FOUND", "Unknown source passage", 404)
    return c.model_dump()


# ---------------------------------------------------------------- debates

@api.post("/debates")
async def create_debate(body: CreateDebate, request: Request):
    # The player is whoever the token says, never a client-supplied id.
    body.user_id = await asyncio.to_thread(current_user, request)
    try:
        d = registry.create(body)
    except DebateError as e:
        return error(e.code, e.message, e.status)
    d.start()
    return {"debate_id": d.id, "rounds": d.rounds, "mode": d.req.mode, "method": "llm" if d.use_llm else "scripted"}


@api.get("/debates")
def list_debates(limit: int = Query(20, ge=1, le=100)):
    """Recent debates, newest first (Sabha TV)."""
    ds = sorted(registry.debates.values(), key=lambda d: d.created_at, reverse=True)[:limit]
    live_ids = {d.id for d in ds}
    stored = []
    if db.enabled():
        from datetime import datetime

        for r in db.select("debates", f"status=eq.completed&select=id,persona_id,question,mode,rounds,final_score,user_won,created_at"
                                      f"&order=created_at.desc&limit={limit}"):
            if r["id"] not in live_ids:
                stored.append({"debate_id": r["id"], "persona_id": r["persona_id"], "question": r["question"], "mode": r["mode"],
                               "stage": "verdict", "round_no": r["rounds"], "rounds": r["rounds"], "status": "completed",
                               "score": r["final_score"], "user_won": r["user_won"],
                               "created_at": datetime.fromisoformat(r["created_at"]).timestamp()})
    return [
        {"debate_id": d.id, "persona_id": d.persona.id, "question": d.topic.question, "mode": d.req.mode,
         "stage": d.stage, "round_no": d.round_no, "rounds": d.rounds, "status": d.status,
         "score": d.verdict["final_score"] if d.verdict else (round(sum(d.round_scores) / len(d.round_scores), 1) if d.round_scores else None),
         "user_won": d.verdict["user_won"] if d.verdict else None, "created_at": d.created_at,
         "tested": d.study()["tested"]}
        for d in ds
    ] + stored


@api.post("/debates/{debate_id}/turns")
async def submit_turn(debate_id: str, body: SubmitTurn):
    try:
        d = await asyncio.to_thread(registry.get, debate_id)  # may rebuild the debate from Supabase after a restart
        d.ensure_running()
        d.submit(body.text)
    except DebateError as e:
        return error(e.code, e.message, e.status)
    return {"accepted": True, "round_no": d.round_no}


@api.post("/debates/{debate_id}/finish")
async def finish_debate(debate_id: str):
    try:
        d = await asyncio.to_thread(registry.get, debate_id)
    except DebateError as e:
        return error(e.code, e.message, e.status)
    if d.busy:
        return error("VALIDATION_ERROR", "A turn is still being processed.", 409)
    if d.stage not in ("siddhanta", "verdict") and d.verdict is None:
        return error("VALIDATION_ERROR", "Finish the rounds first.", 409)
    return d.finish()


@api.get("/debates/{debate_id}/voice/{turn_index}")
async def debate_voice(debate_id: str, turn_index: int):
    """The guru's n-th spoken turn (0-based) as mp3. Text comes from the server transcript, never the client."""
    from pathlib import Path

    from fastapi.responses import StreamingResponse

    from ..voice.tts import TTSUnavailable, start

    try:
        d = await asyncio.to_thread(registry.get, debate_id)
    except DebateError as e:
        return error(e.code, e.message, e.status)
    persona_turns = [t for t in d.turns if t.speaker == "persona"]
    if not 0 <= turn_index < len(persona_turns):
        return error("NOT_FOUND", "No such guru turn yet", 404)
    try:
        got = start(d.persona.id, persona_turns[turn_index].text)
        if not isinstance(got, Path):
            await asyncio.to_thread(got.wait_first_bytes)
    except TTSUnavailable as exc:
        return error("TTS_UNAVAILABLE", str(exc), 503)
    if isinstance(got, Path):
        return FileResponse(got, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=86400"})
    # Still generating: stream what Fish has sent so far and follow it, so playback starts in ~1-2 s.
    return StreamingResponse(got.stream(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


async def _guru_turn_text(debate_id: str, turn_index: int):
    d = await asyncio.to_thread(registry.get, debate_id)
    persona_turns = [t for t in d.turns if t.speaker == "persona"]
    if not 0 <= turn_index < len(persona_turns):
        raise DebateError("NOT_FOUND", "No such guru turn yet", 404)
    return d, persona_turns[turn_index].text


@api.get("/debates/{debate_id}/voice/{turn_index}/parts")
async def debate_voice_parts(debate_id: str, turn_index: int):
    """How many sentence parts the n-th guru turn is spoken in (the player fetches /voice/{n}/{k} in order)."""
    from ..voice.tts import TTSUnavailable, parts, start_parts

    try:
        d, text = await _guru_turn_text(debate_id, turn_index)
    except DebateError as e:
        return error(e.code, e.message, e.status)
    try:
        start_parts(d.persona.id, text)  # make sure every part is queued (already done by the prefetch)
    except TTSUnavailable as exc:
        return error("TTS_UNAVAILABLE", str(exc), 503)
    return {"parts": len(parts(text))}


@api.get("/debates/{debate_id}/voice/{turn_index}/{part}")
async def debate_voice_part(debate_id: str, turn_index: int, part: int):
    """One sentence part as mp3: streamed while Fish generates it, from the disk cache afterwards."""
    from pathlib import Path

    from fastapi.responses import StreamingResponse

    from ..voice.tts import TTSUnavailable, parts, start

    try:
        d, text = await _guru_turn_text(debate_id, turn_index)
    except DebateError as e:
        return error(e.code, e.message, e.status)
    ps = parts(text)
    if not 0 <= part < len(ps):
        return error("NOT_FOUND", "No such part", 404)
    try:
        got = start(d.persona.id, ps[part])
        if not isinstance(got, Path):
            await asyncio.to_thread(got.wait_first_bytes)
    except TTSUnavailable as exc:
        return error("TTS_UNAVAILABLE", str(exc), 503)
    if isinstance(got, Path):
        return FileResponse(got, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=86400"})
    return StreamingResponse(got.stream(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


def _speech_context(debate_id: str | None, match_id: str | None) -> str:
    """Topic and the guru's latest words, so the transcriber hears "spies" in a debate about spies."""
    from ..voice.stt import debate_context

    if debate_id:
        try:
            d = registry.get(debate_id)
        except DebateError:
            return ""
        last = next((t.text for t in reversed(d.turns) if t.speaker == "persona"), "")
        ctx = debate_context(d.persona.name, d.topic.question, d.topic.stance.statement, d.topic.context or "", last)
        return ctx + ("\nLANGUAGE: Hindi" if d.req.language == "hi" else "")
    if match_id:
        try:
            m = lobby.get(match_id)
        except PvpError:
            return ""
        return debate_context(m.persona.name, m.topic.question, m.topic.stance.statement, m.topic.context or "", m.opening)
    return ""


# ---------------------------------------------------------------- live sabha (audience voting)

class VoteBody(BaseModel):
    round_no: int = Field(ge=1, le=10)
    choice: str = Field(min_length=1, max_length=12)
    voter_token: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")


@api.post("/debates/{debate_id}/sabha")
async def open_debate_sabha(debate_id: str, request: Request):
    """Open audience voting for a debate; returns the code, the phone URL and the tally."""
    from ..engine.sabha import sabhas

    try:
        d = await asyncio.to_thread(registry.get, debate_id)
    except DebateError as e:
        return error(e.code, e.message, e.status)
    if d.status != "active":
        return error("VALIDATION_ERROR", "This debate is finished.", 409)
    uid = await asyncio.to_thread(current_user, request)
    from ..engine.store import is_account

    s = sabhas.open("debate", d.id, [("guru", d.persona.name), ("shishya", "The shishya")],
                    created_by=uid if is_account(uid) else None)
    return s.public()


@api.get("/sabha/{code}")
def sabha_state(code: str):
    from ..engine.sabha import SabhaError, sabhas

    try:
        return sabhas.get(code).public()
    except SabhaError as e:
        return error(e.code, e.message, e.status)
    except DebateError as e:
        return error(e.code, e.message, e.status)


@api.post("/sabha/{code}/vote")
def sabha_vote(code: str, body: VoteBody):
    from ..engine.sabha import SabhaError, sabhas

    try:
        s = sabhas.get(code)
        s.vote(body.round_no, body.voter_token, body.choice)
        return {"ok": True, "tally": s.tally(), "crowd_score": s.crowd_score() if s.kind == "debate" else None}
    except SabhaError as e:
        return error(e.code, e.message, e.status)


@api.get("/sabha/{code}/qr.svg")
def sabha_qr(code: str):
    from fastapi.responses import Response

    from ..engine.sabha import SabhaError, qr_svg, sabhas

    try:
        s = sabhas.get(code)
    except SabhaError as e:
        return error(e.code, e.message, e.status)
    return Response(qr_svg(s.public()["join_url"]), media_type="image/svg+xml", headers={"Cache-Control": "max-age=3600"})


# ---------------------------------------------------------------- rivalry sabha (guru vs guru)

class RivalryBody(BaseModel):
    topic_id: str = Field(max_length=80)
    rival_id: str | None = Field(None, max_length=40)
    language: str = Field("en", pattern="^(en|hi)$")


@api.post("/rivalry")
async def create_rivalry(body: RivalryBody):
    from ..engine.rivalry import RivalryError, bouts

    try:
        b = bouts.create(body.topic_id, body.rival_id, body.language)
    except RivalryError as e:
        return error(e.code, e.message, e.status)
    return b.snapshot()


@api.get("/rivalry/{bout_id}")
def get_rivalry(bout_id: str):
    from ..engine.rivalry import bouts
    from ..engine.sabha import SabhaError

    try:
        return bouts.get(bout_id).snapshot()
    except SabhaError as e:
        return error(e.code, e.message, e.status)


@api.post("/rivalry/{bout_id}/close")
def close_rivalry(bout_id: str):
    from ..engine.rivalry import bouts
    from ..engine.sabha import SabhaError

    try:
        b = bouts.get(bout_id)
    except SabhaError as e:
        return error(e.code, e.message, e.status)
    if b.status == "speaking":
        return error("VALIDATION_ERROR", "Let both gurus finish first.", 409)
    return b.close()


@api.get("/rivalry/{bout_id}/voice/{n}")
async def rivalry_voice(bout_id: str, n: int):
    from pathlib import Path

    from fastapi.responses import StreamingResponse

    from ..engine.rivalry import bouts
    from ..engine.sabha import SabhaError
    from ..voice.tts import TTSUnavailable, start

    try:
        b = bouts.get(bout_id)
    except SabhaError as e:
        return error(e.code, e.message, e.status)
    if not 0 <= n < len(b.turns):
        return error("NOT_FOUND", "No such turn yet", 404)
    t = b.turns[n]
    try:
        got = start(t["persona_id"], t["text"])
        if not isinstance(got, Path):
            await asyncio.to_thread(got.wait_first_bytes)
    except TTSUnavailable as exc:
        return error("TTS_UNAVAILABLE", str(exc), 503)
    if isinstance(got, Path):
        return FileResponse(got, media_type="audio/mpeg")
    return StreamingResponse(got.stream(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- today desk

@api.get("/today")
def today_desk(scope: str | None = Query(None, pattern="^(india|world)$")):
    """Present-day topics with their briefing, cited background references and recent headlines."""
    from .. import today

    k = get_knowledge()
    topics = [t for t in k.topics.values() if t.era == "modern" and not t.id.startswith("news-")
              and (scope is None or (t.scope or "world") == scope)]
    topics.sort(key=lambda t: t.scope != "india")
    return [today.record(t) | {"question": t.question, "persona_id": t.persona_id, "scope": t.scope or "world"} for t in topics]


class HeadlineBody(BaseModel):
    title: str = Field(min_length=12, max_length=300)
    url: str = Field("", max_length=600)
    outlet: str = Field("", max_length=120)
    date: str = Field("", max_length=20)


@api.post("/today/debate")
async def headline_debate(body: HeadlineBody):
    """Turn a headline into a grounded debate topic (question, briefing, guru, verified thesis)."""
    from ..headline import HeadlineError, from_headline

    try:
        return await from_headline(body.title, body.url, body.outlet, body.date)
    except HeadlineError as e:
        return error(e.code, e.message, e.status)


@api.get("/today/generated")
def generated_topics():
    """Topics created from headlines, newest first, with the headline each came from."""
    from ..headline import load_generated

    k = get_knowledge()
    items = sorted(load_generated().values(), key=lambda e: e["created_at"], reverse=True)
    return [{"topic": k.topics[e["topic"]["id"]].model_dump(), "headline": e["headline"], "created_at": e["created_at"]}
            for e in items if e["topic"]["id"] in k.topics]


@api.delete("/today/generated/{topic_id}")
def delete_generated(topic_id: str):
    from ..headline import remove

    return {"removed": remove(topic_id)} if topic_id.startswith("news-") else error("VALIDATION_ERROR", "Only generated topics can be removed", 422)


@api.get("/topics/{topic_id}/today")
def topic_today(topic_id: str):
    from .. import today

    t = get_knowledge().topics.get(topic_id)
    if t is None:
        return error("NOT_FOUND", "Unknown topic", 404)
    if t.era != "modern":
        return error("NOT_FOUND", "Classical topics have no present-day briefing", 404)
    return today.record(t)


@api.post("/transcribe")
async def transcribe_audio(request: Request, debate_id: str | None = Query(None, max_length=40),
                           match_id: str | None = Query(None, max_length=40)):
    """Player dictation: raw audio body (Content-Type audio/webm etc.) -> {"text": ...}. Nothing is stored.
    `debate_id` / `match_id` let the transcriber use the debate's topic to pick between sound-alikes."""
    from ..voice.stt import MAX_BYTES, STTUnavailable, transcribe

    if int(request.headers.get("content-length") or 0) > MAX_BYTES:
        return error("VALIDATION_ERROR", "Recording too long", 413)
    body = await request.body()
    context = await asyncio.to_thread(_speech_context, debate_id, match_id)
    try:
        text = await asyncio.to_thread(transcribe, body, request.headers.get("content-type", ""), context)
    except STTUnavailable as exc:
        return error("STT_UNAVAILABLE", str(exc), 503)
    return {"text": text}


@api.get("/debates/{debate_id}")
def get_debate(debate_id: str):
    try:
        return registry.get(debate_id).snapshot()
    except DebateError as e:
        stored = _stored_snapshot(debate_id)  # finished before a server restart: serve the record from Supabase
        return stored if stored else error(e.code, e.message, e.status)


# ---------------------------------------------------------------- learner and map

@api.get("/learners/{user_id}/profile")
def learner_profile(user_id: str, request: Request):
    user_id = resolve_learner(request, user_id)
    if not user_id:
        return error("NOT_FOUND", "Only your own profile is available", 404)
    lr = get_store().learner(user_id)
    return {"user_id": user_id, "ratings": lr.ratings, "rounds_played": lr.rounds_played, "mastery": lr.mastery,
            "history": lr.history[-20:]}


@api.get("/learners/{user_id}/recommendation")
def recommendation(user_id: str, request: Request):
    user_id = resolve_learner(request, user_id) or "demo"
    k = get_knowledge()
    lr = get_store().learner(user_id)
    candidates = [
        (t.persona_id, t.concept_id, k.personas[t.persona_id].weak_points[0].effective_pramana)
        for t in k.topics.values()
    ]
    best = curriculum.recommend(candidates, lr.mastery, lr.ratings)
    if not best:
        return error("NOT_FOUND", "No recommendation", 404)
    persona_id, concept_id, pramana = best
    topic = next(t for t in k.topics.values() if t.persona_id == persona_id and t.concept_id == concept_id)
    return {"persona_id": persona_id, "topic_id": topic.id, "question": topic.question, "focus_pramana": pramana}


@api.get("/map")
def world_map():
    return list(get_store().regions.values())


# ---------------------------------------------------------------- online 1v1

class QueueBody(BaseModel):
    handle: str | None = Field(default=None, max_length=20)


class PvpTurn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@api.post("/pvp/queue")
async def pvp_join(request: Request, body: QueueBody | None = None):
    try:
        pid, handle = await asyncio.to_thread(player_identity, request)
    except PvpError as e:
        return error(e.code, e.message, e.status)
    if body and body.handle and pid.startswith("guest_") and re.fullmatch(r"[A-Za-z0-9_]{3,20}", body.handle):
        handle = body.handle
    lr = await asyncio.to_thread(get_store().learner, pid)
    elo = sum(lr.ratings.values()) / len(lr.ratings)
    t = await lobby.join(pid, handle, elo)
    return {"ticket_id": t.id, "secret": t.secret, "player_id": pid, "handle": handle, "elo": round(elo)}


@api.get("/pvp/queue/{ticket_id}")
def pvp_ticket(ticket_id: str):
    t = lobby.tickets.get(ticket_id)
    if not t:
        return error("NOT_FOUND", "Ticket expired", 404)
    return {"status": "matched" if t.match_id else "waiting", "match_id": t.match_id, "side": t.side,
            "waited": round(time.time() - t.joined, 1), "window": round(t.window()), "elo": round(t.elo),
            "queue_size": len(lobby.waiting())}


@api.delete("/pvp/queue/{ticket_id}")
@api.post("/pvp/queue/{ticket_id}")  # navigator.sendBeacon can only POST (tab closed mid-search)
async def pvp_leave(ticket_id: str):
    await lobby.leave(ticket_id)
    return {"ok": True}


@api.get("/pvp")
def pvp_lobby():
    live = [m for m in lobby.matches.values() if m.status == "active"]
    return {"waiting": len(lobby.waiting()), "live": [
        {"match_id": m.id, "question": m.topic.question, "persona_id": m.persona.id, "round_no": m.round_no,
         "stage": m.stage, "players": [p.public() for p in m.players.values()]} for m in live]}


@api.get("/pvp/{match_id}")
def pvp_match(match_id: str):
    try:
        return lobby.get(match_id).snapshot()
    except PvpError as e:
        return error(e.code, e.message, e.status)


@api.post("/pvp/{match_id}/turns")
def pvp_turn(match_id: str, body: PvpTurn, request: Request):
    try:
        m = lobby.get(match_id)
        m.submit(request.headers.get("x-player-id", ""), request.headers.get("x-player-secret", ""), body.text)
    except PvpError as e:
        return error(e.code, e.message, e.status)
    return {"accepted": True, "round_no": m.round_no}


@api.get("/health")
def health():
    s = get_settings()
    from ..rag.verify import nli_status

    from ..config import tts_voice

    return {"status": "ok", "db": "supabase" if db.enabled() else "memory", "llm": s.llm_available, "llm_model": s.llm_model if s.llm_available else None,
            "llm_keys": len(s.gemini_api_keys) if s.llm_provider == "gemini" else None,
            "demo_mode": s.demo_mode, "nli": nli_status(), "chunks": len(get_knowledge().chunks),
            "voices": [p for p in get_knowledge().personas if s.tts_enabled and tts_voice(p)]}


# ---------------------------------------------------------------- websocket

@ws.websocket("/ws/pvp/{match_id}")
async def pvp_socket(websocket: WebSocket, match_id: str, last_seq: int = Query(0, ge=0)):
    await websocket.accept()
    try:
        m = lobby.get(match_id)
    except PvpError as e:
        await websocket.send_json({"event": "error", "data": {"code": e.code, "message": e.message}, "seq": 0})
        await websocket.close(code=4404)
        return
    q = m.subscribe()
    try:
        for msg in m.events[last_seq:]:
            await websocket.send_json(msg)
        sent = len(m.events)
        while True:
            msg = await q.get()
            if msg["seq"] <= sent:
                continue
            await websocket.send_json(msg)
            sent = msg["seq"]
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        m.unsubscribe(q)


@ws.websocket("/ws/debates/{debate_id}")
async def debate_socket(websocket: WebSocket, debate_id: str, last_seq: int = Query(0, ge=0)):
    await websocket.accept()
    try:
        d = await asyncio.to_thread(registry.get, debate_id)
        d.ensure_running()
    except DebateError as e:
        # Not live any more: replay the stored event log (match reports after a restart), then end.
        rows = await asyncio.to_thread(
            db.select, "debate_events", f"debate_id=eq.{debate_id}&seq=gt.{last_seq}&select=seq,event,data&order=seq.asc"
        ) if db.enabled() else []
        if rows:
            for r in rows:
                await websocket.send_json(r)
            await websocket.send_json({"event": "replay.complete", "data": {}, "seq": rows[-1]["seq"]})
        else:
            await websocket.send_json({"event": "error", "data": {"code": e.code, "message": e.message}, "seq": 0})
        await websocket.close(code=4404 if not rows else 1000)
        return

    queue = d.subscribe()
    try:
        backlog = [m for m in d.events if m["seq"] > last_seq]
        for msg in backlog:
            await websocket.send_json(msg)
        sent = backlog[-1]["seq"] if backlog else last_seq
        while True:
            msg = await queue.get()
            if msg["seq"] <= sent:
                continue
            await websocket.send_json(msg)
            sent = msg["seq"]
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        d.unsubscribe(queue)
