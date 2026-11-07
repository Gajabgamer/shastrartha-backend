"""Complete reports for debates, online duels and Rivalry Sabhas (TASKS.md workstream B).

GET /api/report/debate/{id}, /api/report/duel/{id}, /api/report/rivalry/{id} each return one JSON document:
meta, sides, full transcript, the "all truth" ledger (every guru claim: verified / inferred / dropped, with its
sources and supporting line), per-round rubric scores with reasons, fallacies, Sabha votes, verdict, jury
conclusion and a heat-map matrix. Live records come from memory; debates that finished before a restart are
rebuilt from the Supabase event log through the same fold (engine.jury.fold_events).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..db import supabase as db
from ..engine import jury, scoring
from ..knowledge import get_knowledge

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

UUID = re.compile(r"^[0-9a-f-]{36}$")
RUBRIC = [("structure", "Structure"), ("evidence", "Evidence"), ("pramana_fit", "Pramana fit"),
          ("responsiveness", "Responsiveness"), ("clarity", "Clarity")]
COLUMNS = [{"key": "round_score", "label": "Round score"}, *({"key": k, "label": lbl, "weight": scoring.WEIGHTS[k]} for k, lbl in RUBRIC),
           {"key": "fallacy", "label": "Fallacy-free"}, {"key": "truth", "label": "Guru verified"},
           {"key": "audience", "label": "Audience share"}]
_jury_cache: dict[str, dict[str, Any]] = {}  # on-demand juries for records that lack one


def _error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


# ---------------------------------------------------------------- shared pieces

def _cell(value: float | None, raw: Any, display: str, reason: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"value": round(max(0.0, min(1.0, float(value))), 3), "raw": raw, "display": display, "reason": reason}


def _rubric_cells(rec: dict[str, Any] | None, threshold: float | None) -> dict[str, Any]:
    if not rec or not rec.get("breakdown"):
        return {}
    b, reasons = rec["breakdown"], rec.get("reasons") or {}
    out: dict[str, Any] = {}
    score = rec.get("round_score")
    if score is not None:
        out["round_score"] = _cell(score / 100, score, f"{score:.0f}",
                                   "Weighted rubric × 100 minus the fallacy penalty" + (f"; the win line is {threshold:g}." if threshold else "."))
    for k, lbl in RUBRIC:
        if k in b:
            out[k] = _cell(b[k], b[k], f"{round(100 * b[k])}%", reasons.get(k) or f"{lbl} {round(100 * b[k])}% (weight {scoring.WEIGHTS[k]:.0%}).")
    pen = float(rec.get("fallacy_penalty", scoring.fallacy_penalty(len(rec.get("fallacies") or []))))
    names = ", ".join(sorted({jury.FALLACY_NAME.get(f.get("type"), str(f.get("type"))) for f in rec.get("fallacies") or []}))
    out["fallacy"] = _cell(1 - pen / scoring.FALLACY_PENALTY_CAP, -pen, f"−{pen:.0f}" if pen else "0",
                           f"Penalty {pen:.0f}: {names}." if pen else "No fallacy flagged.")
    return out


def _truth_cell(claims: list[dict[str, Any]], where: str) -> dict[str, Any] | None:
    c = jury.claim_counts(claims)
    if not c["total"]:
        return None
    return _cell(c["verified"] / c["total"], c, f"{c['verified']}/{c['total']}",
                 f"{where}: {c['verified']} verified, {c['inferred']} inferred, {c['dropped']} withdrawn.")


def _row(key: str, label: str, cells: dict[str, Any]) -> tuple[dict[str, str], list[Any]]:
    return {"key": key, "label": label}, [cells.get(c["key"]) for c in COLUMNS]


def _heatmap(rows: list[tuple[dict[str, str], list[Any]]]) -> dict[str, Any]:
    return {"columns": COLUMNS, "rows": [r for r, _ in rows], "cells": [c for _, c in rows],
            "scale": "0 = weak (sindoor), 0.5 = middling (haldi), 1 = strong (mor)"}


def _ledger(claims: list[dict[str, Any]], speaker_of: dict[int, tuple[str, str]]) -> list[dict[str, Any]]:
    """Every claim with its sources resolved (work, section, edition, licence)."""
    k = get_knowledge()
    cache: dict[str, dict[str, Any]] = {}

    def src(cid: str) -> dict[str, Any]:
        if cid not in cache:
            c = k.chunk(cid)
            cache[cid] = ({"chunk_id": cid, "work": c.work, "section": c.section, "edition": c.edition, "license": c.license,
                           "source_type": c.source_type, "url": c.url} if c else {"chunk_id": cid, "work": None, "section": None})
        return cache[cid]

    out = []
    for c in claims:
        who = speaker_of.get(c.get("turn_index", -1), ("", ""))
        out.append(c | {"text": jury.clean(c.get("text")), "speaker": who[0], "speaker_label": who[1],
                        "sources": [src(s) for s in c.get("source_ids") or []]})
    return out


def _truth(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    c = jury.claim_counts(ledger)
    return {"claims": ledger, "counts": c, "rate": round(c["verified"] / c["total"], 3) if c["total"] else None}


def _sabha_block(tally: dict[str, Any] | None, code: str | None, keys: tuple[str, str], focus: str) -> dict[str, Any] | None:
    """Per-round votes; `focus` is the choice whose share is reported (shishya in a debate)."""
    if not tally:
        return None
    rounds = []
    for r, v in sorted(tally.get("rounds", {}).items(), key=lambda x: int(x[0])):
        n = sum(v.values())
        rounds.append({"round_no": int(r), "votes": {k: v.get(k, 0) for k in keys}, "total": n,
                       "share": round(v.get(focus, 0) / n, 3) if n else None})
    total = {k: tally.get("total", {}).get(k, 0) for k in keys}
    n = sum(total.values())
    return {"code": code, "rounds": rounds, "total": total, "votes": n, "voters": tally.get("voters", 0),
            "share": round(total.get(focus, 0) / n, 3) if n else None, "focus": focus}


async def _attach_jury(report: dict[str, Any], key: str, inp: jury.JuryInput | None, use_llm: bool) -> dict[str, Any]:
    """Use the stored jury; else the rules jury now and, with the LLM, an upgrade in the background (polled)."""
    if report.get("jury"):
        return report
    if key in _jury_cache:
        report["jury"] = _jury_cache[key] | {"pending": jury.pending(key)}
        return report
    if inp is None:
        return report
    fb = await asyncio.to_thread(jury.fallback, inp)
    pending = jury.pending(key)  # the post-verdict hook is still writing it (same key: the record id)
    if report.get("verdict") and not pending:  # unfinished records get a fresh rules jury on every view
        if use_llm:
            pending = jury.run_later(key, lambda: jury.conclude(inp, True), lambda r: _jury_cache.__setitem__(key, r))
        else:
            _jury_cache[key] = fb
    report["jury"] = fb | {"pending": pending}
    return report


# ---------------------------------------------------------------- debate

def _stored_sabha(debate_id: str) -> tuple[dict[str, Any] | None, str | None]:
    sess = db.select("sabha_sessions", f"debate_id=eq.{debate_id}&select=code")
    if not sess:
        return None, None
    code = sess[0]["code"]
    rows = db.select("sabha_votes", f"code=eq.{code}&select=round_no,voter_token,choice")
    rounds: dict[str, dict[str, int]] = {}
    for r in rows:
        rounds.setdefault(str(r["round_no"]), {"guru": 0, "shishya": 0})
        rounds[str(r["round_no"])][r["choice"]] = rounds[str(r["round_no"])].get(r["choice"], 0) + 1
    total = {k: sum(v.get(k, 0) for v in rounds.values()) for k in ("guru", "shishya")}
    return {"rounds": rounds, "total": total, "voters": len({r["voter_token"] for r in rows})}, code


def _load_debate(debate_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict | None, str | None] | None:
    """(meta, events, sabha tally, sabha code) from memory, else from Supabase."""
    from ..engine.debate import DebateError, registry
    from ..engine.sabha import sabhas

    try:
        d = registry.get(debate_id)
    except DebateError:
        d = None
    if d is not None:
        s = sabhas.for_target(d.id)
        meta = {"id": d.id, "persona_id": d.persona.id, "topic": d.topic.model_dump(), "mode": d.req.mode, "clan_id": d.req.clan_id,
                "rounds": d.rounds, "method": "llm" if d.use_llm else "scripted", "status": d.status, "stage": d.stage,
                "created_at": d.created_at, "language": d.req.language, "stored": False}
        return meta, list(d.events), (s.tally() if s else None), (s.code if s else None)
    if not db.enabled() or not UUID.match(debate_id):
        return None
    rows = db.select("debates", f"id=eq.{debate_id}&select=*")
    if not rows:
        return None
    row = rows[0]
    events = db.select("debate_events", f"debate_id=eq.{debate_id}&select=seq,event,data&order=seq.asc")
    from datetime import datetime

    k = get_knowledge()
    topic = k.topics.get(row["topic_id"])
    meta = {"id": row["id"], "persona_id": row["persona_id"],
            "topic": topic.model_dump() if topic else {"id": row["topic_id"], "question": row["question"]},
            "mode": row["mode"], "clan_id": row["clan_id"], "rounds": row["rounds"], "method": row["method"],
            "status": row["status"], "stage": "verdict" if row["status"] == "completed" else None,
            "created_at": datetime.fromisoformat(row["created_at"]).timestamp(), "language": "en", "stored": True,
            "_row": row}
    tally, code = _stored_sabha(debate_id)
    return meta, events, tally, code


def build_debate(debate_id: str) -> tuple[dict[str, Any], jury.JuryInput | None] | None:
    loaded = _load_debate(debate_id)
    if loaded is None:
        return None
    meta, events, tally, code = loaded
    row = meta.pop("_row", None)
    k = get_knowledge()
    persona = k.personas.get(meta["persona_id"])
    name = persona.name if persona else meta["persona_id"]
    threshold = persona.threshold if persona else scoring.DEFAULT_THRESHOLD
    fold = jury.fold_events(events)
    verdict = fold["verdict"]
    if verdict is None and row and row.get("status") == "completed":
        verdict = {"ai_score": row["ai_score"], "crowd_score": row["crowd_score"], "final_score": row["final_score"],
                   "user_won": row["user_won"], "threshold": row["threshold"], "round_scores": row["round_scores"]}
        fold["verdict"] = verdict
    threshold = (verdict or {}).get("threshold") or threshold

    labels = {"guru": name, "shishya": "The shishya"}
    speaker_of = {t["index"]: (t["speaker"], labels[t["speaker"]]) for t in fold["turns"]}
    ledger = _ledger(fold["claims"], speaker_of)
    sabha = _sabha_block(tally, code, ("guru", "shishya"), "shishya")
    sabha_rounds = {r["round_no"]: r for r in (sabha or {}).get("rounds", [])}

    fall_by_round: dict[int, list] = {}
    for f in fold["fallacies"]:
        fall_by_round.setdefault(int(f.get("round_no") or 0), []).append(f)
    rounds, running = [], []
    for r in sorted(fold["rounds"]):
        rec = fold["rounds"][r] | {"fallacies": fall_by_round.get(r, [])}
        running.append(rec.get("round_score") or 0)
        user = next((t for t in fold["turns"] if t["speaker"] == "shishya" and t["round_no"] == r), None)
        rounds.append(rec | {"round_no": r, "running_avg": round(sum(running) / len(running), 1),
                             "pramanas": sorted({p["pramana"] for p in ((user or {}).get("analysis") or {}).get("pramanas", [])}),
                             "won": (rec.get("round_score") or 0) >= threshold})

    rows = []
    opening = [c for c in ledger if c["speaker"] == "guru" and c["round_no"] == 0]
    if opening:
        rows.append(_row("r0", "Opening", {"truth": _truth_cell(opening, "Opening thesis")}))
    for rec in rounds:
        r = rec["round_no"]
        cells = _rubric_cells(rec, threshold)
        cells["truth"] = _truth_cell([c for c in ledger if c["speaker"] == "guru" and c["round_no"] == r], f"Guru's turns in round {r}")
        sr = sabha_rounds.get(r)
        if sr and sr["total"]:
            cells["audience"] = _cell(sr["share"], sr["votes"], f"{round(100 * sr['share'])}%",
                                      f"{sr['votes']['shishya']} of {sr['total']} votes backed the shishya in round {r}.")
        rows.append(_row(f"r{r}", f"Round {r}", cells))

    transcript = [t | {"label": labels[t["speaker"]], "text": t["text"]} for t in fold["turns"]]
    report = {
        "kind": "debate", "id": meta["id"], "meta": meta | {"persona_name": name, "threshold": threshold},
        "sides": [{"key": "guru", "label": name, "persona_id": meta["persona_id"]}, {"key": "shishya", "label": "The shishya"}],
        "transcript": transcript, "truth": _truth(ledger), "rounds": rounds, "fallacies": fold["fallacies"],
        "sabha": sabha, "verdict": verdict, "territory": fold["territory"], "learner": fold["learner"], "jury": fold["jury"],
        "heatmap": _heatmap(rows), "error": fold["error"],
    }
    inp = jury.debate_input(fold, name, meta["topic"].get("question", ""), threshold) if fold["turns"] else None
    return report, inp


@router.get("/report/debate/{debate_id}")
async def debate_report(debate_id: str):
    from ..config import get_settings

    built = await asyncio.to_thread(build_debate, debate_id)
    if built is None:
        return _error("NOT_FOUND", "Debate not found", 404)
    report, inp = built
    return await _attach_jury(report, debate_id, inp, get_settings().llm_available)


# ---------------------------------------------------------------- duel

def _stored_duel_events(match_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not db.enabled() or not UUID.match(match_id):
        return None
    rows = db.select("pvp_matches", f"id=eq.{match_id}&select=*")
    if not rows:
        return None
    row = rows[0]
    events: list[dict[str, Any]] = []
    tr = row.get("transcript") or []
    n_rounds = max((len(p.get("scores") or []) for p in tr), default=0)
    for r in range(1, n_rounds + 1):
        events.append({"seq": r, "event": "round.scored", "data": {"round_no": r, "results": [
            {"player_id": p["player_id"], "text": (p.get("texts") or {}).get(str(r)) or (p.get("texts") or {}).get(r) or "",
             "round_score": (p.get("scores") or [0] * r)[r - 1], "breakdown": {}, "fallacies": []} for p in tr]}})
    if row.get("status") == "completed" and row.get("scores"):
        sc = row["scores"]
        best = sorted(sc.items(), key=lambda x: -x[1])
        winner = best[0][0] if len(best) > 1 and best[0][1] != best[1][1] else None
        events.append({"seq": n_rounds + 1, "event": "match.verdict", "data": {"scores": sc, "winner": winner, "deltas": {}}})
    from datetime import datetime

    meta = {"id": row["id"], "persona_id": row["persona_id"], "topic": {"id": row["topic_id"], "question": row["question"]},
            "players": row["players"], "rounds": n_rounds, "status": row["status"],
            "created_at": datetime.fromisoformat(row["created_at"]).timestamp(), "stored": True}
    t = get_knowledge().topics.get(row["topic_id"])
    if t:
        meta["topic"] = t.model_dump()
    return meta, events


def build_duel(match_id: str) -> tuple[dict[str, Any], jury.JuryInput | None] | None:
    from ..engine.pvp import ROUNDS, PvpError, lobby

    reasons: dict[tuple[str, int], dict[str, str]] = {}
    try:
        m = lobby.get(match_id)
        meta = {"id": m.id, "persona_id": m.persona.id, "topic": m.topic.model_dump(), "players": [p.public() for p in m.players.values()],
                "rounds": ROUNDS, "status": m.status, "stage": m.stage, "created_at": m.created_at, "stored": False}
        events, reasons, stored_jury = list(m.events), dict(m.reasons), m.jury
    except PvpError:
        got = _stored_duel_events(match_id)
        if got is None:
            return None
        (meta, events), stored_jury = got, None
    k = get_knowledge()
    persona = k.personas.get(meta["persona_id"])
    name = persona.name if persona else meta["persona_id"]
    players = meta["players"]
    handle = {p["player_id"]: p.get("handle") or p["player_id"] for p in players}
    fold = jury.fold_events(events)
    labels = {"guru": name, **handle}
    speaker_of = {t["index"]: (t["speaker"], labels.get(t["speaker"], t["speaker"])) for t in fold["turns"]}
    ledger = _ledger(fold["claims"], speaker_of)

    rows, rounds = [], []
    if ledger:
        rows.append(_row("r0", f"Opening · {name}", {"truth": _truth_cell(ledger, f"{name}'s opening")}))
    for r, results in sorted(fold["duel_rounds"].items()):
        entry = []
        for res in results:
            pid = res.get("player_id")
            rec = res | {"reasons": reasons.get((pid, r)) or ({"structure": res["reason"]} if res.get("reason") else {}),
                         "fallacy_penalty": scoring.fallacy_penalty(len(res.get("fallacies") or []))}
            entry.append(rec | {"handle": handle.get(pid, pid)})
            cells = _rubric_cells(rec, None) or {"round_score": _cell((res.get("round_score") or 0) / 100, res.get("round_score"),
                                                                      f"{res.get('round_score') or 0:.0f}", "Round score (stored record).")}
            rows.append(_row(f"r{r}_{pid}", f"Round {r} · {handle.get(pid, pid)}", cells))
        rounds.append({"round_no": r, "results": entry})

    transcript = [t | {"label": labels.get(t["speaker"], t["speaker"])} for t in fold["turns"]]
    report = {
        "kind": "duel", "id": meta["id"], "meta": meta | {"persona_name": name},
        "sides": [{"key": p["player_id"], "label": handle[p["player_id"]], "side": p.get("side")} for p in players],
        "transcript": transcript, "truth": _truth(ledger), "rounds": rounds, "fallacies": fold["fallacies"], "sabha": None,
        "verdict": fold["verdict"], "jury": fold["jury"] or stored_jury, "heatmap": _heatmap(rows), "error": fold["error"],
    }
    inp = jury.duel_input(fold, players, name, meta["topic"].get("question", ""), reasons) if fold["turns"] else None
    return report, inp


@router.get("/report/duel/{match_id}")
async def duel_report(match_id: str):
    from ..config import get_settings

    built = await asyncio.to_thread(build_duel, match_id)
    if built is None:
        return _error("NOT_FOUND", "Match not found", 404)
    report, inp = built
    return await _attach_jury(report, match_id, inp, get_settings().llm_available)


# ---------------------------------------------------------------- rivalry

def build_rivalry(bout_id: str) -> tuple[dict[str, Any], jury.JuryInput | None] | None:
    from ..agents.analyzer import analyze_rules
    from ..agents.referee import score_rules
    from ..engine.rivalry import bouts
    from ..engine.sabha import SabhaError, sabhas

    try:
        b = bouts.get(bout_id)
    except SabhaError:
        return None
    names = {b.a.id: b.a.name, b.b.id: b.b.name}
    other = {b.a.id: b.b, b.b.id: b.a}
    s = sabhas.for_target(b.id)
    tally = s.tally() if s else None
    if tally is None and b.verdict:
        tally = {"rounds": {"1": b.verdict.get("votes") or {}}, "total": b.verdict.get("votes") or {}, "voters": b.verdict.get("voters", 0)}
    sabha = _sabha_block(tally, s.code if s else None, ("a", "b"), "a")
    side_key = {b.a.id: "a", b.b.id: "b"}

    claims, rows, transcript, rounds = [], [], [], []
    for t in b.turns:
        idx = len(transcript)
        tc = [v | {"turn_index": idx, "round_no": t["n"] + 1, "stage": t["stage"]} for v in t.get("verified", [])]
        tc += [v | {"turn_index": idx, "round_no": t["n"] + 1, "stage": t["stage"], "status": "dropped"} for v in t.get("dropped", [])]
        claims += tc
        transcript.append({"index": idx, "speaker": t["persona_id"], "label": names[t["persona_id"]], "stage": t["stage"],
                           "round_no": t["n"] + 1, "text": t["text"], "method": t.get("method"), "claims": t.get("claims", []),
                           "passages": t.get("passages", [])})
        # The same rules judge that scores a shishya, applied to the guru's turn against his rival's weak point.
        text = jury.clean(t["text"])
        prev = jury.clean(b.turns[t["n"] - 1]["text"]) if t["n"] > 0 else b.topic.question
        analysis = analyze_rules(text)
        ref = score_rules(text, prev, analysis, other[t["persona_id"]])
        comps = ref.scores.model_dump()
        rec = {"breakdown": comps, "reasons": ref.reasons, "round_score": scoring.round_score(comps, len(analysis.fallacies)),
               "fallacy_penalty": scoring.fallacy_penalty(len(analysis.fallacies)),
               "fallacies": [f.model_dump() for f in analysis.fallacies], "method": "rules"}
        rounds.append(rec | {"round_no": t["n"] + 1, "speaker": t["persona_id"], "label": names[t["persona_id"]], "stage": t["stage"]})
        cells = _rubric_cells(rec, None)
        cells["truth"] = _truth_cell(tc, f"{names[t['persona_id']]}'s {t['stage']}")
        if sabha and sabha["votes"]:
            mine = sabha["total"][side_key[t["persona_id"]]]
            cells["audience"] = _cell(mine / sabha["votes"], mine, f"{round(100 * mine / sabha['votes'])}%",
                                      f"{mine} of {sabha['votes']} audience votes for {names[t['persona_id']]} (whole Sabha).")
        rows.append(_row(f"t{t['n']}", f"{t['n'] + 1} · {names[t['persona_id']]}", cells))

    speaker_of = {t["index"]: (t["speaker"], t["label"]) for t in transcript}
    ledger = _ledger(claims, speaker_of)
    meta = {"id": b.id, "topic": b.topic.model_dump(), "a": b.a.id, "b": b.b.id, "a_name": b.a.name, "b_name": b.b.name,
            "language": b.language, "status": b.status, "created_at": b.created_at, "integrity": b.integrity(), "stored": False}
    report = {
        "kind": "rivalry", "id": b.id, "meta": meta,
        "sides": [{"key": b.a.id, "label": b.a.name, "persona_id": b.a.id, "vote_key": "a"},
                  {"key": b.b.id, "label": b.b.name, "persona_id": b.b.id, "vote_key": "b"}],
        "transcript": transcript, "truth": _truth(ledger), "rounds": rounds,
        "fallacies": [f | {"round_no": r["round_no"], "speaker": r["speaker"]} for r in rounds for f in r["fallacies"]],
        "sabha": sabha, "verdict": b.verdict, "jury": b.jury, "heatmap": _heatmap(rows), "error": b.error,
    }
    return report, (jury.rivalry_input(b) if b.turns else None)


@router.get("/report/rivalry/{bout_id}")
async def rivalry_report(bout_id: str):
    from ..config import get_settings

    built = await asyncio.to_thread(build_rivalry, bout_id)
    if built is None:
        return _error("NOT_FOUND", "No such rivalry", 404)
    report, inp = built
    return await _attach_jury(report, bout_id, inp, get_settings().llm_available)
