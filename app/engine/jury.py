"""Jury conclusion for a finished debate, online duel or Rivalry Sabha.

One LLM call (JuryOut) writes a neutral summary and, for each side, what it got right (affirmations) and what it
conceded. Every affirmation must quote that side's own turns verbatim (case, punctuation and spacing ignored);
quotes that are not found are dropped, so the jury cannot put words in anyone's mouth. Without the LLM (or when it
fails) a rules fallback builds the same shape from the scores: best-scoring sentences and verified claims as
affirmations, fallacies, weak rubric components, inferred and withdrawn claims as concessions.

This module also folds an event log ({event, data, seq}) into turns, claims and rounds. The report API and the
debate/duel hooks use the same fold, so a live debate and one rebuilt from Supabase read identically.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from ..config import get_settings
from ..rag.text import overlap, sentences

log = logging.getLogger(__name__)

MARKER = re.compile(r"\s*\[\s*(?:src\s*:[^\]]*|inferred)\s*\]", re.I)
MAX_QUOTE_WORDS = 25
MIN_QUOTE_WORDS = 3
COMPONENTS = ("structure", "evidence", "pramana_fit", "responsiveness", "clarity")
COMPONENT_LABEL = {"structure": "structure", "evidence": "evidence", "pramana_fit": "choice of pramana",
                   "responsiveness": "responsiveness", "clarity": "clarity"}
FALLACY_NAME = {"asiddha": "Asiddha (unproven reason)", "savyabhichara": "Savyabhichara (inconclusive reason)",
                "viruddha": "Viruddha (contradictory reason)", "satpratipaksha": "Satpratipaksha (counterbalanced reason)",
                "badhita": "Badhita (contradicted by evidence)"}
REASON_WORDS = re.compile(r"\b(because|therefore|thus|hence|since|for example|for instance|so that|which means|in the same way)\b", re.I)


# ---------------------------------------------------------------- text helpers

def clean(text: str | None) -> str:
    """Drop [src:...] / [inferred] markers and collapse whitespace."""
    return re.sub(r"\s+", " ", MARKER.sub("", text or "")).strip()


def _norm(text: str) -> str:
    s = clean(text).lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def quote_in(quote: str, text: str) -> bool:
    """True when the quote (pieces split on an ellipsis, in order) occurs verbatim in text."""
    q = clean(quote)
    words = len(q.split())
    if words < MIN_QUOTE_WORDS or words > MAX_QUOTE_WORDS + 5:
        return False
    hay = f" {_norm(text)} "
    pos = 0
    for piece in re.split(r"…|\.\.\.", q):
        n = _norm(piece)
        if not n:
            continue
        i = hay.find(f" {n} ", pos)
        if i == -1:
            return False
        pos = i + len(n) + 1
    return True


def trim_quote(text: str, n: int = MAX_QUOTE_WORDS) -> str:
    q = " ".join(clean(text).split()[:n]).rstrip(" ,;:")
    return q.strip(" \"'“”‘’").strip("…").strip(" \"'“”‘’")


def best_sentence(text: str, against: str = "", avoid: tuple[str, ...] = ()) -> str:
    """The sentence of a turn that best carries its reasoning (reason words, engagement, length)."""
    sents = [s for s in sentences(clean(text)) if len(s.split()) >= 5] or [clean(text)]

    def score(s: str) -> float:
        return (bool(REASON_WORDS.search(s)) + overlap(s, against) + min(len(s.split()), 25) / 50
                - 2 * any(a and clean(a) in s for a in avoid))

    return trim_quote(max(sents, key=score)) if sents and sents[0] else ""


# ---------------------------------------------------------------- event fold (debates and duels)

def fold_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Turns, claims (verified / inferred / dropped), rounds, fallacies, verdict and jury from an event log."""
    turns: list[dict[str, Any]] = []
    claims: dict[str, dict[str, Any]] = {}
    claim_meta: dict[str, tuple[dict[str, Any], int]] = {}
    rounds: dict[int, dict[str, Any]] = {}
    duel_rounds: dict[int, list[dict[str, Any]]] = {}
    fallacies: list[dict[str, Any]] = []
    out: dict[str, Any] = {"verdict": None, "territory": None, "learner": None, "jury": None, "error": None}
    passages: list[dict[str, Any]] = []

    def last_turn(speaker: str, round_no: int | None = None) -> dict[str, Any] | None:
        for t in reversed(turns):
            if t["speaker"] == speaker and (round_no is None or t["round_no"] == round_no):
                return t
        return None

    for e in sorted(events, key=lambda x: x.get("seq") or 0):
        ev, data = e.get("event"), e.get("data") or {}
        if ev == "retrieval.passages":
            passages = list(data.get("passages") or [])
        elif ev == "persona.completed":
            idx = len(turns)
            turns.append({"index": idx, "speaker": "guru", "stage": data.get("stage"), "round_no": data.get("round_no") or 0,
                          "text": data.get("text") or "", "method": data.get("method"), "claims": list(data.get("claims") or []),
                          "passages": passages})
            passages = []
            for c in data.get("claims") or []:
                claim_meta[c["claim_id"]] = (c, idx)
        elif ev == "user.submitted":
            turns.append({"index": len(turns), "speaker": "shishya", "stage": "khandana", "round_no": data.get("round_no") or 0,
                          "text": data.get("text") or "", "analysis": None, "fallacies": []})
        elif ev in ("claim.verified", "claim.dropped"):
            cid = data.get("claim_id")
            if not cid:
                continue
            meta, idx = claim_meta.get(cid, ({}, turns[-1]["index"] if turns else -1))
            t = turns[idx] if 0 <= idx < len(turns) else None
            claims[cid] = {
                "claim_id": cid, "turn_index": idx, "round_no": t["round_no"] if t else 0, "stage": t["stage"] if t else None,
                "text": data.get("text") or meta.get("text") or "",
                "status": "dropped" if ev == "claim.dropped" else data.get("status"),
                "source_ids": list(data.get("source_ids") or meta.get("source_ids") or []),
                "nli_label": data.get("nli_label"), "nli_prob": data.get("nli_prob"), "support": data.get("support"),
                "method": data.get("method"),
            }
        elif ev == "analysis.completed":
            t = last_turn("shishya", data.get("round_no"))
            if t is not None:
                t["analysis"] = data
        elif ev == "fallacy.flagged":
            fallacies.append(dict(data))
            t = last_turn("shishya", data.get("round_no"))
            if t is not None:
                t["fallacies"].append(dict(data))
        elif ev == "score.updated":
            rounds[int(data.get("round_no") or 0)] = dict(data)
        elif ev == "round.scored":
            r = int(data.get("round_no") or 0)
            duel_rounds[r] = list(data.get("results") or [])
            for res in duel_rounds[r]:
                for f in res.get("fallacies") or []:
                    fallacies.append(dict(f) | {"round_no": r, "player_id": res.get("player_id")})
                if res.get("text"):
                    turns.append({"index": len(turns), "speaker": res.get("player_id"), "stage": "round", "round_no": r,
                                  "text": res["text"], "analysis": {"structure": res.get("structure") or {}, "pramanas": [
                                      {"pramana": p, "span": ""} for p in res.get("pramanas") or []]},
                                  "fallacies": [dict(f) | {"round_no": r} for f in res.get("fallacies") or []]})
        elif ev in ("debate.verdict", "match.verdict"):
            out["verdict"] = dict(data)
        elif ev in ("territory.changed", "territory.unchanged"):
            out["territory"] = {"changed": ev == "territory.changed", **data}
        elif ev == "learner.updated":
            out["learner"] = dict(data)
        elif ev == "jury.conclusion":
            out["jury"] = dict(data)
        elif ev in ("debate.error", "match.error"):
            out["error"] = data.get("message")
    return out | {"turns": turns, "claims": list(claims.values()), "rounds": rounds, "duel_rounds": duel_rounds,
                  "fallacies": fallacies}


def claim_counts(claims: list[dict[str, Any]]) -> dict[str, int]:
    c = {"verified": 0, "inferred": 0, "dropped": 0}
    for x in claims:
        if x.get("status") in c:
            c[x["status"]] += 1
    return c | {"total": sum(c.values())}


# ---------------------------------------------------------------- jury input

@dataclass
class SideInput:
    key: str
    label: str
    role: str
    turns: list[str]
    good: list[dict[str, str]] = field(default_factory=list)  # fallback affirmations {point, quote}
    bad: list[dict[str, str]] = field(default_factory=list)   # fallback concessions

    def corpus(self) -> str:
        return " \n ".join(clean(t) for t in self.turns)


@dataclass
class JuryInput:
    kind: str  # debate | duel | rivalry
    question: str
    sides: list[SideInput]
    transcript: list[tuple[str, str, str]]  # (side key, stage, text)
    outcome: str
    summary: str
    decisive: str
    reason: str
    winner: str | None = None


def _source_label(source_ids: list[str]) -> str:
    from ..knowledge import get_knowledge

    for sid in source_ids:
        c = get_knowledge().chunk(sid)
        if c:
            return f"{c.work.split('(')[0].strip()}, {c.section}"
    return "the cited passage"


def _guru_points(claims: list[dict[str, Any]], user_won: bool | None = None) -> tuple[list[dict], list[dict]]:
    verified = sorted([c for c in claims if c.get("status") == "verified"], key=lambda c: -(c.get("nli_prob") or 0))
    good = [{"point": f"Verified against {_source_label(c['source_ids'])}.", "quote": trim_quote(c["text"])}
            for c in verified]
    bad = [{"point": "Went beyond the text here: an interpretation, marked inferred.", "quote": trim_quote(c["text"])}
           for c in claims if c.get("status") == "inferred"]
    bad += [{"point": f"Withdrew a claim its source did not support: “{trim_quote(c['text'], 12)}…”", "quote": ""}
            for c in claims if c.get("status") == "dropped"]
    if user_won:
        bad.insert(0, {"point": "Did not hold the position: the shishya's refutation cleared the bar.", "quote": ""})
    return good, bad


def _rubric_points(rounds: list[tuple[int, str, dict[str, Any], str]]) -> tuple[list[dict], list[dict]]:
    """rounds: (round_no, text, score record with breakdown/reasons/round_score, opponent text)."""
    good, bad = [], []
    for r, text, rec, against in sorted(rounds, key=lambda x: -(x[2].get("round_score") or 0)):
        b = rec.get("breakdown") or {}
        reasons = rec.get("reasons") or {}
        if not b or not text:
            continue
        top = max(COMPONENTS, key=lambda k: b.get(k, 0))
        spans = tuple(f.get("span", "") for f in rec.get("fallacies") or [])
        reason = reasons.get(top) or ""
        good.append({"point": f"Round {r}: strongest on {COMPONENT_LABEL[top]} ({round(100 * b.get(top, 0))}%). {reason}".strip(),
                     "quote": best_sentence(text, against, spans)})
    for r, text, rec, _ in sorted(rounds, key=lambda x: x[0]):
        for f in rec.get("fallacies") or []:
            bad.append({"point": f"Round {r}: {FALLACY_NAME.get(f.get('type'), f.get('type'))}. {f.get('explanation', '')}".strip(),
                        "quote": trim_quote(f.get("span", ""))})
        b = rec.get("breakdown") or {}
        if b:
            low = min(COMPONENTS, key=lambda k: b.get(k, 1))
            if b.get(low, 1) < 0.6:
                bad.append({"point": f"Round {r}: weakest on {COMPONENT_LABEL[low]} ({round(100 * b.get(low, 0))}%). "
                                     f"{(rec.get('reasons') or {}).get(low, '')}".strip(), "quote": ""})
    return good, bad


def debate_input(fold: dict[str, Any], persona_name: str, question: str, threshold: float) -> JuryInput:
    turns = fold["turns"]
    verdict = fold.get("verdict") or {}
    guru_turns = [t for t in turns if t["speaker"] == "guru"]
    user_turns = [t for t in turns if t["speaker"] == "shishya"]
    fall_by_round: dict[int, list] = {}
    for f in fold["fallacies"]:
        fall_by_round.setdefault(int(f.get("round_no") or 0), []).append(f)
    scored = []
    for t in user_turns:
        rec = dict(fold["rounds"].get(t["round_no"]) or {})
        rec["fallacies"] = fall_by_round.get(t["round_no"], [])
        before = [g["text"] for g in guru_turns if g["index"] < t["index"]]
        scored.append((t["round_no"], t["text"], rec, before[-1] if before else ""))
    u_good, u_bad = _rubric_points(scored)
    won = verdict.get("user_won")
    g_good, g_bad = _guru_points(fold["claims"], won)
    for r, _, rec, _ in scored:  # rounds the shishya won are the guru's concessions
        if (rec.get("round_score") or 0) >= threshold:
            g_bad.append({"point": f"Round {r}: the shishya's refutation scored {rec['round_score']:.0f}, above the bar; "
                                   "the reply did not undo it.", "quote": ""})
    if won is False:
        u_bad.insert(0, {"point": f"Fell short of the bar: final score {verdict.get('final_score', 0):.0f} against {threshold:.0f}.", "quote": ""})
    sides = [
        SideInput("guru", persona_name, "the guru, defending his thesis from his own texts", [t["text"] for t in guru_turns], g_good, g_bad),
        SideInput("shishya", "The shishya", "the student, refuting the guru's thesis", [t["text"] for t in user_turns], u_good, u_bad),
    ]
    counts = claim_counts(fold["claims"])
    if verdict:
        winner = "shishya" if won else "guru"
        crowd = verdict.get("crowd_score")
        aud = f", audience {crowd}" if crowd is not None else ", no audience votes"
        outcome = (f"{'The shishya' if won else persona_name} won. The shishya's final score was {verdict.get('final_score')} "
                   f"against a threshold of {threshold} (judge {verdict.get('ai_score')}{aud}).")
        summary = (f"On “{question}”, the shishya scored {verdict.get('final_score', 0):.0f} against a bar of {threshold:.0f}, so "
                   + ("the refutation stands." if won else f"{persona_name}'s thesis holds.")
                   + f" The guru made {counts['total']} checked claims: {counts['verified']} verified, "
                   f"{counts['inferred']} inferred, {counts['dropped']} withdrawn.")
        reason = (f"Final {verdict.get('final_score')} = 0.7 × judge {verdict.get('ai_score')} + 0.3 × audience {crowd} "
                  f"({verdict.get('crowd_votes', 0)} votes); the shishya needed {threshold}." if crowd is not None else
                  f"Judge score {verdict.get('ai_score')} (mean of the round scores {verdict.get('round_scores')}) against the "
                  f"threshold {threshold}; no audience votes.")
    else:
        winner, outcome = None, "The debate has not reached a verdict yet."
        summary, reason = f"On “{question}”, the debate is still in progress.", "No verdict yet."
    decisive = "No round was scored."
    if scored:
        pick = (max if won else min)(scored, key=lambda x: x[2].get("round_score") or 0)
        r, _, rec, _ = pick
        decisive = f"Round {r}: the shishya's argument scored {rec.get('round_score', 0):.0f} against the bar of {threshold:.0f}"
        if rec.get("fallacies"):
            decisive += f", with {FALLACY_NAME.get(rec['fallacies'][0].get('type'), 'a fallacy')} flagged"
        decisive += "."
    transcript = [("guru" if t["speaker"] == "guru" else "shishya", t["stage"] or "", t["text"]) for t in turns]
    return JuryInput("debate", question, sides, transcript, outcome, summary, decisive, reason, winner)


def duel_input(fold: dict[str, Any], players: list[dict[str, Any]], persona_name: str, question: str,
               reasons: dict[tuple[str, int], dict[str, str]] | None = None) -> JuryInput:
    reasons = reasons or {}
    verdict = fold.get("verdict") or {}
    sides = []
    for p in players:
        pid = p["player_id"]
        other = next((q["player_id"] for q in players if q["player_id"] != pid), None)
        scored = []
        for r, results in sorted(fold["duel_rounds"].items()):
            mine = next((x for x in results if x.get("player_id") == pid), None)
            theirs = next((x for x in results if x.get("player_id") == other), None)
            if mine:
                rec = dict(mine) | {"reasons": reasons.get((pid, r)) or {}}
                scored.append((r, mine.get("text") or "", rec, (theirs or {}).get("text") or ""))
        good, bad = _rubric_points(scored)
        if verdict.get("winner") and verdict["winner"] != pid:
            bad.insert(0, {"point": f"Lost on the judge's average: {verdict['scores'].get(pid, 0):.0f} against "
                                    f"{verdict['scores'].get(other, 0):.0f}.", "quote": ""})
        role = "defends the guru's thesis" if p.get("side") == "for" else "attacks the guru's thesis"
        sides.append(SideInput(pid, p.get("handle") or pid, role, [t["text"] for t in fold["turns"] if t["speaker"] == pid], good, bad))
    label = {p["player_id"]: p.get("handle") for p in players}
    if verdict:
        w = verdict.get("winner")
        outcome = f"{label.get(w, 'Nobody')} won on the judge's average score {verdict.get('scores')}." if w else "A draw."
        summary = (f"On “{question}”, moderated by {persona_name}, " +
                   (f"{label.get(w)} won with the higher average score." if w else "the judge scored both players level."))
        reason = "The same judge scores both arguments on the five-part rubric; the higher average across rounds wins."
    else:
        outcome, summary, reason, w = "No verdict yet.", f"On “{question}”, the duel is still in progress.", "No verdict yet.", None
    best = max(((r, x) for r, xs in fold["duel_rounds"].items() for x in xs), key=lambda rx: rx[1].get("round_score") or 0, default=None)
    decisive = (f"Round {best[0]}: {label.get(best[1].get('player_id'))} scored {best[1].get('round_score', 0):.0f}, the best "
                f"argument of the duel." if best else "No round was scored.")
    transcript = [(t["speaker"], t["stage"] or "", t["text"]) for t in fold["turns"]]
    return JuryInput("duel", question, sides, transcript, outcome, summary, decisive, reason, w)


def rivalry_input(bout: Any) -> JuryInput:
    a, b = bout.a, bout.b
    verdict = bout.verdict or {}
    integ = bout.integrity()
    sides = []
    for p, role in ((a, "states the thesis from his own texts"), (b, "answers from his own texts")):
        turns = [t for t in bout.turns if t["persona_id"] == p.id]
        claims = [v for t in turns for v in t.get("verified", [])] + [v | {"status": "dropped"} for t in turns for v in t.get("dropped", [])]
        good, bad = _guru_points(claims)
        sides.append(SideInput(p.id, p.name, role, [t["text"] for t in turns], good, bad))
    votes = verdict.get("votes") or {}
    w = verdict.get("winner")
    names = {a.id: a.name, b.id: b.name}
    ia, ib = integ.get(a.id, {}), integ.get(b.id, {})
    summary = (f"{a.name} and {b.name} argued “{bout.topic.question}”. {a.name} verified {ia.get('verified', 0)} of "
               f"{ia.get('claims', 0)} claims against his texts, {b.name} {ib.get('verified', 0)} of {ib.get('claims', 0)}.")
    if verdict:
        outcome = (f"The audience chose {names[w]} ({votes.get('a', 0)} to {votes.get('b', 0)})." if w else
                   f"The audience split evenly or did not vote ({verdict.get('voters', 0)} votes).")
        reason = "In a Rivalry Sabha the audience decides; the verification count is the objective half of the record."
    else:
        outcome, reason = "The Sabha has not been closed yet.", "Votes are still open."
    best = max(bout.turns, key=lambda t: sum(v.get("status") == "verified" for v in t.get("verified", [])), default=None)
    decisive = (f"Turn {best['n'] + 1}: {names[best['persona_id']]} ({best['stage']}) carried "
                f"{sum(v.get('status') == 'verified' for v in best.get('verified', []))} verified citations." if best else "No turns.")
    transcript = [(t["persona_id"], t["stage"], t["text"]) for t in bout.turns]
    return JuryInput("rivalry", bout.topic.question, sides, transcript, outcome, summary, decisive, reason, w)


# ---------------------------------------------------------------- conclusion

class JuryPoint(BaseModel):
    point: str = ""
    quote: str = ""


class JurySideOut(BaseModel):
    side: str
    affirmations: list[JuryPoint] = []
    concessions: list[JuryPoint] = []
    strongest_point: str = ""


class JuryOut(BaseModel):
    summary: str
    sides: list[JurySideOut]
    decisive_moment: str = ""
    verdict_reason: str = ""


def _grounded(points: list[dict[str, str]], side: SideInput, need_quote: bool, limit: int = 3) -> list[dict[str, str]]:
    corpus, out, seen = side.corpus(), [], set()
    for p in points:
        point = re.sub(r"\s+", " ", p.get("point") or "").strip()[:300]
        quote = trim_quote(p.get("quote") or "", MAX_QUOTE_WORDS + 5)
        if quote and not quote_in(quote, corpus):
            quote = ""
        if (need_quote and not quote) or not point or (quote or point) in seen:
            continue
        seen.add(quote or point)
        out.append({"point": point, "quote": quote})
        if len(out) >= limit:
            break
    return out


def _result(inp: JuryInput, method: str, summary: str, sides: list[dict], decisive: str, reason: str) -> dict[str, Any]:
    return {"kind": inp.kind, "method": method, "summary": summary, "sides": sides, "decisive_moment": decisive,
            "verdict_reason": reason, "outcome": inp.outcome, "winner": inp.winner, "created_at": time.time()}


def fallback(inp: JuryInput) -> dict[str, Any]:
    """Deterministic jury from the scores and the verifier; never calls the LLM."""
    sides = []
    for s in inp.sides:
        aff = _grounded(s.good, s, need_quote=True)
        con = _grounded(s.bad, s, need_quote=False)
        sides.append({"side": s.key, "label": s.label, "role": s.role, "affirmations": aff, "concessions": con,
                      "strongest_point": aff[0]["point"] if aff else ""})
    return _result(inp, "rules", inp.summary, sides, inp.decisive, inp.reason)


def merge(inp: JuryInput, out: JuryOut, base: dict[str, Any]) -> dict[str, Any]:
    """Validate the LLM's jury against the transcript; sides it got wrong fall back to the rules jury."""
    by_key = {s.key: s for s in inp.sides}
    by_label = {s.label.lower(): s for s in inp.sides}
    got: dict[str, dict[str, Any]] = {}
    for so in out.sides:
        s = by_key.get(so.side) or by_label.get(so.side.lower())
        if s is None or s.key in got:
            continue
        aff = _grounded([a.model_dump() for a in so.affirmations], s, need_quote=True)
        con = _grounded([c.model_dump() for c in so.concessions], s, need_quote=False)
        got[s.key] = {"side": s.key, "label": s.label, "role": s.role, "affirmations": aff, "concessions": con,
                      "strongest_point": clean(so.strongest_point)[:300] or (aff[0]["point"] if aff else "")}
    sides = []
    for s, b in zip(inp.sides, base["sides"]):
        g = got.get(s.key)
        if g is None:
            g = b
        elif not g["affirmations"]:
            g = g | {"affirmations": b["affirmations"]}  # every affirmation stays grounded in a real quote
        sides.append(g)
    return _result(inp, "llm", clean(out.summary)[:800] or inp.summary, sides,
                   clean(out.decisive_moment)[:400] or inp.decisive, clean(out.verdict_reason)[:400] or inp.reason)


SYSTEM = ("You are the jury of a shastrartha, a structured classical Indian debate. You are neutral and fair to both "
          "sides, and you judge reasoning, not people. Return JSON only. ")
KIND_DESC = {"debate": "a student (shishya) refutes an AI interpretation of a historical guru",
             "duel": "two students argue a present-day question; a guru moderates",
             "rivalry": "two AI interpretations of historical gurus argue one question from their own texts"}
PROMPT = """KIND: {kind}
QUESTION: {question}
SIDES:
{sides}
OUTCOME (already decided by the scores; explain it, do not change it): {outcome}

TRANSCRIPT (each turn starts with its side key):
{transcript}

Write the jury's conclusion.
- summary: 2-3 neutral sentences on how the argument went.
- For EACH side key: up to 3 "affirmations" (what that side got right) and up to 2 "concessions" (what it failed to answer, left unproven or got wrong). Each item: "point" (one sentence) and "quote": an exact phrase of 6-20 words copied from THAT side's own turns, without citation markers. Leave "quote" empty if you cannot copy one exactly.
- strongest_point: one sentence per side.
- decisive_moment: one sentence naming the turn that decided it.
- verdict_reason: one sentence explaining the outcome.
Rules: use only what is in the transcript; do not name or judge living people or political parties; no religious rulings.
Return JSON: {{"summary": "...", "sides": [{{"side": "<side key>", "affirmations": [{{"point": "...", "quote": "..."}}], "concessions": [{{"point": "...", "quote": "..."}}], "strongest_point": "..."}}], "decisive_moment": "...", "verdict_reason": "..."}}"""


def _prompt(inp: JuryInput) -> str:
    sides = "\n".join(f'- key "{s.key}": {s.label}, {s.role}' for s in inp.sides)
    lines = "\n".join(f"[{k}] ({stage}): {clean(text)[:1400]}" for k, stage, text in inp.transcript if text)
    return PROMPT.format(kind=KIND_DESC.get(inp.kind, inp.kind), question=inp.question, sides=sides, outcome=inp.outcome,
                         transcript=lines[:9000])


async def conclude(inp: JuryInput, use_llm: bool | None = None) -> dict[str, Any]:
    """One LLM call, validated; the rules jury when the LLM is off or fails. Never raises."""
    try:
        base = fallback(inp)
    except Exception:  # noqa: BLE001
        log.exception("Jury fallback failed")
        base = _result(inp, "rules", inp.summary, [{"side": s.key, "label": s.label, "role": s.role, "affirmations": [],
                                                    "concessions": [], "strongest_point": ""} for s in inp.sides],
                       inp.decisive, inp.reason)
    if use_llm is None:
        use_llm = get_settings().llm_available
    if not use_llm or not inp.transcript:
        return base
    from ..agents.llm import SAFETY, LLMUnavailable, complete_json

    try:
        out = await complete_json(SYSTEM + SAFETY, _prompt(inp), JuryOut, max_tokens=1100, cap=False)
        return merge(inp, out, base)
    except LLMUnavailable as exc:
        log.info("Jury falling back to rules: %s", exc)
    except Exception:  # noqa: BLE001
        log.exception("Jury LLM call failed")
    return base


# ---------------------------------------------------------------- background scheduling

_pending: set[str] = set()
_tasks: set[asyncio.Task] = set()


def pending(key: str) -> bool:
    return key in _pending


def run_later(key: str, make: Callable[[], Awaitable[dict[str, Any]]], on_done: Callable[[dict[str, Any]], Any]) -> bool:
    """Run `make()` in the background once per key (event loop if there is one, else a daemon thread)."""
    if key in _pending:
        return False
    _pending.add(key)

    async def go() -> None:
        result = None
        try:
            result = await make()
        except Exception:  # noqa: BLE001
            log.exception("Jury for %s failed", key)
        finally:
            _pending.discard(key)
        if result is not None:
            try:
                on_done(result)
            except Exception:  # noqa: BLE001
                log.exception("Jury callback for %s failed", key)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        task = loop.create_task(go())
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    else:
        threading.Thread(target=lambda: asyncio.run(go()), daemon=True, name=f"jury-{key[:8]}").start()
    return True
