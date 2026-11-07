"""Headline -> debate: turn a news headline into a playable modern topic, grounded like the curated ones.

Pipeline (two Gemini calls, then deterministic checks):
1. FRAME: the headline becomes a neutral yes/no debate question, a short briefing that restates only what the
   headline says, a retrieval query, and 1-2 Wikipedia article titles for background. Questions about
   religion, communities, or a named living person are refused.
2. GROUND: each playable guru's best passages for the question are retrieved from his own texts (BM25 + dense).
   Gemini picks the guru and passage that speak to it best and restates that passage as the stance, plus up
   to two counter-points from the same pool.
3. CHECK: the stance and each counter-point are run through the claim verifier against their passage; the
   stance must be "verified" or the topic is rejected. The question must not name a party or leader.

Accepted topics are stored in data/today/generated.json (with the source headline) and registered at once;
their background articles are fetched into the Today desk. Same headline twice -> same topic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .agents.llm import LLMUnavailable, complete_json
from .config import get_settings
from .knowledge import PLAYABLE, get_knowledge
from .rag.retrieve import retrieve
from .rag.verify import verify_claim
from .schemas import Claim, Stance, SupportPoint, Topic
from . import today

log = logging.getLogger(__name__)
MAX_GENERATED = 40
_lock = threading.Lock()


class HeadlineError(Exception):
    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class _Frame(BaseModel):
    ok: bool = True
    reason: str | None = ""
    question: str | None = ""
    scope: str | None = "world"
    briefing: str | None = ""
    query: str | None = ""
    wiki: list[str] = []


class _Point(BaseModel):
    chunk_id: str
    angle: str = ""


class _Ground(BaseModel):
    persona_id: str
    chunk_id: str
    statement: str
    support: list[_Point] = Field(default_factory=list)


FRAME_SYSTEM = """You turn a news headline into a question for a student debate game where classical thinkers
(Chanakya, Sun Tzu, Napoleon, Marcus Aurelius) argue present-day policy from their own texts. Return JSON only.
Rules:
- question: one neutral yes/no question about the POLICY or STRATEGY the headline raises, at most 16 words,
  ending with "?". Name no person, party or government; countries and institutions are fine.
- scope: "india" if it is about Indian public policy, otherwise "world".
- briefing: two or three neutral sentences. Restate only what the headline says, then why the question is
  debated (the case for and the case against). Add no numbers, dates or names that are not in the headline.
- query: 6-12 words describing the underlying classical theme (e.g. "alliances neutrality balance between rival powers").
- wiki: one or two existing English Wikipedia article titles for neutral background on the policy (not on a person).
- Set ok=false with a short reason if the only honest question would be about religion, caste or community,
  about a named living person, about a crime or court case against an individual, or about cheering a side in
  an ongoing war. Sport, celebrity and entertainment news: ok=false.
Shape: {"ok": true, "reason": "", "question": "...?", "scope": "india", "briefing": "...", "query": "...", "wiki": ["..."]}
- Courts, commissions or parliament weighing a PUBLIC POLICY (e.g. a court hearing on election freebies) are
  fine: debate the policy, not the case. Opinion pieces and exam-style essay prompts are fine too if a clear
  policy question sits underneath."""

GROUND_SYSTEM = """You choose which classical thinker's own text best answers a modern debate question. Return JSON only.
You are given candidate PASSAGES grouped by thinker; each has an id. Pick the ONE passage whose doctrine most
directly bears on the question, and its thinker.
- statement: one sentence that restates what that passage itself says (close to its wording, no modern terms,
  no claims it does not make). This is the thinker's thesis for the debate.
- support: up to two other passages from the SAME thinker that a student could be answered with, each with a
  one-sentence angle that restates what that passage says.
Use only ids that appear in PASSAGES. Use exactly these keys:
{"persona_id": "chanakya", "chunk_id": "<id>", "statement": "...", "support": [{"chunk_id": "<id>", "angle": "..."}]}"""


def close_to_text(statement: str, passage: str) -> bool:
    """Generated theses must also keep the passage's own wording (lexical support >= 0.6), on top of the NLI check,
    so a thesis verifies whether or not the NLI model is loaded."""
    from .rag.text import best_sentence, overlap

    return max(overlap(statement, passage), overlap(statement, best_sentence(passage, statement))) >= 0.6


def _path() -> Path:
    return get_settings().data_dir / "today" / "generated.json"


def load_generated() -> dict[str, dict]:
    p = _path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(entries: dict[str, dict]) -> None:
    p = _path()
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def topic_id_for(title: str) -> str:
    return "news-" + hashlib.sha1(re.sub(r"\W+", " ", title.lower()).strip().encode()).hexdigest()[:8]


def _candidates(question: str, query: str) -> dict[str, list]:
    k = get_knowledge()
    out = {}
    for pid in PLAYABLE:
        hits = retrieve(pid, f"{question} {query}", k=4)
        out[pid] = [h for h in hits if h.chunk.source_type in ("primary", "account") or h.chunk.chunk_id in k.chunks][:4]
    return out


async def from_headline(title: str, url: str = "", outlet: str = "", date: str = "") -> dict:
    title = re.sub(r"\s+", " ", title).strip()
    if not 12 <= len(title) <= 300:
        raise HeadlineError("VALIDATION_ERROR", "Give a headline between 12 and 300 characters.")
    tid = topic_id_for(title)
    k = get_knowledge()
    if tid in k.topics:
        return {"topic": k.topics[tid].model_dump(), "created": False}
    if not get_settings().llm_available:
        raise HeadlineError("LLM_UNAVAILABLE", "Turning headlines into debates needs the LLM.", 503)
    if len(load_generated()) >= MAX_GENERATED:
        raise HeadlineError("VALIDATION_ERROR", "The news desk is full; remove old generated topics first.", 409)

    try:
        frame = await complete_json(FRAME_SYSTEM, f"HEADLINE: {title}\nOUTLET: {outlet or 'unknown'}", _Frame, max_tokens=500)
    except LLMUnavailable as exc:
        raise HeadlineError("LLM_UNAVAILABLE", str(exc), 503) from exc
    if not frame.ok:
        raise HeadlineError("NOT_DEBATABLE", frame.reason or "This headline is not one the gurus can debate.")
    q = (frame.question or "").strip()
    frame.briefing, frame.query, frame.scope = frame.briefing or "", frame.query or "", frame.scope or "world"
    if not q.endswith("?") or len(q.split()) > 22 or today._POLITICAL.search(q):
        raise HeadlineError("NOT_DEBATABLE", "Could not frame a neutral question from this headline.")

    cands = _candidates(q, frame.query)
    allowed = {h.chunk.chunk_id: (pid, h.chunk) for pid, hs in cands.items() for h in hs}
    block = "\n".join(f"[{pid}]\n" + "\n".join(f"  ({h.chunk.chunk_id}) {h.chunk.text[:600]}" for h in hs) for pid, hs in cands.items())
    try:
        g = await complete_json(GROUND_SYSTEM, f"QUESTION: {q}\nPASSAGES:\n{block}", _Ground, max_tokens=600)
    except LLMUnavailable as exc:
        raise HeadlineError("LLM_UNAVAILABLE", str(exc), 503) from exc
    if g.chunk_id not in allowed or allowed[g.chunk_id][0] != g.persona_id:
        raise HeadlineError("NOT_GROUNDED", "No guru's text could be tied to this question.")

    v = verify_claim(Claim(claim_id=f"{tid}_stance", text=g.statement, source_ids=[g.chunk_id]))
    if v.status != "verified" or not close_to_text(g.statement, allowed[g.chunk_id][1].text):
        raise HeadlineError("NOT_GROUNDED", "The proposed thesis did not verify against the guru's text.")
    support = []
    for p in g.support[:2]:
        if p.chunk_id in allowed and allowed[p.chunk_id][0] == g.persona_id and p.chunk_id != g.chunk_id:
            if p.angle and close_to_text(p.angle, allowed[p.chunk_id][1].text) and verify_claim(Claim(claim_id=f"{tid}_s", text=p.angle, source_ids=[p.chunk_id])).status == "verified":
                support.append(SupportPoint(chunk_id=p.chunk_id, angle=p.angle))

    # The topic's stance and support must resolve as curated chunks (library passages are registered on retrieval).
    for cid in [g.chunk_id, *(s.chunk_id for s in support)]:
        k.register(allowed[cid][1])

    topic = Topic(
        id=tid, persona_id=g.persona_id, question=q, concept_id=f"news.{tid}", era="modern",
        scope="india" if frame.scope == "india" else "world", context=frame.briefing.strip(),
        stance=Stance(statement=g.statement, source_id=g.chunk_id), support=support, handbook_ref="headline",
    )
    entry = {"topic": topic.model_dump(), "headline": {"title": title, "url": url, "outlet": outlet, "date": date},
             "wiki": frame.wiki[:2], "stance_chunk": allowed[g.chunk_id][1].model_dump(),
             "support_chunks": [allowed[s.chunk_id][1].model_dump() for s in support],
             "verified": {"stance": v.nli_prob, "method": v.method}, "created_at": time.time()}
    with _lock:
        entries = load_generated()
        entries[tid] = entry
        _save(entries)
    k.topics[tid] = topic
    threading.Thread(target=_fetch_background, args=(tid, frame.wiki[:2]), daemon=True).start()
    log.info("Headline topic %s -> %s on %s (stance %s)", tid, g.persona_id, g.chunk_id, v.nli_prob)
    return {"topic": topic.model_dump(), "created": True}


def _fetch_background(tid: str, titles: list[str]) -> None:
    """Wikipedia background for a generated topic, added to the Today desk (and its cache file)."""
    k = get_knowledge()
    got = []
    for t in titles:
        try:
            got += today.fetch_wiki(t)
        except Exception as exc:  # noqa: BLE001
            log.info("Background '%s' for %s skipped: %s", t, tid, exc)
    if not got:
        return
    known = {c.chunk_id for c in k.today}
    new = [c for c in got if c.chunk_id not in known]
    k.today += new
    for c in new:
        k.chunks.setdefault(c.chunk_id, c)
    path = get_settings().data_dir / "today" / "wiki.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.writelines(c.model_dump_json() + "\n" for c in new)
    with _lock:
        entries = load_generated()
        if tid in entries:
            entries[tid]["wiki"] = [c.work.removeprefix("Wikipedia: ") for c in got if c.chunk_id.endswith(":01")] or entries[tid]["wiki"]
            _save(entries)


def remove(tid: str) -> bool:
    with _lock:
        entries = load_generated()
        if tid not in entries:
            return False
        del entries[tid]
        _save(entries)
    get_knowledge().topics.pop(tid, None)
    return True
