"""Study roadmaps (Read tab): curriculum loading, progress, quiz scoring, study level and the "Tested" seal.

Curricula are built once by `python -m scripts.build_curriculum` into data/curriculum/<persona>.json. Those files
hold the quiz answers; `public_view` strips them, and quiz results reveal an answer only once it was answered
correctly or the module is passed.

Progress per (user, persona): lessons marked complete, best quiz score and attempts per module. A module is passed
at >= PASS_MARK; module N opens when module N-1 is passed. Study level = passed modules; "tested" = all passed.
Guests ("demo" and any non-uuid id) live in memory; accounts write through to Supabase `study_progress`
(supabase/schema.sql, v3 study block). If that table is missing, reads return nothing and writes only log warnings.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

from ..config import get_settings
from ..db import supabase as db
from ..knowledge import PLAYABLE, get_knowledge
from .store import is_account

log = logging.getLogger(__name__)

PASS_MARK = 0.75
TABLE = "study_progress"


class StudyError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


# ---------------------------------------------------------------- curriculum

@lru_cache
def curriculum(persona_id: str) -> dict | None:
    if persona_id not in PLAYABLE:
        return None
    path = get_settings().data_dir / "curriculum" / f"{persona_id}.json"
    if not path.exists():
        log.warning("No curriculum for %s; run: python -m scripts.build_curriculum", persona_id)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _module(cur: dict, module_id: str) -> tuple[int, dict]:
    for i, m in enumerate(cur["modules"]):
        if m["id"] == module_id:
            return i, m
    raise StudyError("NOT_FOUND", "Unknown module", 404)


def _lesson(cur: dict, lesson_id: str) -> tuple[int, dict]:
    for i, m in enumerate(cur["modules"]):
        for les in m["lessons"]:
            if les["id"] == lesson_id:
                return i, les
    raise StudyError("NOT_FOUND", "Unknown lesson", 404)


# ---------------------------------------------------------------- progress store

@dataclass
class Progress:
    lessons_done: set[str] = field(default_factory=set)
    quiz_best: dict[str, float] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)


class StudyStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], Progress] = {}
        self.loaded: set[str] = set()
        self.lock = threading.Lock()

    def get(self, user_id: str, persona_id: str) -> Progress:
        with self.lock:
            if user_id not in self.loaded:
                self.loaded.add(user_id)
                if is_account(user_id):
                    self._load(user_id)
            return self.rows.setdefault((user_id, persona_id), Progress())

    def _load(self, user_id: str) -> None:
        for r in db.select(TABLE, f"user_id=eq.{user_id}&select=persona_id,lessons_done,quiz_best,attempts"):
            self.rows[(user_id, r["persona_id"])] = Progress(
                set(r.get("lessons_done") or []),
                {k: float(v) for k, v in (r.get("quiz_best") or {}).items()},
                {k: int(v) for k, v in (r.get("attempts") or {}).items()},
            )

    def save(self, user_id: str, persona_id: str, p: Progress) -> None:
        if not is_account(user_id):
            return
        cert = _cert(curriculum(persona_id), p)
        db.upsert(TABLE, {
            "user_id": user_id, "persona_id": persona_id, "lessons_done": sorted(p.lessons_done),
            "quiz_best": p.quiz_best, "attempts": p.attempts, "level": cert["level"], "tested": cert["tested"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="user_id,persona_id")

    def reset(self, user_id: str) -> None:
        """Forget a user's in-memory progress (tests, guest reset)."""
        with self.lock:
            self.loaded.discard(user_id)
            for key in [k for k in self.rows if k[0] == user_id]:
                del self.rows[key]


store = StudyStore()


# ---------------------------------------------------------------- derived state

def _passed(m: dict, p: Progress) -> bool:
    return p.quiz_best.get(m["id"], 0.0) >= PASS_MARK


def _statuses(cur: dict, p: Progress) -> list[str]:
    out, prev_passed = [], True
    for m in cur["modules"]:
        passed = _passed(m, p)
        out.append("passed" if passed else "open" if prev_passed else "locked")
        prev_passed = passed
    return out


def _cert(cur: dict | None, p: Progress) -> dict:
    if not cur:
        return {"level": 0, "max": 0, "tested": False}
    level = sum(_passed(m, p) for m in cur["modules"])
    total = len(cur["modules"])
    return {"level": level, "max": total, "tested": total > 0 and level == total}


def certification(user_id: str, persona_id: str) -> dict:
    """{"level": passed modules, "max": modules in the course, "tested": every module passed} for one guru."""
    return _cert(curriculum(persona_id), store.get(user_id, persona_id))


def _progress(cur: dict, p: Progress) -> dict:
    lesson_ids = [les["id"] for m in cur["modules"] for les in m["lessons"]]
    done = sum(1 for lid in lesson_ids if lid in p.lessons_done)
    cert = _cert(cur, p)
    steps = len(lesson_ids) + cert["max"]  # each lesson and each passed quiz is one step of the course
    return {**cert, "lessons_done": done, "lessons_total": len(lesson_ids),
            "percent": round(100 * (done + cert["level"]) / steps) if steps else 0, "pass_mark": PASS_MARK}


def _passage(chunk_id: str) -> dict:
    c = get_knowledge().chunk(chunk_id)
    if c is None:
        return {"chunk_id": chunk_id, "text": "", "work": "", "edition": "", "section": "", "license": "", "url": ""}
    return {"chunk_id": c.chunk_id, "text": c.text, "work": c.work, "edition": c.edition, "section": c.section,
            "license": c.license, "url": c.url}


def public_view(user_id: str, persona_id: str) -> dict:
    """The course without answers, plus the caller's progress."""
    cur = curriculum(persona_id)
    if cur is None:
        raise StudyError("NOT_FOUND", "No study course for this guru", 404)
    p = store.get(user_id, persona_id)
    statuses = _statuses(cur, p)
    modules = []
    for m, status in zip(cur["modules"], statuses):
        lessons = [{
            "id": les["id"], "title": les["title"], "explainer": les["explainer"],
            "explainer_by": les.get("explainer_by", "hand"), "key_terms": les["key_terms"],
            "passages": [_passage(cid) for cid in les["chunk_ids"]], "done": les["id"] in p.lessons_done,
        } for les in m["lessons"]]
        modules.append({
            "id": m["id"], "title": m["title"], "summary": m["summary"], "status": status, "lessons": lessons,
            "lessons_done": sum(les["done"] for les in lessons),
            "quiz": [{"id": q["id"], "question": q["question"], "options": q["options"], "chunk_id": q["chunk_id"]}
                     for q in m["quiz"]],
            "best_score": p.quiz_best.get(m["id"]), "attempts": p.attempts.get(m["id"], 0),
        })
    return {"persona_id": persona_id, "title": cur["title"], "work": cur["work"],
            "generated_by": cur.get("generated_by", "rules"), "modules": modules, "progress": _progress(cur, p)}


def complete_lesson(user_id: str, persona_id: str, lesson_id: str) -> dict:
    cur = curriculum(persona_id)
    if cur is None:
        raise StudyError("NOT_FOUND", "No study course for this guru", 404)
    mi, _ = _lesson(cur, lesson_id)
    p = store.get(user_id, persona_id)
    if _statuses(cur, p)[mi] == "locked":
        raise StudyError("LOCKED", "Pass the previous module's quick check first", 409)
    p.lessons_done.add(lesson_id)
    store.save(user_id, persona_id, p)
    m = cur["modules"][mi]
    return {"lesson_id": lesson_id, "module_id": m["id"],
            "module_lessons_done": sum(les["id"] in p.lessons_done for les in m["lessons"]),
            "module_lessons_total": len(m["lessons"]), "progress": _progress(cur, p)}


def submit_quiz(user_id: str, persona_id: str, module_id: str, answers: list[int]) -> dict:
    cur = curriculum(persona_id)
    if cur is None:
        raise StudyError("NOT_FOUND", "No study course for this guru", 404)
    mi, m = _module(cur, module_id)
    p = store.get(user_id, persona_id)
    if _statuses(cur, p)[mi] == "locked":
        raise StudyError("LOCKED", "Pass the previous module's quick check first", 409)
    missing = [les["title"] for les in m["lessons"] if les["id"] not in p.lessons_done]
    if missing:
        raise StudyError("LESSONS_INCOMPLETE", f"Mark these lessons complete first: {', '.join(missing)}", 409)
    qs = m["quiz"]
    if len(answers) != len(qs) or any(not 0 <= a < len(q["options"]) for a, q in zip(answers, qs)):
        raise StudyError("VALIDATION_ERROR", f"Send one answer index per question ({len(qs)})", 422)

    before = _cert(cur, p)
    right = [a == q["answer"] for a, q in zip(answers, qs)]
    score = round(sum(right) / len(qs), 4)
    p.attempts[module_id] = p.attempts.get(module_id, 0) + 1
    p.quiz_best[module_id] = max(score, p.quiz_best.get(module_id, 0.0))
    store.save(user_id, persona_id, p)
    after = _cert(cur, p)
    passed = score >= PASS_MARK
    reveal = _passed(m, p)  # once the module is passed, every answer is shown

    results = []
    for q, a, ok in zip(qs, answers, right):
        c = get_knowledge().chunk(q["chunk_id"])
        r = {"question_id": q["id"], "chosen": a, "correct": ok, "chunk_id": q["chunk_id"],
             "section": c.section if c else ""}
        if ok or reveal:
            r |= {"answer": q["answer"], "explanation": q["explanation"], "support": q.get("support")}
        results.append(r)
    nxt = cur["modules"][mi + 1]["id"] if passed and mi + 1 < len(cur["modules"]) else None
    return {
        "module_id": module_id, "score": score, "correct": sum(right), "total": len(qs), "passed": passed,
        "best_score": p.quiz_best[module_id], "attempts": p.attempts[module_id], "results": results,
        "level": after["level"], "max": after["max"], "tested": after["tested"],
        "level_up": after["level"] > before["level"], "newly_tested": after["tested"] and not before["tested"],
        "next_module_id": nxt, "progress": _progress(cur, p),
    }


def summary(user_id: str) -> list[dict]:
    """Per-guru study level and seal, for the profile and badges."""
    k = get_knowledge()
    out = []
    for pid in PLAYABLE:
        cur = curriculum(pid)
        if cur is None:
            continue
        prog = _progress(cur, store.get(user_id, pid))
        out.append({"persona_id": pid, "name": k.personas[pid].name, "title": cur["title"], **prog})
    return out
