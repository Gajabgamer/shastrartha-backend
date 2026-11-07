"""Study roadmaps for the Read tab (TASKS.md workstream A).

GET  /api/study/me                                         per-guru study level and Tested seal
GET  /api/study/{persona_id}                               course (no answers) + my progress
POST /api/study/{persona_id}/lessons/{lesson_id}/complete  mark a lesson complete
POST /api/study/{persona_id}/modules/{module_id}/quiz      {answers: [int]} -> score, feedback, level
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..engine import study
from .routes import current_user, error

router = APIRouter(prefix="/api")


class QuizBody(BaseModel):
    answers: list[int] = Field(min_length=1, max_length=10)


def _run(fn, *args):
    try:
        return fn(*args)
    except study.StudyError as exc:
        return error(exc.code, exc.message, exc.status)


@router.get("/study/me")
def my_study(request: Request):
    uid = current_user(request)
    return {"account": uid != "demo", "gurus": study.summary(uid)}


@router.get("/study/{persona_id}")
def study_course(persona_id: str, request: Request):
    return _run(study.public_view, current_user(request), persona_id)


@router.post("/study/{persona_id}/lessons/{lesson_id}/complete")
def study_lesson_complete(persona_id: str, lesson_id: str, request: Request):
    return _run(study.complete_lesson, current_user(request), persona_id, lesson_id)


@router.post("/study/{persona_id}/modules/{module_id}/quiz")
def study_quiz(persona_id: str, module_id: str, body: QuizBody, request: Request):
    return _run(study.submit_quiz, current_user(request), persona_id, module_id, body.answers)
