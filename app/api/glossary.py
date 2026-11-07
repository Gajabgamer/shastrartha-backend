"""Terminology dictionary (TASKS.md workstream C): GET /api/glossary and GET /api/glossary/{slug}.

Entries live in data/glossary.json (written by hand, not generated). Each entry's source_ids are curated corpus
chunks whose text contains the term or one of its English glosses (checked by tests/test_glossary.py).
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config import get_settings

router = APIRouter(prefix="/api")

SEARCH_FIELDS = ("term", "devanagari", "short", "long", "aliases", "glosses")


def fold(text: str) -> str:
    """Lower-case and strip diacritics so 'Várta' matches 'varta'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


@lru_cache
def load_glossary() -> dict[str, Any]:
    """The parsed glossary file, read once per process."""
    path = get_settings().data_dir / "glossary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for e in data["entries"]:
        e.setdefault("devanagari", None)
        e.setdefault("origin", None)
        e.setdefault("pronunciation", None)
        e.setdefault("aliases", [])
        e.setdefault("glosses", [])
        e.setdefault("related", [])
        e.setdefault("source_ids", [])
        e.setdefault("auto", True)
    return data


def entries() -> list[dict[str, Any]]:
    return load_glossary()["entries"]


def categories() -> list[str]:
    seen: list[str] = []
    for e in entries():
        if e["category"] not in seen:
            seen.append(e["category"])
    return seen


def _haystack(e: dict[str, Any]) -> str:
    parts: list[str] = []
    for f in SEARCH_FIELDS:
        v = e.get(f)
        if isinstance(v, list):
            parts.extend(v)
        elif v:
            parts.append(v)
    return fold("\n".join(parts))


def search(q: str | None = None, category: str | None = None) -> list[dict[str, Any]]:
    """Entries whose term, Devanagari, meanings, aliases or glosses contain q (case and accent-insensitive).

    Term and alias hits are ranked before hits that only occur in the meanings.
    """
    out = entries()
    if category:
        cat = fold(category)
        out = [e for e in out if fold(e["category"]) == cat]
    if q and q.strip():
        needle = fold(q.strip())
        hits = [e for e in out if needle in _haystack(e)]

        def rank(e: dict[str, Any]) -> int:
            names = [fold(e["term"]), *(fold(a) for a in e["aliases"]), e["slug"]]
            if any(n == needle for n in names):
                return 0
            if any(n.startswith(needle) for n in names):
                return 1
            return 2

        out = sorted(hits, key=rank)
    return out


def _error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@router.get("/glossary")
def list_glossary(q: str | None = None, category: str | None = None) -> Any:
    if q is not None and len(q) > 200:
        return _error("VALIDATION_ERROR", "Search text is limited to 200 characters.", 422)
    found = search(q, category)
    return {"count": len(found), "total": len(entries()), "categories": categories(), "entries": found}


@router.get("/glossary/{slug}")
def get_term(slug: str) -> Any:
    key = fold(slug.strip())
    for e in entries():
        if e["slug"] == key:
            return e
    # also accept the display term or an alias ("Matsya nyaya", "matsyanyaya")
    for e in entries():
        if key == fold(e["term"]) or key in (fold(a) for a in e["aliases"]):
            return e
    return _error("NOT_FOUND", f"No glossary entry '{slug}'.", 404)
