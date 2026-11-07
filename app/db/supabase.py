"""Supabase over its HTTP APIs (PostgREST + GoTrue). No SDK needed.

- The backend uses the service_role key, so it bypasses RLS; it is the only writer of scores, ratings,
  debates and regions (supabase/schema.sql). The key lives in .env only and is never sent to browsers.
- Writes go to one background thread (ordered, never block the debate loop). Reads are synchronous and
  are only made from request handlers or once per learner (cached by the store).
- If SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are unset, `enabled()` is False and the app stays in-memory.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from ..config import get_settings

log = logging.getLogger(__name__)

_session = requests.Session()
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="supabase-writer")
_user_cache: dict[str, tuple[str | None, float]] = {}
USER_CACHE_S = 300


def enabled() -> bool:
    s = get_settings()
    return bool(s.supabase_url and s.supabase_service_key)


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    key = get_settings().supabase_service_key
    h = {"apikey": key, "Content-Type": "application/json"}
    if not key.startswith("sb_"):  # legacy JWT service_role key also goes in Authorization; new sb_secret_ keys must not
        h["Authorization"] = f"Bearer {key}"
    return {**h, **(extra or {})}


def _rest(path: str) -> str:
    return f"{get_settings().supabase_url.rstrip('/')}/rest/v1/{path}"


# ---------------------------------------------------------------- auth

def user_id_from_token(token: str) -> str | None:
    """Validate a user's access token with Supabase Auth; returns the user id or None. Cached 5 min."""
    if not (enabled() and token):
        return None
    hit = _user_cache.get(token)
    if hit and hit[1] > time.time():
        return hit[0]
    try:
        r = _session.get(
            f"{get_settings().supabase_url.rstrip('/')}/auth/v1/user",
            headers={"apikey": get_settings().supabase_service_key, "Authorization": f"Bearer {token}"},
            timeout=10,
        )
        uid = r.json().get("id") if r.ok else None
    except requests.RequestException as exc:
        log.warning("Supabase auth check failed: %s", exc)
        return None
    _user_cache[token] = (uid, time.time() + USER_CACHE_S)
    return uid


# ---------------------------------------------------------------- reads (sync)

def select(table: str, query: str = "") -> list[dict[str, Any]]:
    try:
        r = _session.get(_rest(f"{table}?{query}"), headers=_headers(), timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        log.warning("Supabase select %s failed: %s", table, exc)
        return []


# ---------------------------------------------------------------- writes (background, ordered)

def _do_write(method: str, table: str, rows: Any, on_conflict: str | None, query: str) -> None:
    url = _rest(table + (f"?on_conflict={on_conflict}" if on_conflict else f"?{query}" if query else ""))
    prefer = "resolution=merge-duplicates,return=minimal" if on_conflict else "return=minimal"
    try:
        r = _session.request(method, url, json=rows, headers=_headers({"Prefer": prefer}), timeout=20)
        if r.status_code >= 300:
            log.warning("Supabase %s %s -> %s %s", method, table, r.status_code, r.text[:200])
    except requests.RequestException as exc:
        log.warning("Supabase %s %s failed: %s", method, table, exc)


def upsert(table: str, rows: Any, on_conflict: str) -> None:
    if enabled():
        _writer.submit(_do_write, "POST", table, rows, on_conflict, "")


def insert(table: str, rows: Any) -> None:
    if enabled():
        _writer.submit(_do_write, "POST", table, rows, None, "")


def update(table: str, query: str, values: dict[str, Any]) -> None:
    if enabled():
        _writer.submit(_do_write, "PATCH", table, values, None, query)


def flush(timeout: float = 10) -> None:
    """Wait for queued writes (used by tests and shutdown)."""
    _writer.submit(lambda: None).result(timeout=timeout)
