"""Live Sabha: the audience votes from their phones (scan a QR, no login).

A Sabha is attached to one target:
- kind "debate": a shishya vs guru debate. Each round the audience answers "who argued better this round?"
  (guru | shishya). The share of shishya votes over all rounds is the crowd score, and the final verdict
  becomes 0.7 * AI judge + 0.3 * crowd (docs/DEBATE_ENGINE_SPEC.md). Persisted to sabha_sessions/sabha_votes.
- kind "rivalry": two gurus argue one question; the audience picks which one argued better (in memory).

One vote per phone per round (a random token kept in the phone's localStorage); a phone may change its vote
while the round is open. Votes close when the debate reaches its verdict or the rivalry is closed.
"""

from __future__ import annotations

import random
import socket
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import get_settings
from ..db import supabase as db

CODE_CHARS = "".join(c for c in string.ascii_uppercase + string.digits if c not in "0O1IL")


class SabhaError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass
class Sabha:
    code: str
    kind: str  # debate | rivalry
    target_id: str
    choices: list[tuple[str, str]]  # (key, label)
    created_at: float = field(default_factory=time.time)
    votes: dict[tuple[int, str], str] = field(default_factory=dict)  # (round_no, voter token) -> choice
    closed: bool = False

    # ------------------------------------------------------------ target state

    def _target(self):
        if self.kind == "debate":
            from .debate import registry

            return registry.get(self.target_id)
        from .rivalry import bouts

        return bouts.get(self.target_id)

    def open_round(self) -> int | None:
        """The round the audience can vote on now, or None (not started yet / closed)."""
        if self.closed:
            return None
        t = self._target()
        if self.kind == "debate":
            if t.status != "active":
                return None
            argued = [x.round_no for x in t.turns if x.speaker == "user"]
            return max(argued) if argued else None
        return 1 if t.turns and t.status != "closed" else None

    def tally(self) -> dict[str, Any]:
        rounds: dict[int, dict[str, int]] = {}
        for (r, _), c in self.votes.items():
            rounds.setdefault(r, {k: 0 for k, _ in self.choices})[c] += 1
        total = {k: sum(x[k] for x in rounds.values()) for k, _ in self.choices}
        return {"rounds": {str(r): v for r, v in sorted(rounds.items())}, "total": total,
                "voters": len({tok for (_, tok) in self.votes})}

    def crowd_score(self) -> float | None:
        """Debates only: % of all votes that went to the shishya (None when nobody voted)."""
        t = self.tally()["total"]
        n = sum(t.values())
        return round(100 * t.get("shishya", 0) / n, 1) if n else None

    def vote(self, round_no: int, token: str, choice: str) -> None:
        if choice not in {k for k, _ in self.choices}:
            raise SabhaError("VALIDATION_ERROR", "Unknown choice", 422)
        open_r = self.open_round()
        if open_r is None:
            raise SabhaError("VALIDATION_ERROR", "Voting is not open right now.", 409)
        if round_no != open_r:
            raise SabhaError("VALIDATION_ERROR", f"Round {round_no} is closed; vote on round {open_r}.", 409)
        self.votes[(round_no, token)] = choice
        if self.kind == "debate":
            db.upsert("sabha_votes", {"code": self.code, "round_no": round_no, "voter_token": token, "choice": choice},
                      on_conflict="code,round_no,voter_token")

    def public(self) -> dict[str, Any]:
        t = self._target()
        ctx: dict[str, Any] = {}
        if self.kind == "debate":
            r = self.open_round()
            user = next((x.text for x in reversed(t.turns) if x.speaker == "user" and x.round_no == r), "")
            guru = next((x.text for x in reversed(t.turns) if x.speaker == "persona" and x.round_no == r), "")
            ctx = {"question": t.topic.question, "persona_id": t.persona.id, "shishya": user, "guru": guru,
                   "status": t.status, "stage": t.stage, "verdict": t.verdict, "study": t.study()}
        else:
            ctx = {"question": t.topic.question, "a": t.a.id, "b": t.b.id,
                   "turns": [{"speaker": x["persona_id"], "text": x["text"]} for x in t.turns[-2:]], "status": t.status}
        return {"code": self.code, "kind": self.kind, "target_id": self.target_id, "round_no": self.open_round(),
                "choices": [{"key": k, "label": lbl} for k, lbl in self.choices], "tally": self.tally(),
                "crowd_score": self.crowd_score() if self.kind == "debate" else None, "closed": self.closed,
                "join_url": join_url(self.code), "context": ctx}


# ---------------------------------------------------------------- join url (phones on the same Wi-Fi)

def _lan_ip() -> str:
    """This laptop's address on the local network (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def join_url(code: str) -> str:
    s = get_settings()
    base = s.sabha_public_url.rstrip("/") if s.sabha_public_url else ""
    if not base:
        port = s.frontend_url.rsplit(":", 1)[-1].strip("/") if s.frontend_url.count(":") == 2 else "3000"
        base = f"http://{_lan_ip()}:{port}"
    return f"{base}/vote/{code}"


def qr_svg(text: str) -> str:
    import qrcode
    import qrcode.image.svg

    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
    return img.to_string(encoding="unicode")


# ---------------------------------------------------------------- registry

class SabhaRegistry:
    def __init__(self) -> None:
        self.by_code: dict[str, Sabha] = {}
        self.by_target: dict[str, str] = {}
        self._lock = threading.Lock()

    def open(self, kind: str, target_id: str, choices: list[tuple[str, str]], created_by: str | None = None) -> Sabha:
        with self._lock:
            if target_id in self.by_target:
                return self.by_code[self.by_target[target_id]]
            code = "".join(random.choices(CODE_CHARS, k=5))
            while code in self.by_code:
                code = "".join(random.choices(CODE_CHARS, k=5))
            s = Sabha(code, kind, target_id, choices)
            self.by_code[code] = s
            self.by_target[target_id] = code
        if kind == "debate":
            db.insert("sabha_sessions", {"code": code, "debate_id": target_id, "created_by": created_by})
        return s

    def get(self, code: str) -> Sabha:
        s = self.by_code.get(code.upper())
        if s is None:
            raise SabhaError("NOT_FOUND", "No Sabha with that code", 404)
        return s

    def for_target(self, target_id: str) -> Sabha | None:
        code = self.by_target.get(target_id)
        return self.by_code.get(code) if code else None

    def close(self, target_id: str) -> None:
        s = self.for_target(target_id)
        if s and not s.closed:
            s.closed = True
            if s.kind == "debate":
                db.update("sabha_sessions", f"code=eq.{s.code}", {"status": "closed"})


sabhas = SabhaRegistry()
