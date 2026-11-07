"""Shared text utilities: tokenising, sentence splitting, quote extraction."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """a an the and or but if then of to in on at by for with from as is are was were be been being it its this that these
    those he she they them his her their we our you your i me my not no nor so such than too very can could should would
    shall will may might must do does did done have has had having into out over under up down about against between
    which who whom whose what when where why how all any both each few more most other some own same only also just
    there here one ones thus hence therefore because since while whilst upon unto thou thee thy""".split()
)

_WORD = re.compile(r"[a-z]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def stem(word: str) -> str:
    for suffix in ("ings", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(normalize(text)) if w not in STOPWORDS and len(w) > 2]


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?;])\s+(?=[\"“'(A-Z0-9])", flat)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def overlap(a: str, b: str) -> float:
    """Share of a's content tokens that also occur in b (0..1)."""
    ta = set(tokens(a))
    if not ta:
        return 0.0
    return len(ta & set(tokens(b))) / len(ta)


def best_sentence(passage: str, query: str, min_words: int = 0) -> str:
    sents = sentences(passage) or [passage]
    if min_words:
        sents = [s for s in sents if len(s.split()) >= min_words] or sents
    return max(sents, key=lambda s: (overlap(query, s) + 0.5 * overlap(s, query), -abs(len(s.split()) - 20)))


def short_quote(passage: str, query: str, max_words: int = 15) -> str:
    """Verbatim quote of at most max_words words from the most relevant sentence (rule: quote <= 15 words)."""
    sent = re.sub(r"^\d+(?:, ?\d+)*\.\s*", "", best_sentence(passage, query, min_words=9))
    sent = re.sub(r"^Sun Tz\S* said:\s*", "", sent)
    sent = sent.replace("_", "")
    ws = sent.split()
    if len(ws) <= max_words:
        return sent.rstrip(" ;:,")
    # Choose the window of max_words with the most query overlap.
    q = set(tokens(query))
    best_i, best_hits = 0, -1
    for i in range(0, len(ws) - max_words + 1):
        hits = len(set(tokens(" ".join(ws[i : i + max_words]))) & q)
        if hits > best_hits:
            best_i, best_hits = i, hits
    end = best_i + max_words
    window = " ".join(ws[best_i:end]).rstrip(" ;:,")
    tail = "" if end >= len(ws) else "…"
    if tail:
        window = window.rstrip(".")
    return ("…" if best_i > 0 else "") + window + tail
