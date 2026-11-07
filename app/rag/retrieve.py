"""Hybrid retrieval: BM25 over the curated corpus + dense search over the full ChromaDB library.

- Curated corpus (data/corpus/chunks.jsonl): hand-sectioned passages that topics and stances cite.
- Library (data/chroma, built by scripts.knowledge.*): every downloaded primary text and ancient account.

The two ranked lists are merged with reciprocal-rank fusion (RRF, k=60). Library passages that repeat a
curated passage are dropped, so the cleaner curated section wins. With no Chroma index this is plain BM25.
A cross-encoder rerank stage can sit behind `retrieve()` without changing callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from ..knowledge import get_knowledge
from ..schemas import Chunk
from . import dense
from .text import tokens

RRF_K = 60
CANDIDATES = 30
ACCOUNT_CANDIDATES = 10
ACCOUNT_WEIGHT = 0.5


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower()))


class PersonaIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        docs = [tokens(f"{c.section} {c.text}") for c in chunks]
        self.bm25 = BM25Okapi(docs) if docs else None
        self.normed = [(_norm(c.text), c) for c in chunks]

    def search(self, query: str, k: int = 5, exclude: set[str] | None = None) -> list[Hit]:
        q = tokens(query)
        if not q or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        hits: list[Hit] = []
        for i in order:
            c = self.chunks[i]
            if exclude and c.chunk_id in exclude:
                continue
            if scores[i] <= 0:
                break
            hits.append(Hit(c, float(scores[i]), len(hits) + 1))
            if len(hits) == k:
                break
        return hits

    def curated_twin(self, text: str) -> Chunk | None:
        """The curated passage that a library passage overlaps (same translation, different chunking)."""
        words = _norm(text).split()
        if len(words) < 16:
            return None
        probes = [" ".join(words[i : i + 10]) for i in (2, len(words) // 2 - 5, len(words) - 12)]
        for normed, c in self.normed:
            if any(p in normed for p in probes):
                return c
        return None


@lru_cache
def index_for(persona_id: str) -> PersonaIndex:
    return PersonaIndex(get_knowledge().chunks_for(persona_id))


def retrieve(persona_id: str, query: str, k: int = 5, exclude: set[str] | None = None) -> list[Hit]:
    idx = index_for(persona_id)
    sparse = idx.search(query, k=CANDIDATES, exclude=exclude)
    def library_list(types: tuple[str, ...], n: int) -> list[Chunk]:
        out: list[Chunk] = []
        for c, _ in dense.search(persona_id, query, k=n, types=types):
            c = idx.curated_twin(c.text) or c  # prefer the curated section; agreement then boosts it in RRF
            if (exclude and c.chunk_id in exclude) or any(x.chunk_id == c.chunk_id for x in out):
                continue
            out.append(c)
        return out

    # The persona's own words outrank accounts about them (memoirs can outnumber primary text 6:1).
    primary = library_list(("primary",), CANDIDATES)
    accounts = library_list(("account",), ACCOUNT_CANDIDATES)
    if not (primary or accounts):
        return [Hit(h.chunk, h.score, i + 1) for i, h in enumerate(sparse[:k])]

    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranked, weight in (([h.chunk for h in sparse], 1.0), (primary, 1.0), (accounts, ACCOUNT_WEIGHT)):
        for r, c in enumerate(ranked, start=1):
            fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + weight / (RRF_K + r)
            by_id[c.chunk_id] = c
    order = sorted(fused, key=fused.get, reverse=True)[:k]
    knowledge = get_knowledge()
    hits = []
    for i, cid in enumerate(order, start=1):
        knowledge.register(by_id[cid])  # so /sources/{id} and verification resolve library citations
        hits.append(Hit(by_id[cid], fused[cid], i))
    return hits
