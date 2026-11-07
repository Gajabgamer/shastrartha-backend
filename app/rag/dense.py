"""Dense retrieval over the full persona library in ChromaDB (data/chroma, built by scripts.knowledge.index).

Optional by design: if chromadb is not installed or the index is missing, every call returns [] and the
BM25 path carries on alone (CLAUDE.md rule 4: every feature has a fallback).
"""

from __future__ import annotations

import logging
import threading

from ..config import get_settings
from ..schemas import Chunk

log = logging.getLogger(__name__)

# The persona's own words and eyewitness/ancient accounts. "scholarship" (modern secondary works)
# and "background" (Wikipedia, CC BY-SA) are never used as the persona's evidence.
EVIDENCE_TYPES = ("primary", "account")


_UNSET = object()
_col = _UNSET
# Chroma's client and the ONNX embedder are not safe to initialise or query from several threads at
# once (startup warm-up, verification thread, request handlers). One lock serialises all access.
_lock = threading.RLock()


def _collection():
    global _col
    with _lock:
        if _col is not _UNSET:
            return _col
        s = get_settings()
        _col = None
        if s.dense_enabled and s.chroma_dir.exists():
            try:
                import chromadb
                from chromadb.utils import embedding_functions

                client = chromadb.PersistentClient(path=str(s.chroma_dir))
                _col = client.get_collection(s.chroma_collection,
                                             embedding_function=embedding_functions.DefaultEmbeddingFunction())
            except Exception as exc:  # noqa: BLE001 - any failure means "no dense index", not a crash
                log.warning("Dense retrieval disabled: %s", exc)
        return _col


def available() -> bool:
    return _collection() is not None


def _to_chunk(cid: str, doc: str, meta: dict) -> Chunk:
    return Chunk(
        chunk_id=cid, source_id=meta.get("source_id", ""), persona_id=meta.get("persona_id", ""),
        work=meta.get("title", ""), edition=meta.get("edition", ""), section=meta.get("section", ""),
        language="en", license=meta.get("license", ""), url=meta.get("url", ""), text=doc,
        source_type=meta.get("source_type", "primary"),
    )


def search(persona_id: str, query: str, k: int = 20, types: tuple[str, ...] = EVIDENCE_TYPES) -> list[tuple[Chunk, float]]:
    """(chunk, cosine similarity) best first. Empty list when the index is unavailable."""
    col = _collection()
    if col is None or not query.strip():
        return []
    where = {"$and": [{"persona_id": persona_id}, {"source_type": {"$in": list(types)}}]}
    try:
        with _lock:
            res = col.query(query_texts=[query], n_results=k, where=where)
    except Exception as exc:  # noqa: BLE001
        log.warning("Dense query failed: %s", exc)
        return []
    return [(_to_chunk(cid, doc, meta), 1.0 - dist)
            for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def get(chunk_id: str) -> Chunk | None:
    col = _collection()
    if col is None:
        return None
    with _lock:
        res = col.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not res["ids"]:
        return None
    return _to_chunk(res["ids"][0], res["documents"][0], res["metadatas"][0])
