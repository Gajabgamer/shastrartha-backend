"""Loads personas, topics, regions and corpus chunks from data/."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import get_settings
from .schemas import Chunk, Persona, Topic

PLAYABLE = ("chanakya", "sun_tzu", "napoleon", "marcus_aurelius")


class Knowledge:
    def __init__(self, data_dir: Path) -> None:
        self.personas: dict[str, Persona] = {}
        for pid in PLAYABLE:
            path = data_dir / "personas" / f"{pid}.seed.json"
            self.personas[pid] = Persona.model_validate_json(path.read_text(encoding="utf-8"))

        topics = json.loads((data_dir / "topics.json").read_text(encoding="utf-8"))
        self.topics: dict[str, Topic] = {t["id"]: Topic.model_validate(t) for t in topics}

        self.regions: list[dict] = json.loads((data_dir / "regions.json").read_text(encoding="utf-8"))

        chunks_path = data_dir / "corpus" / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"{chunks_path} missing. Run: python -m scripts.build_corpus")
        self.chunks: dict[str, Chunk] = {}
        self.library: set[str] = set()  # ids that came from the dense library, not the curated corpus
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                c = Chunk.model_validate_json(line)
                self.chunks[c.chunk_id] = c

        # Present-day background (data/today, Wikipedia leads). Resolvable as sources, never retrieved as a guru's text.
        from .today import load_passages

        self.today: list[Chunk] = load_passages()
        for c in self.today:
            self.chunks.setdefault(c.chunk_id, c)

        # Topics generated from news headlines (data/today/generated.json), with the passages they cite.
        gen_path = data_dir / "today" / "generated.json"
        if gen_path.exists():
            for tid, e in json.loads(gen_path.read_text(encoding="utf-8")).items():
                for raw in [e["stance_chunk"], *e.get("support_chunks", [])]:
                    c = Chunk.model_validate(raw)
                    if c.chunk_id not in self.chunks:
                        self.chunks[c.chunk_id] = c
                        self.library.add(c.chunk_id)
                self.topics[tid] = Topic.model_validate(e["topic"])

        cited = [t.stance.source_id for t in self.topics.values()]
        cited += [s.chunk_id for t in self.topics.values() for s in t.support]
        cited += [c for p in self.personas.values() for n in p.knowledge_nodes for c in n.get("chunk_ids", [])]
        missing = [c for c in cited if c not in self.chunks]
        if missing:
            raise ValueError(f"Topic stances cite unknown chunks: {missing}")

    def topics_for(self, persona_id: str) -> list[Topic]:
        return [t for t in self.topics.values() if t.persona_id == persona_id]

    def chunks_for(self, persona_id: str) -> list[Chunk]:
        return [c for c in self.chunks.values() if c.persona_id == persona_id and c.chunk_id not in self.library]

    def register(self, chunk: Chunk) -> None:
        """Remember a passage retrieved from the dense library so citations to it resolve later."""
        if chunk.chunk_id not in self.chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.library.add(chunk.chunk_id)

    def chunk(self, chunk_id: str) -> Chunk | None:
        """Curated corpus first, then the ChromaDB library (data/knowledge)."""
        c = self.chunks.get(chunk_id)
        if c is None:
            from .rag import dense

            c = dense.get(chunk_id)
            if c is not None:
                self.register(c)
        return c


@lru_cache
def get_knowledge() -> Knowledge:
    return Knowledge(get_settings().data_dir)
