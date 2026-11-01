"""Load the chunked legal corpus produced by scripts/chunk_docs.py."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """One retrievable fragment; several chunks can belong to the same article (`art_id`)."""

    id: str
    art_id: str
    norma: str
    label: str
    article_number: str
    text: str
    version: str | None = None


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read a chunks.jsonl file. Raises FileNotFoundError if the corpus has not been built."""
    chunks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        chunks.append(Chunk(id=r["id"], art_id=r["art_id"], norma=r["norma"], label=r["label"],
                            article_number=r.get("article_number") or r["label"], text=r["text"],
                            version=r.get("version")))
    return chunks
