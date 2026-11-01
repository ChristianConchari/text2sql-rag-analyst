"""Retriever interface and the passage type shared by every retriever implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Passage:
    """Evidence from one article: all of its retrieved chunks joined, scored by the best chunk."""

    art_id: str          # stable id, "<norma>|<label>", e.g. "Ley 19886|Artículo 8 bis"
    norma: str           # e.g. "Ley 19886"
    label: str           # e.g. "Artículo 8 bis"
    article_number: str  # e.g. "8 bis"
    text: str
    score: float         # cosine similarity of the best chunk
    version: str | None = None


class Retriever(Protocol):
    """`search` returns at most `k` passages, one per article, best first, all with score >= the retriever's
    minimum. An EMPTY list means "no evidence": callers must not answer from outside knowledge."""

    def search(self, query: str, k: int | None = None) -> list[Passage]: ...
