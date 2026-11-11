"""Citation format shared by the answer prompt and the citation verifier: `[Ley 19.886 art. 8 bis]`.

One place defines how a passage is labelled and how a label found in free text is normalized, so the prompt and
the verifier cannot drift apart.
"""
from __future__ import annotations

import re
import unicodedata

from src.rag.retrievers.base import Passage

# `[<norma> art. <number>]`; "art", "art." and "artículo" are all accepted when reading an answer. One bracket may
# hold several references separated by ";" (`[Decreto 661 art. 59; Ley 19.886 art. 9]`): each is read on its own.
BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
REFERENCE_RE = re.compile(r"\s*(.*?\S)\s+art(?:\.|[íi]culo)?\s+(.*?\S)\s*", re.IGNORECASE)


def citation_label(passage: Passage) -> str:
    """Human-facing label the model must copy, e.g. "Ley 19.886 art. 8 bis"."""
    m = re.fullmatch(r"(\D+?)\s*(\d+)", passage.norma)
    norma = f"{m.group(1).strip()} {int(m.group(2)):,}".replace(",", ".") if m else passage.norma
    return f"{norma} art. {passage.article_number}"


def citation_key(norma: str, article_number: str) -> str:
    """Normalized identity of a citation: case, accents, dots and spacing are ignored."""
    def fold(text: str) -> str:
        text = unicodedata.normalize("NFD", text.lower())
        return " ".join("".join(c for c in text if unicodedata.category(c) != "Mn").replace(".", "").split())
    return f"{fold(norma)}|{fold(article_number)}"


def passage_key(passage: Passage) -> str:
    return citation_key(passage.norma, passage.article_number)


def find_citations(text: str) -> list[tuple[str, str, str]]:
    """Every citation-shaped reference in `text` as (raw, norma, article_number), in order of appearance.

    `raw` is the single reference in brackets, e.g. "[Ley 19.886 art. 9]", even when it came from a bracket with
    several references. Bracket parts that are not citation-shaped are ignored.
    """
    found: list[tuple[str, str, str]] = []
    for bracket in BRACKET_RE.finditer(text):
        for part in bracket.group(1).split(";"):
            m = REFERENCE_RE.fullmatch(part)
            if m:
                found.append((f"[{part.strip()}]", m.group(1), m.group(2)))
    return found
