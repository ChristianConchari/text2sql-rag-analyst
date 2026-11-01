"""Deterministic citation verifier for the document route (the counterpart of the SQL verifier).

`verify_citations(answer, passages)` checks an LLM answer against what was actually retrieved. Rules:
  no_citation          the answer has no citation in the form `[<norma> art. <n>]`: an uncited legal claim is
                       unverifiable, so it is rejected
  invented_citation    a citation that matches none of the retrieved passages (norma + article number, ignoring
                       case, accents and thousands dots): the model made up a source
Guarantees: `Verdict.citations` only ever contains retrieved `Passage` objects, never text taken from the answer,
so a fabricated source cannot reach the user. Fail-closed: any violation rejects the whole answer.
Does NOT check that the cited article really supports the sentence (that needs a human or a judge model) nor that
the answer is complete.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.citations import citation_key, find_citations, passage_key
from src.rag.retrievers.base import Passage


@dataclass
class CitationVerdict:
    """`ok` is True only with at least one citation and no invented one."""

    ok: bool
    citations: list[Passage] = field(default_factory=list)   # distinct retrieved passages cited, in answer order
    violations: list[str] = field(default_factory=list)


def verify_citations(answer: str, passages: list[Passage]) -> CitationVerdict:
    """Check every citation in `answer` against `passages`."""
    by_key = {passage_key(p): p for p in passages}
    found = find_citations(answer)
    if not found:
        return CitationVerdict(False, violations=["no_citation"])
    cited: dict[str, Passage] = {}
    violations: list[str] = []
    for raw, norma, number in found:
        key = citation_key(norma, number)
        if key in by_key:
            cited.setdefault(key, by_key[key])
        elif f"invented_citation: {raw}" not in violations:
            violations.append(f"invented_citation: {raw}")
    return CitationVerdict(not violations, list(cited.values()) if not violations else [], violations)
