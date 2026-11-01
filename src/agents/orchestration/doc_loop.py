"""Answer loop for the document route: retrieve-then-generate with a deterministic citation check.

Mirrors the SQL loop: ask the model, verify, feed the critique back, up to `1 + max_retries` calls, then decline.
Guarantees: no passages -> no model call and status "no_evidence"; an "ok" answer has at least one citation and
every citation is a retrieved passage. Not covered: whether the cited text supports the claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.policies.citation_verifier import verify_citations
from src.rag.citations import citation_label
from src.rag.prompts.doc_prompt import NO_EVIDENCE_TOKEN, build_doc_messages
from src.rag.retrievers.base import Passage, Retriever
from src.runtime.providers.llm_client import ChatModel, LLMError

EXCERPT_CHARS = 300


@dataclass(frozen=True)
class Citation:
    """A source shown to the user. Built from a retrieved passage, never from model output."""

    label: str      # "Ley 19.886 art. 7"
    norma: str
    article: str    # article number, e.g. "8 bis"
    excerpt: str


@dataclass
class DocResult:
    status: str                     # ok | no_evidence | declined | failed
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    violations: list[list[str]] = field(default_factory=list)   # per failed attempt
    attempts: int = 0
    reason: str | None = None


def make_citation(passage: Passage) -> Citation:
    body = re.sub(rf"^{re.escape(passage.norma)}, {re.escape(passage.label)}:\s*", "", passage.text)
    body = " ".join(body.split())
    excerpt = body if len(body) <= EXCERPT_CHARS else body[: EXCERPT_CHARS - 1] + "…"
    return Citation(citation_label(passage), passage.norma, passage.article_number, excerpt)


def run_doc_loop(question: str, llm: ChatModel, retriever: Retriever, max_retries: int = 3) -> DocResult:
    """Retrieve, generate and verify. See the module docstring for the status values."""
    passages = retriever.search(question)
    if not passages:
        return DocResult("no_evidence", reason="no passage above the similarity threshold")
    messages = build_doc_messages(question, passages)
    result = DocResult("failed")
    for _ in range(1 + max_retries):
        try:
            raw = llm.chat(messages, max_tokens=500, temperature=0).strip()
        except LLMError as e:
            result.reason = f"llm_error: {e}"
            return result
        result.attempts += 1
        if raw.upper().startswith(NO_EVIDENCE_TOKEN):
            result.status, result.reason = "declined", raw[len(NO_EVIDENCE_TOKEN):].lstrip(": ").strip() or None
            return result
        verdict = verify_citations(raw, passages)
        if verdict.ok:
            result.status, result.answer = "ok", raw
            result.citations = [make_citation(p) for p in verdict.citations]
            return result
        result.violations.append(verdict.violations)
        allowed = ", ".join(f"[{citation_label(p)}]" for p in passages)
        feedback = ("El verificador de citas rechazó la respuesta: " + "; ".join(verdict.violations)
                    + f". Cita solo con estas referencias: {allowed}. Corrige la respuesta.")
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}]
    result.reason = "max_retries_exceeded"
    return result
