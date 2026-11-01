"""End-to-end pipeline: router -> SQL loop (generator, verifier, executor, output guardrail) -> composer.

`Text2SQLPipeline.answer` never raises for model, SQL or retrieval problems; it returns an `Answer` whose `status`
says what happened. The "docs" route retrieves law passages and answers with verified article citations; without
a retriever it answers "not_implemented". There is NO mixed route yet: a question that needs both figures and
legal text is sent to one tool only (whichever the router picks).
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agents.orchestration.composer import compose
from src.agents.orchestration.doc_loop import run_doc_loop
from src.agents.orchestration.router import route_question
from src.agents.orchestration.sql_loop import run_sql_loop
from src.policies.sql_verifier import Policy, load_policy
from src.rag.prompts.sql_prompt import build_schema_context, build_sql_system
from src.rag.retrievers.base import Retriever
from src.runtime.providers.llm_client import ChatModel, LLMClient
from src.toolkits.db.executor import run_query

ROOT = Path(__file__).resolve().parents[3]
log = logging.getLogger(__name__)

NO_EVIDENCE_TEXT = "No encontré evidencia en la Ley 19.886 ni en su reglamento para responder esa pregunta."


@dataclass
class Answer:
    """What the API returns. `rows` are already masked."""

    route: str
    status: str            # ok | declined | failed | refused | no_evidence | not_implemented | needs_clarification
    answer: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)   # masked
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0
    citations: list[dict] = field(default_factory=list)   # docs route: label, norma, article, excerpt (all retrieved)

    def to_dict(self) -> dict:
        """Plain dict for JSON serialization."""
        return asdict(self)


class Text2SQLPipeline:
    """Wires the model, policy, database and prompt together."""

    def __init__(self, llm: ChatModel, policy: Policy, db_path: str | Path, system_prompt: str, max_retries: int = 3,
                 retriever: Retriever | None = None):
        self.llm, self.policy, self.db_path = llm, policy, Path(db_path)
        self.system_prompt, self.max_retries, self.retriever = system_prompt, max_retries, retriever

    def answer(self, question: str) -> Answer:
        """Answer one question end to end."""
        route = route_question(question, self.llm)
        if route == "refuse":
            return Answer(route, "refused", "No puedo ayudar con esa solicitud.")
        if route == "docs":
            return self._answer_docs(question)
        if route == "unknown":
            return Answer(route, "needs_clarification", "No entendí la pregunta; ¿puedes reformularla?")
        res = run_sql_loop(question, self.llm, self.policy, lambda s: run_query(self.db_path, s),
                           self.system_prompt, self.max_retries)
        if res.status == "declined":
            return Answer(route, "declined", f"Con estos datos no puedo responder: {res.reason or 'sin detalle'}",
                          attempts=len(res.attempts))
        if res.status == "failed":
            return Answer(route, "failed", "No logré generar una consulta válida para tu pregunta.",
                          attempts=len(res.attempts), warnings=[res.reason] if res.reason else [])
        text = compose(question, res.columns, res.rows, self.llm)
        return Answer(route, "ok", text, res.sql, res.columns, [list(r) for r in res.rows],
                      res.warnings, len(res.attempts))


    def _answer_docs(self, question: str) -> Answer:
        if self.retriever is None:
            return Answer("docs", "not_implemented", "La consulta de leyes y reglamento no está configurada.")
        res = run_doc_loop(question, self.llm, self.retriever, self.max_retries)
        if res.status == "no_evidence":
            return Answer("docs", "no_evidence", NO_EVIDENCE_TEXT)
        if res.status == "declined":
            return Answer("docs", "declined", f"Con los textos recuperados no puedo responder: {res.reason or 'sin detalle'}",
                          attempts=res.attempts)
        if res.status == "failed":
            return Answer("docs", "failed", "No logré producir una respuesta con citas verificables.",
                          attempts=res.attempts, warnings=[res.reason] if res.reason else [])
        return Answer("docs", "ok", res.answer, attempts=res.attempts, citations=[asdict(c) for c in res.citations])


def build_default_retriever() -> Retriever | None:
    """The bge-m3 retriever over data/docs/chunks.jsonl, or None (with a warning) if torch or the corpus is missing."""
    from src.rag.retrievers.dense import build_dense_retriever, load_retrieval_config
    from src.rag.retrievers.embedders import SentenceTransformerEmbedder
    config = load_retrieval_config()
    try:
        return build_dense_retriever(config, SentenceTransformerEmbedder(config.model))
    except (ImportError, FileNotFoundError) as e:
        log.warning("document route disabled: %s", e)
        return None


def build_default_pipeline() -> Text2SQLPipeline:
    """Production wiring: configs/model/llm.yaml, configs/domain/chilecompra.yaml, the glossary; DB_PATH overrides the DB."""
    policy = load_policy(ROOT / "configs/domain/chilecompra.yaml")
    ctx = build_schema_context(ROOT / "domains/chilecompra/glossary.yaml", policy.blocked_columns)
    return Text2SQLPipeline(LLMClient.from_config(), policy,
                            os.environ.get("DB_PATH", ROOT / "data/chilecompra.duckdb"), build_sql_system(ctx),
                            retriever=build_default_retriever())
