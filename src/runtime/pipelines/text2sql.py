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
from src.agents.orchestration.sql_loop import LoopResult, run_sql_loop
from src.policies.sql_verifier import Policy, load_policy
from src.rag.prompts.sql_prompt import build_plain_schema_context, build_schema_context, build_sql_system
from src.rag.retrievers.base import Retriever
from src.rag.retrievers.value_index import ValueIndex, format_hits, load_value_lookup_config
from src.runtime.features import SQLFeatures, load_features
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
                 retriever: Retriever | None = None, value_index: ValueIndex | None = None,
                 features: SQLFeatures | None = None):
        self.llm, self.policy, self.db_path = llm, policy, Path(db_path)
        self.system_prompt, self.max_retries, self.retriever = system_prompt, max_retries, retriever
        self.value_index = value_index
        # Flags only switch helpers off. `glossary` is applied by whoever builds `system_prompt`; the rest here.
        self.features = features or SQLFeatures()

    def _shared_prefix(self, agent: str) -> str | None:
        """Static SQL block to put before an agent's unchanged instruction (see features.PROMPT_LAYOUTS), else None.

        The goal is that a single-slot model server keeps the SQL prefix cached while agents alternate. The router
        only gets it in the experimental `cache_friendly_router` layout: it weakens refusals, so it is not safe.
        """
        layout = self.features.prompt_layout
        if layout == "legacy" or (agent == "router" and layout != "cache_friendly_router"):
            return None
        return self.system_prompt

    def generate_sql(self, question: str) -> LoopResult:
        """The SQL route without router or composer: value lookup (if on) + generator/verifier loop.

        Safety rules and masking always apply; `features.critic` / `features.retries` only control the critic and
        the retries (see configs/pipeline/sql_features.yaml).
        """
        f = self.features
        hits = self.value_index.lookup(question) if (self.value_index and f.value_lookup) else []
        context = format_hits(hits) if hits else None
        retries = min(self.max_retries, f.effective_retries)
        return run_sql_loop(question, self.llm, self.policy, lambda s: run_query(self.db_path, s),
                            self.system_prompt, retries, context=context, critic=f.critic,
                            max_completion_tokens=f.max_completion_tokens)

    def answer(self, question: str) -> Answer:
        """Answer one question end to end."""
        route = route_question(question, self.llm, self._shared_prefix("router"))
        if route == "refuse":
            return Answer(route, "refused", "No puedo ayudar con esa solicitud.")
        if route == "docs":
            return self._answer_docs(question)
        if route == "unknown":
            return Answer(route, "needs_clarification", "No entendí la pregunta; ¿puedes reformularla?")
        res = self.generate_sql(question)
        if res.status == "declined":
            return Answer(route, "declined", f"Con estos datos no puedo responder: {res.reason or 'sin detalle'}",
                          attempts=len(res.attempts))
        if res.status == "failed":
            return Answer(route, "failed", "No logré generar una consulta válida para tu pregunta.",
                          attempts=len(res.attempts), warnings=[res.reason] if res.reason else [])
        text = compose(question, res.columns, res.rows, self.llm, self._shared_prefix("composer"))
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


def build_default_value_index(db_path: str | Path) -> ValueIndex | None:
    """Value lookup over the configured columns, or None (with a warning) if not configured or the DB is missing."""
    config = load_value_lookup_config(ROOT / "configs/domain/chilecompra.yaml")
    if config is None:
        return None
    try:
        return ValueIndex.from_duckdb(db_path, config)
    except Exception as e:   # noqa: BLE001 - a missing/old DB must not stop the API; the loop works without it
        log.warning("value lookup disabled: %s", e)
        return None


def build_default_pipeline(features: SQLFeatures | None = None, llm: ChatModel | None = None,
                           with_retriever: bool = True) -> Text2SQLPipeline:
    """Production wiring: configs/model/llm.yaml, configs/domain/chilecompra.yaml, the glossary; DB_PATH overrides the DB.

    `features` defaults to configs/pipeline/sql_features.yaml (all on). `llm` and `with_retriever=False` exist for
    evaluation scripts (a metered model; no need to load the document retriever).
    """
    features = features or load_features()
    policy = load_policy(ROOT / "configs/domain/chilecompra.yaml")
    db_path = os.environ.get("DB_PATH", ROOT / "data/chilecompra.duckdb")
    if features.glossary:
        ctx = build_schema_context(ROOT / "domains/chilecompra/glossary.yaml", policy.blocked_columns)
    else:
        ctx = build_plain_schema_context(db_path, next(iter(sorted(policy.allowed_tables))), policy.blocked_columns)
    value_index = build_default_value_index(db_path) if features.value_lookup else None
    return Text2SQLPipeline(llm or LLMClient.from_config(), policy, db_path, build_sql_system(ctx),
                            features.max_retries,
                            retriever=build_default_retriever() if with_retriever else None,
                            value_index=value_index, features=features)
