"""HTTP surface under /api: POST /api/ask runs the pipeline (the verifier and the output guardrail are inside it);
GET /api/health. Response shapes are in `schemas.py`.

No authentication or rate limiting yet: do not expose it beyond localhost as is.
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI

from src.interfaces.fastapi_app.schemas import AskRequest, AskResponse, Citation, HealthResponse
from src.runtime.pipelines.text2sql import Text2SQLPipeline
from src.runtime.providers.llm_client import ChatModel

__all__ = ["AskRequest", "AskResponse", "create_app"]


def model_name(llm: ChatModel) -> str:
    """The configured model id, or the client's class name (e.g. FakeLLM in tests)."""
    inner = getattr(llm, "inner", llm)
    return getattr(getattr(inner, "config", None), "model", None) or type(inner).__name__


def create_app(pipeline: Text2SQLPipeline) -> FastAPI:
    """Build the app around a pipeline (injected so tests can use a FakeLLM)."""
    app = FastAPI(title="text2sql-rag-analyst")

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/api/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        t0 = time.perf_counter()
        a = pipeline.answer(req.question)
        return AskResponse(
            request_id=uuid.uuid4(), route=a.route, status=a.status, answer=a.answer, sql=a.sql,
            columns=a.columns, rows=a.rows, warnings=a.warnings, attempts=a.attempts,
            citations=[Citation(label=c["label"], norma=c["norma"], articulo=c["article"], excerpt=c["excerpt"])
                       for c in a.citations],
            trace=a.trace, latency_ms=round((time.perf_counter() - t0) * 1000, 1), model=model_name(pipeline.llm))

    return app
