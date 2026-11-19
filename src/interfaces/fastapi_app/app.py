"""HTTP surface under /api: POST /api/ask runs the pipeline (the verifier and the output guardrail are inside it);
GET /api/health. Response shapes are in `schemas.py`. If a built frontend directory is given and exists, it is
served at "/" (same origin as the API, so no CORS).

No authentication or rate limiting yet: do not expose it beyond localhost as is.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.interfaces.fastapi_app.schemas import AskRequest, AskResponse, Citation, HealthResponse
from src.runtime.pipelines.text2sql import Text2SQLPipeline
from src.runtime.providers.llm_client import ChatModel

__all__ = ["AskRequest", "AskResponse", "create_app"]


def model_name(llm: ChatModel) -> str:
    """The configured model id, or the client's class name (e.g. FakeLLM in tests)."""
    inner = getattr(llm, "inner", llm)
    return getattr(getattr(inner, "config", None), "model", None) or type(inner).__name__


def create_app(pipeline: Text2SQLPipeline, static_dir: str | Path | None = None) -> FastAPI:
    """Build the app around a pipeline (injected so tests can use a FakeLLM).

    `static_dir`: the frontend build (`frontend/dist/frontend/browser`). Mounted at "/" after the /api routes,
    only if it exists, so the API works the same without a frontend build.
    """
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

    if static_dir is not None and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app
