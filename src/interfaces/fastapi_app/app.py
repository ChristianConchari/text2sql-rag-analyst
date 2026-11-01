"""HTTP surface. POST /ask runs the pipeline (the verifier and the output guardrail are inside it); GET /health.

No authentication or rate limiting yet: do not expose it beyond localhost as is.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.runtime.pipelines.text2sql import Text2SQLPipeline


class AskRequest(BaseModel):
    """Request body of POST /ask."""

    question: str = Field(min_length=1, max_length=1000)


def create_app(pipeline: Text2SQLPipeline) -> FastAPI:
    """Build the app around a pipeline (injected so tests can use a FakeLLM)."""
    app = FastAPI(title="text2sql-rag-analyst")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: AskRequest):
        return pipeline.answer(req.question).to_dict()

    return app
