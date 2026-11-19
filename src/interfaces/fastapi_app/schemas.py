"""Typed HTTP contract of /api. The OpenAPI schema generated from these models is what the frontend types are
built from (frontend `npm run gen:api`), so a change here is a contract change.
"""
from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

Route = Literal["sql", "docs", "out_of_scope", "refuse", "unknown"]
Status = Literal["ok", "declined", "failed", "refused", "out_of_scope", "no_evidence", "not_implemented",
                 "needs_clarification"]
Outcome = Literal["ok", "objection", "error", "skipped"]


class AskRequest(BaseModel):
    """Request body of POST /api/ask."""

    question: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    """A retrieved article cited by a docs answer (built from the corpus, never from model output)."""

    label: str = Field(description='Citation as shown in the answer, e.g. "Ley 19.886 art. 7"')
    norma: str = Field(description='e.g. "Ley 19886"')
    articulo: str = Field(description='Article number, e.g. "8 bis"')
    excerpt: str = Field(description="Start of the article text (at most 300 characters)")


class TraceStep(BaseModel):
    """One agent step. `detail` is short Spanish text written by the app: counts, codes and fixed phrases only,
    never row values, database error text or model output."""

    agent: str
    detail: str
    outcome: Outcome
    duration_ms: float
    attempt: int | None = None


class AskResponse(BaseModel):
    request_id: UUID
    route: Route
    status: Status
    answer: str
    sql: str | None = None
    columns: list[str] = []
    rows: list[list[Any]] = Field(default=[], description="Already masked by the output guardrail")
    warnings: list[str] = []
    attempts: int = 0
    citations: list[Citation] = []
    trace: list[TraceStep] = []
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
