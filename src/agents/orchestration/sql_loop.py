"""Generator <-> verifier loop: ask the model for SQL, verify, execute, and feed failures back.

Guarantees: only SQL returned by `verify` is ever executed, and the rows in the result are already masked by
the output guardrail. Gives up after `1 + max_retries` model calls. Does not judge whether a valid query
actually answers the question.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.policies.output_guardrail import mask_rows
from src.policies.sql_verifier import DEFAULT_DIALECT, Policy, Verdict, verify
from src.runtime.providers.llm_client import ChatModel, LLMError
from src.toolkits.db.executor import SQLExecutionError

Execute = Callable[[str], tuple[list[str], list[tuple[Any, ...]]]]


@dataclass
class Attempt:
    """One model reply and what happened to it."""

    raw: str
    sql: str | None
    violations: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class LoopResult:
    """Final outcome; `attempts` keeps the full history for debugging and evaluation."""

    status: str                      # ok | declined | failed
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)   # ALREADY masked by the output guardrail
    warnings: list[str] = field(default_factory=list)
    reason: str | None = None
    attempts: list[Attempt] = field(default_factory=list)


def extract_sql(raw: str) -> str | None:
    """The SQL inside a ```sql fence, else the whole reply; None if empty."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.S | re.I)
    text = (m.group(1) if m else raw).strip()
    return text or None


def run_sql_loop(question: str, llm: ChatModel, policy: Policy, execute: Execute, system_prompt: str,
                 max_retries: int = 3, dialect: str = DEFAULT_DIALECT, context: str | None = None) -> LoopResult:
    """Generate, verify and execute SQL for `question`.

    status "ok": `sql`/`columns`/`rows` are set (rows masked). "declined": the model answered NO_SQL (the
    question cannot be answered with the data). "failed": retries exhausted or the model call failed.
    `context` (e.g. real values found by the value lookup) is appended to the question as data for the generator.
    """
    user = question if not context else f"{question}\n\n{context}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    res = LoopResult(status="failed")
    for _ in range(1 + max_retries):
        try:
            raw = llm.chat(messages, max_tokens=700, temperature=0)
        except LLMError as e:
            res.reason = f"llm_error: {e}"
            return res
        if raw.strip().upper().startswith("NO_SQL"):
            res.status, res.reason = "declined", raw.strip()[len("NO_SQL"):].lstrip(": ").strip() or None
            return res
        sql = extract_sql(raw)
        att = Attempt(raw=raw, sql=sql)
        res.attempts.append(att)
        feedback = None
        if sql is None:
            att.violations, feedback = ["empty_response"], "La respuesta no contiene SQL. Responde con un bloque ```sql."
        else:
            verdict: Verdict = verify(sql, policy, dialect=dialect)
            if not verdict.ok:
                att.violations = verdict.violations
                feedback = "El verificador rechazó la consulta: " + "; ".join(verdict.violations) + ". Corrígela."
            else:
                try:
                    cols, rows = execute(verdict.sql)
                except SQLExecutionError as e:
                    att.error = str(e)
                    feedback = f"La ejecución falló: {str(e)[:300]}. Corrige la consulta."
                else:
                    res.status, res.sql, res.columns = "ok", verdict.sql, cols
                    res.rows = mask_rows(cols, rows, verdict.masked_columns, verdict.flag_column)
                    res.warnings = verdict.warnings
                    return res
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}]
    res.reason = "max_retries_exceeded"
    return res
