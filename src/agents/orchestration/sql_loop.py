"""Generator <-> verifier loop: ask the model for SQL, verify, execute, and feed failures back.

Guarantees: only SQL returned by `verify` is ever executed, and the rows in the result are already masked by
the output guardrail. Gives up after `1 + max_retries` model calls. Does not judge whether a valid query
actually answers the question.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.policies.output_guardrail import mask_rows
from src.policies.sql_verifier import DEFAULT_DIALECT, Policy, Verdict, verify
from src.runtime.providers.llm_client import ChatModel, LLMError
from src.runtime.trace import (EXECUTOR, GUARDRAIL, SQL_GENERATOR, SQL_VERIFIER, Trace, describe_sql,
                               describe_violations)
from src.toolkits.db.executor import SQLExecutionError

Execute = Callable[[str], tuple[list[str], list[tuple[Any, ...]]]]


@dataclass
class Attempt:
    """One model reply and what happened to it."""

    raw: str
    sql: str | None
    violations: list[str] = field(default_factory=list)
    error: str | None = None
    advisory: list[str] = field(default_factory=list)   # critic findings not enforced (critic=False); see Verdict


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
                 max_retries: int = 3, dialect: str = DEFAULT_DIALECT, context: str | None = None,
                 critic: bool = True, max_completion_tokens: int = 700, trace: Trace | None = None) -> LoopResult:
    """Generate, verify and execute SQL for `question`.

    status "ok": `sql`/`columns`/`rows` are set (rows masked). "declined": the model answered NO_SQL (the
    question cannot be answered with the data). "failed": retries exhausted or the model call failed.
    `context` (e.g. real values found by the value lookup) is appended to the question as data for the generator.
    `critic=False` turns off only the semantic critic (order-grain rule); safety rules and masking always apply.
    `max_retries=0` means a single attempt: any rejection or execution error ends in "failed".
    `trace`: steps are appended to it (generator, verifier, executor, guardrail); see src/runtime/trace.py for
    what a step may contain. Execution errors are recorded as a fixed phrase, never the database's message.
    """
    tr = trace if trace is not None else Trace()
    user = question if not context else f"{question}\n\n{context}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    res = LoopResult(status="failed")
    for n in range(1, 2 + max_retries):
        t0 = time.perf_counter()
        try:
            raw = llm.chat(messages, max_tokens=max_completion_tokens, temperature=0)
        except LLMError as e:
            tr.add(SQL_GENERATOR, f"Intento {n}: el modelo no respondió", "error", t0, n)
            res.reason = f"llm_error: {e}"
            return res
        if raw.strip().upper().startswith("NO_SQL"):
            tr.add(SQL_GENERATOR, f"Intento {n}: declina, los datos no permiten responder", "ok", t0, n)
            res.status, res.reason = "declined", raw.strip()[len("NO_SQL"):].lstrip(": ").strip() or None
            return res
        sql = extract_sql(raw)
        tr.add(SQL_GENERATOR, f"Intento {n}: {describe_sql(sql, policy.allowed_tables, dialect)}", "ok", t0, n)
        att = Attempt(raw=raw, sql=sql)
        res.attempts.append(att)
        feedback = None
        t0 = time.perf_counter()
        if sql is None:
            att.violations, feedback = ["empty_response"], "La respuesta no contiene SQL. Responde con un bloque ```sql."
            tr.add(SQL_VERIFIER, "Objeta: " + describe_violations(att.violations), "objection", t0, n)
        else:
            verdict: Verdict = verify(sql, policy, dialect=dialect, critic=critic)
            att.advisory = verdict.advisory
            if not verdict.ok:
                att.violations = verdict.violations
                feedback = "El verificador rechazó la consulta: " + "; ".join(verdict.violations) + ". Corrígela."
                tr.add(SQL_VERIFIER, "Objeta: " + describe_violations(verdict.violations), "objection", t0, n)
            else:
                tr.add(SQL_VERIFIER, "Aprueba: solo lectura, tablas y columnas permitidas, LIMIT aplicado",
                       "ok", t0, n)
                t0 = time.perf_counter()
                try:
                    cols, rows = execute(verdict.sql)
                except SQLExecutionError as e:
                    # The message can quote data (e.g. a supplier name in a conversion error): never trace it.
                    tr.add(EXECUTOR, "La base de datos rechazó la consulta; se pide corregirla", "error", t0, n)
                    att.error = str(e)
                    feedback = f"La ejecución falló: {str(e)[:300]}. Corrige la consulta."
                else:
                    tr.add(EXECUTOR, f"{len(rows)} {'fila' if len(rows) == 1 else 'filas'}, DuckDB de solo lectura",
                           "ok", t0, n)
                    res.status, res.sql, res.columns = "ok", verdict.sql, cols
                    t0 = time.perf_counter()
                    res.rows = mask_rows(cols, rows, verdict.masked_columns, verdict.flag_column)
                    if verdict.masked_columns:
                        hidden = sum(a != b for r_in, r_out in zip(rows, res.rows) for a, b in zip(r_in, r_out))
                        tr.add(GUARDRAIL, f"{hidden} {'valor enmascarado' if hidden == 1 else 'valores enmascarados'}"
                               " (personas naturales)", "ok", t0, n)
                    res.warnings = verdict.warnings
                    return res
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}]
    res.reason = "max_retries_exceeded"
    return res
