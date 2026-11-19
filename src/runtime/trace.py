"""Per-request agent trace: what each agent did, in order, with its outcome and duration.

Shown to the user ("Cómo lo resolvieron los agentes") and meant to be the basis of the audit log, so it is held
to a strict rule: a step's `detail` is ALWAYS text we write ourselves (fixed phrases, counts, violation codes,
schema identifiers from the policy). It never carries row values, raw database or parser error text, model
output, or the question itself: any of those can hold the name of a natural person.

Helpers here turn pipeline facts into safe details; callers must not build details from free text.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

OUTCOMES = ("ok", "objection", "error", "skipped")

# Agent names as the user sees them.
ROUTER, SQL_GENERATOR, SQL_VERIFIER, EXECUTOR, GUARDRAIL, COMPOSER = (
    "Enrutador", "Generador SQL", "Verificador SQL", "Ejecutor", "Guardia de salida", "Redactor")
RETRIEVER, DOC_WRITER, CITATION_VERIFIER, FIXED_REPLY = (
    "Recuperador", "Redactor", "Verificador de citas", "Respuesta fija")

# Verifier violation code -> short Spanish reason. Codes come from src/policies/{sql,citation}_verifier.py.
VIOLATION_REASONS = {
    "empty_query": "no trae SQL",
    "empty_response": "no trae SQL",
    "parse_error": "SQL no válido",
    "comments_not_allowed": "contiene comentarios",
    "multiple_statements": "más de una sentencia",
    "not_select": "no es una consulta SELECT",
    "forbidden_statement": "sentencia prohibida",
    "table_function_or_file_source": "lee archivos o funciones de tabla",
    "blocked_function": "usa una función bloqueada",
    "qualified_table_not_allowed": "tabla no permitida",
    "table_not_allowed": "tabla no permitida",
    "select_star_exposes_blocked_columns": "SELECT * expondría columnas bloqueadas",
    "blocked_column": "usa una columna bloqueada",
    "whole_row_reference": "referencia a la fila completa",
    "filter_only_predicate": "columna solo para filtros usada de otra forma",
    "filter_only_outside_where": "columna solo para filtros fuera del WHERE",
    "order_level_aggregate_over_lines": "agrega un valor de la orden en cada línea; deduplicar por orden",
    "limit_not_integer_literal": "LIMIT no es un entero",
    "no_citation": "la respuesta no cita ningún artículo",
    "invented_citation": "cita un artículo que no fue recuperado",
}
_CODE = re.compile(r"^[a-z_]+$")


@dataclass
class TraceStep:
    agent: str
    detail: str
    outcome: str = "ok"           # one of OUTCOMES
    duration_ms: float = 0.0
    attempt: int | None = None


@dataclass
class Trace:
    """Collects steps in order. Pipelines and loops take `trace=None` and work the same without one."""

    steps: list[TraceStep] = field(default_factory=list)

    def add(self, agent: str, detail: str, outcome: str = "ok", since: float | None = None,
            attempt: int | None = None) -> None:
        """Append a step; `since` is a `time.perf_counter()` value taken when the step started."""
        assert outcome in OUTCOMES, outcome
        ms = round((time.perf_counter() - since) * 1000, 1) if since is not None else 0.0
        self.steps.append(TraceStep(agent, detail, outcome, ms, attempt))

    def to_list(self) -> list[dict]:
        return [asdict(s) for s in self.steps]


def violation_code(violation: str) -> str:
    """The code part of a violation string ("blocked_column: X" -> "blocked_column"), or "policy_rule"."""
    code = violation.split(":", 1)[0].strip()
    return code if _CODE.match(code) else "policy_rule"


def describe_violations(violations: list[str]) -> str:
    """Spanish reasons for violation strings, using only their codes (the text after ':' may quote the query)."""
    reasons = [VIOLATION_REASONS.get(violation_code(v), violation_code(v)) for v in violations]
    return "; ".join(dict.fromkeys(reasons)) or "regla de política"


def describe_sql(sql: str | None, allowed_tables: frozenset[str] | set[str], dialect: str = "duckdb") -> str:
    """A safe summary of a query: aggregate function names and allowed table names, never literals.

    E.g. "COUNT(DISTINCT) sobre oc_lineas". Falls back to "consulta SQL" when nothing safe can be said.
    """
    if not sql:
        return "respuesta sin SQL"
    try:
        tree = parse_one(sql, read=dialect)
    except SqlglotError:
        return "consulta SQL (no se pudo analizar)"
    if tree is None:
        return "consulta SQL"
    allowed = {t.lower() for t in allowed_tables}
    aggs = []
    for node in tree.find_all(exp.AggFunc):
        name = (node.key or "").upper()
        if not _CODE.match(name.lower()):
            continue
        if node.find(exp.Distinct):
            name += "(DISTINCT)"
        aggs.append(name)
    tables = [t.name for t in tree.find_all(exp.Table) if t.name.lower() in allowed]
    parts = ", ".join(dict.fromkeys(aggs)) or "SELECT"
    where = " sobre " + ", ".join(dict.fromkeys(tables)) if tables else ""
    return parts + where
