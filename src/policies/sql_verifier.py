"""Deterministic SQL verifier: decides whether an LLM-generated query may run, and normalizes it.

`verify(sql, policy, dialect)` is a pure function: no LLM, no database access, no I/O. It guarantees that an
accepted query is ONE read-only SELECT (or UNION of SELECTs) over allowed tables only, with an integer LIMIT
<= policy.max_limit, free of comments and of the blocked constructs below. It does NOT check semantics (is the
answer right?) or cost (a heavy query inside the LIMIT still runs). Result masking is the output guardrail's job.

Fail-closed: anything unparseable, ambiguous or not on the allow-list is rejected, never repaired. The only
rewrites are the LIMIT (default/clamp) and the masking flag (see `masking.py`).

Rules (violation code -> why):
  empty_query                        nothing to run
  parse_error                        sqlglot cannot parse it (also covers PRAGMA/EXPORT/INSTALL-style statements)
  comments_not_allowed               comments can smuggle instructions to downstream LLMs (checked on tokens and,
                                     redundantly, on the raw text outside string literals)
  multiple_statements                stacked statements are the classic injection
  not_select                         only SELECT/WITH...SELECT/UNION; DDL, DML, COPY, ATTACH, SET, ... never run
  forbidden_statement                a forbidden statement type nested anywhere (e.g. SELECT ... INTO)
  blocked_function                   file/network/env access: read_*, *_scan, glob, getenv, duckdb_*, COLUMNS(...), ...
  table_function_or_file_source      FROM read_csv('x') or FROM 'file.csv' (DuckDB replacement scan)
  table_not_allowed                  table is neither in policy.allowed_tables nor a CTE defined in the query
  qualified_table_not_allowed        db.table / catalog.db.table could reach another attached database
  blocked_column                     personal-data columns listed in the policy
  select_star_exposes_blocked_columns  `*` would return the blocked columns (COUNT(*) is fine)
  whole_row_reference                `SELECT t FROM tbl t` returns the whole row as a struct, bypassing column checks
  limit_not_integer_literal          LIMIT must be an integer literal so it can be clamped
Warnings (query still accepted): restricted_column, flag_added, mask_without_flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from src.policies.masking import prepare_masking
from src.policies.policy import Policy, load_policy

__all__ = ["DEFAULT_DIALECT", "Policy", "Verdict", "load_policy", "verify"]

DEFAULT_DIALECT = "duckdb"

# File / network / environment access (DuckDB). Also any read_*/ *_scan function.
BLOCKED_FUNCTIONS = frozenset({
    "glob", "sniff_csv", "query", "query_table", "getenv", "current_setting",
    "load_extension", "install_extension", "http_get", "http_post", "columns",
    "parquet_metadata", "parquet_schema", "parquet_file_metadata", "iceberg_scan",
    "delta_scan", "duckdb_settings", "duckdb_extensions", "pragma_database_list",
    "write_csv", "write_parquet", "sqlite_scan", "postgres_scan", "mysql_scan",
})
BLOCKED_FUNCTION_RE = re.compile(r"^(read_.*|.*_scan|pragma_.*|duckdb_.*)$")

FORBIDDEN_NODES = tuple(
    getattr(exp, n) for n in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
        "Copy", "Attach", "Detach", "Pragma", "Command", "Set", "Use", "Into", "Transaction",
        "Commit", "Rollback", "Export", "LoadData", "Install", "Describe", "Show",
    ) if hasattr(exp, n)
)



@dataclass
class Verdict:
    """Outcome of `verify`. When `ok` is False, `sql` is None and `violations` explains why."""

    ok: bool
    sql: str | None = None          # normalized SQL with the LIMIT enforced (only if ok)
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Output column names (aliases included) that carry masked data; the output guardrail masks them.
    masked_columns: list[str] = field(default_factory=list)
    flag_column: str | None = None


def _reject(*violations: str) -> Verdict:
    return Verdict(False, violations=list(violations))


def verify(sql: str, policy: Policy, dialect: str = DEFAULT_DIALECT) -> Verdict:
    """Check `sql` against `policy`. Returns a `Verdict`; the caller must run `verdict.sql`, never the input."""
    if not sql or not sql.strip():
        return _reject("empty_query")
    try:
        comment_violations = _comment_violations(sql, dialect)
    except SqlglotError as e:
        return _reject(f"parse_error: {e}")
    tree, structure_violations = _parse_single_select(sql, dialect)
    if tree is None:
        return _reject(*comment_violations, *structure_violations)

    table_violations, referenced_names = _table_violations(tree, policy)
    column_violations, warnings = _column_findings(tree, policy, referenced_names)
    violations = _dedupe(comment_violations + _forbidden_statement_violations(tree) + _function_violations(tree)
                         + table_violations + column_violations)
    if violations:
        return Verdict(False, violations=violations, warnings=warnings)

    limit_violation = _enforce_limit(tree, policy)
    if limit_violation:
        return Verdict(False, violations=[limit_violation], warnings=warnings)

    masking = prepare_masking(tree, policy)
    return Verdict(True, sql=tree.sql(dialect=dialect), warnings=warnings + masking.warnings,
                   masked_columns=masking.masked_columns,
                   flag_column=policy.flag_column if masking.masked_columns else None)


def _comment_violations(sql: str, dialect: str) -> list[str]:
    """Reject every comment: via tokens, and via raw text with string literals blanked out (defence in depth)."""
    has_comment = any(t.comments for t in sqlglot.tokenize(sql, read=dialect))
    has_comment = has_comment or bool(re.search(r"(--|/\*)", _blank_string_literals(sql)))
    return ["comments_not_allowed"] if has_comment else []


def _parse_single_select(sql: str, dialect: str) -> tuple[exp.Expression | None, list[str]]:
    """Parse exactly one SELECT-like statement; otherwise (None, [reason])."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as e:
        return None, [f"parse_error: {e}"]
    if not statements:
        return None, ["empty_query"]
    if len(statements) > 1:
        return None, ["multiple_statements"]
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return None, [f"not_select: {type(tree).__name__}"]
    return tree, []


def _forbidden_statement_violations(tree: exp.Expression) -> list[str]:
    return [f"forbidden_statement: {type(n).__name__}" for n in tree.walk() if isinstance(n, FORBIDDEN_NODES)]


def _function_violations(tree: exp.Expression) -> list[str]:
    found: list[str] = []
    for node in tree.walk():
        if isinstance(node, exp.Table) and not isinstance(node.this, exp.Identifier):
            found.append("table_function_or_file_source")   # read_csv('x') in FROM, or FROM 'file.csv'
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in BLOCKED_FUNCTIONS or BLOCKED_FUNCTION_RE.match(name):
                found.append(f"blocked_function: {name}")
        if isinstance(node, exp.Columns):
            found.append("blocked_function: columns")
    return found


def _table_violations(tree: exp.Expression, policy: Policy) -> tuple[list[str], set[str]]:
    """Check every table against the allow-list. Also returns all table/alias/CTE names seen (lower-cased)."""
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    names: set[str] = set(ctes)
    found: list[str] = []
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue   # table functions / file sources are reported by _function_violations
        name = table.name.lower()
        if table.args.get("db") or table.args.get("catalog"):
            found.append(f"qualified_table_not_allowed: {table.sql()}")
        elif name not in ctes and name not in policy.allowed_tables:
            found.append(f"table_not_allowed: {table.name}")
        names.add(name)
        if table.alias:
            names.add(table.alias.lower())
    return found, names


def _column_findings(tree: exp.Expression, policy: Policy, table_names: set[str]) -> tuple[list[str], list[str]]:
    """Return (violations, warnings) for blocked/restricted columns, `*` and whole-row references."""
    violations: list[str] = []
    warnings: list[str] = []
    for node in tree.walk():
        if isinstance(node, exp.Star):
            if not isinstance(node.parent, exp.Count) and policy.blocked_columns:
                violations.append("select_star_exposes_blocked_columns")
        elif isinstance(node, exp.Column):
            column = node.name.lower()
            if column in policy.blocked_columns:
                violations.append(f"blocked_column: {node.name}")
            elif column in policy.restricted_columns:
                warnings.append(f"restricted_column: {node.name} (mask in the answer)")
            elif not node.table and column in table_names:
                violations.append(f"whole_row_reference: {node.name}")
    return violations, warnings


def _enforce_limit(tree: exp.Expression, policy: Policy) -> str | None:
    """Inject the default LIMIT or clamp a larger one, in place. Returns a violation code if LIMIT is not an int."""
    limit = tree.args.get("limit")
    if limit is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(policy.default_limit)))
        return None
    value = limit.expression
    if not (isinstance(value, exp.Literal) and value.is_int):
        return "limit_not_integer_literal"
    if int(value.name) > policy.max_limit:
        limit.set("expression", exp.Literal.number(policy.max_limit))
    return None


def _blank_string_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "''", sql)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
