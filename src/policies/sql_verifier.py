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
  select_star_exposes_blocked_columns  `*` would return the blocked or filter-only columns (COUNT(*) is fine)
  filter_only_outside_where          a filter-only column (free text with personal data) used outside WHERE
  filter_only_predicate              a filter-only column inside WHERE but not in a whitelisted text predicate
                                     against string literals; anything else could echo the value (e.g. a failing
                                     CAST puts the text in DuckDB's error message, which goes back to the LLM)
  whole_row_reference                `SELECT t FROM tbl t` returns the whole row as a struct, bypassing column checks
  limit_not_integer_literal          LIMIT must be an integer literal so it can be clamped
  order_level_aggregate_over_lines   SUM/AVG/MEDIAN/... of an order-level column (repeated on every line) whose rows
                                     are not one per order: it must read from a subquery/CTE that groups by the
                                     order key or is DISTINCT (also rejects SUM(DISTINCT amount)). Name-based: an
                                     alias of the column in a CTE without deduplication is not detected.
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


# Filter-only columns: the only shapes allowed (column, optionally wrapped, compared with string literals).
FILTER_ONLY_WRAPPERS = (exp.Lower, exp.Upper, exp.Trim)
FILTER_ONLY_WRAPPER_FUNCS = frozenset({"strip_accents"})
# Aggregates whose result changes when an order's value is repeated once per line.
GRAIN_SENSITIVE_AGGS = tuple(getattr(exp, n) for n in (
    "Sum", "Avg", "Median", "Quantile", "PercentileCont", "PercentileDisc", "Stddev", "StddevPop", "StddevSamp",
    "Variance", "VariancePop") if hasattr(exp, n))
MAX_SOURCE_DEPTH = 10

FILTER_ONLY_PREDICATES = (exp.Like, exp.ILike, exp.EQ, exp.NEQ, exp.In, exp.Is, exp.RegexpLike, exp.Contains)


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
                         + table_violations + column_violations + _filter_only_violations(tree, policy)
                         + _order_grain_violations(tree, policy))
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
            if not isinstance(node.parent, exp.Count) and (policy.blocked_columns or policy.filter_only_columns):
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


def _filter_only_violations(tree: exp.Expression, policy: Policy) -> list[str]:
    """Filter-only columns may appear only in WHERE, as `[wrappers(]col[)] <predicate> <string literals>`."""
    found: list[str] = []
    for column in tree.find_all(exp.Column):
        if column.name.lower() not in policy.filter_only_columns:
            continue
        predicate = _unwrap_to_predicate(column)
        if predicate is None or not _compares_with_string_literals(predicate):
            found.append(f"filter_only_predicate: {column.name}")
        elif not _inside_where(predicate):
            found.append(f"filter_only_outside_where: {column.name}")
    return found


def _unwrap_to_predicate(column: exp.Column) -> exp.Expression | None:
    """Climb from the column through allowed wrappers (as their only argument); return the predicate or None."""
    node: exp.Expression = column
    while True:
        parent = node.parent
        if (isinstance(parent, exp.Anonymous) and parent.name.lower() in FILTER_ONLY_WRAPPER_FUNCS
                and len(parent.expressions) == 1 and parent.expressions[0] is node):
            node = parent
            continue
        if parent is None or parent.args.get("this") is not node:
            return None   # the column is not the predicate's subject (e.g. it is the pattern, or a function arg)
        if isinstance(parent, FILTER_ONLY_WRAPPERS):
            node = parent
            continue
        return parent if isinstance(parent, FILTER_ONLY_PREDICATES) else None


def _compares_with_string_literals(predicate: exp.Expression) -> bool:
    if isinstance(predicate, exp.Is):
        return isinstance(predicate.expression, exp.Null)
    if isinstance(predicate, exp.In):
        values = predicate.expressions
        return bool(values) and predicate.args.get("query") is None and all(_is_string_literal(v) for v in values)
    if isinstance(predicate, (exp.Like, exp.ILike)) and predicate.args.get("escape") is not None:
        return False
    return _is_string_literal(predicate.expression)


def _is_string_literal(node: exp.Expression | None) -> bool:
    return isinstance(node, exp.Literal) and node.is_string


def _inside_where(node: exp.Expression) -> bool:
    """True if the nearest enclosing clause is a WHERE (not a projection, GROUP BY, ORDER BY, HAVING, JOIN ...)."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Where):
            return True
        if isinstance(parent, (exp.Select, exp.Group, exp.Order, exp.Having, exp.Join, exp.Window, exp.Qualify)):
            return False
        parent = parent.parent
    return False


def _order_grain_violations(tree: exp.Expression, policy: Policy) -> list[str]:
    """Grain-sensitive aggregates over order-level columns must read one row per order."""
    if not policy.order_level_columns:
        return []
    ctes = {c.alias.lower(): c.this for c in tree.find_all(exp.CTE)}
    found: list[str] = []
    for agg in tree.find_all(*GRAIN_SENSITIVE_AGGS):
        if not any(c.name.lower() in policy.order_level_columns for c in agg.find_all(exp.Column)):
            continue
        select = agg.find_ancestor(exp.Select)
        if isinstance(agg.this, exp.Distinct) or select is None or not _reads_one_row_per_order(select, policy, ctes, 0):
            found.append(_grain_feedback(agg, select, policy))
    return found


def _grain_feedback(agg: exp.Expression, select: exp.Select | None, policy: Policy) -> str:
    """Violation text written for the generator: what is wrong and the shape of the fix."""
    key = policy.order_key or "la clave de la orden"
    msg = (f"order_level_aggregate_over_lines: {agg.sql()} agrega un valor de nivel orden que se repite en cada "
           f"línea. Deduplica primero por {key} en un CTE que traiga TODO lo que necesitas, una fila por orden: "
           f"WITH o AS (SELECT {key}, any_value(<monto>) AS monto, any_value(<columna para filtrar o agrupar>) AS ... "
           f"FROM ... WHERE ... GROUP BY {key}) y agrega sobre o, sin volver a unir con la tabla de líneas")
    if select is not None and select.args.get("joins"):
        msg += ". Un JOIN con la tabla de líneas vuelve a repetir cada orden"
    return msg


def _reads_one_row_per_order(select: exp.Select, policy: Policy, ctes: dict[str, exp.Expression], depth: int) -> bool:
    """True if the FROM source of `select` is a subquery/CTE with one row per order (GROUP BY key or DISTINCT)."""
    from_ = select.args.get("from_") or select.args.get("from")
    if from_ is None or select.args.get("joins") or depth > MAX_SOURCE_DEPTH:
        return False
    source = from_.this
    if isinstance(source, exp.Subquery):
        inner = source.this
    elif isinstance(source, exp.Table) and source.name.lower() in ctes:
        inner = ctes[source.name.lower()]
    else:
        return False   # the base table: one row per line
    return _is_one_row_per_order(inner, policy, ctes, depth + 1)


def _is_one_row_per_order(query: exp.Expression, policy: Policy, ctes: dict[str, exp.Expression], depth: int) -> bool:
    if not isinstance(query, exp.Select):
        return False   # UNION and friends: not analysed, fail closed
    key = policy.order_key
    group = query.args.get("group")
    if key and group and any(isinstance(e, exp.Column) and e.name.lower() == key for e in group.expressions):
        return True
    distinct = query.args.get("distinct")
    if key and distinct is not None:
        on = distinct.args.get("on")
        keys_in = on.expressions if on is not None else query.expressions   # DISTINCT ON (key) / DISTINCT key, ...
        if any(isinstance(e.unalias(), exp.Column) and e.unalias().name.lower() == key for e in keys_in):
            return True
    if group:
        return False   # grouped by something else: not one row per order
    return _reads_one_row_per_order(query, policy, ctes, depth)   # a plain filter over a deduplicated source


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
