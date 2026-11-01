"""Read-only DuckDB execution with external access disabled.

Defence in depth behind the SQL verifier: even a query that slipped through cannot write, read host files or
use the network. It does not limit memory or result size; row count is bounded by the verifier's LIMIT.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_TIMEOUT_S = 30.0


class SQLExecutionError(RuntimeError):
    """The database could not be opened, the query failed, or it hit the timeout."""


def run_query(db_path: str | Path, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Run `sql` on a fresh read-only connection. Returns (column names, rows); raises SQLExecutionError."""
    try:
        con = duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})
    except duckdb.Error as e:
        raise SQLExecutionError(f"cannot open database: {e}") from e
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        return [d[0] for d in cur.description], cur.fetchall()
    except duckdb.Error as e:
        raise SQLExecutionError(str(e)) from e
    finally:
        timer.cancel()
        con.close()
