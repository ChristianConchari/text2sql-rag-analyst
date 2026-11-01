"""Output guardrail: masks personal data in query results before anything downstream sees them.

Complements `sql_verifier`, which only sees the SQL, never the results. Runs between the database and both the
answer composer (an LLM) and the API response, so masked values never reach either.

Rules:
  1. Every column in `masked_columns` (aliases included, from `Verdict.masked_columns`) is replaced by `MASK`
     in each row, unless that row's flag column is exactly `False`.
     Why: the flag says "legal entity"; only then is the name public information.
  2. Fail-closed: flag `True`, `None` (unparseable RUT), any non-`False` value (0, "false", ""), or a flag column
     missing from the result all mask. Showing a value requires positive evidence.
  3. `masked_columns=None` means "unknown" and falls back to `NAME_COLUMN`; an empty list means "nothing to mask".

Limits: it masks by column, not by content. Personal data inside free-text columns is not detected here.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MASK = "[ENMASCARADO: persona natural]"
NAME_COLUMN = "NombreProveedor"       # fallback when the caller does not know which columns to mask
FLAG_COLUMN = "es_persona_natural"    # fallback flag column; the policy's `masking.flag_column` overrides it


def mask_rows(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    masked_columns: Sequence[str] | None = None,
    flag_column: str | None = None,
) -> list[tuple[Any, ...]]:
    """Return `rows` with `masked_columns` replaced by `MASK` unless the row's flag is exactly False.

    Args:
        columns: result column names, in row order (case-insensitive match).
        rows: result rows.
        masked_columns: output names to mask; None = [NAME_COLUMN]; [] = mask nothing.
        flag_column: name of the per-row flag column; None = FLAG_COLUMN.
    """
    targets = {c.lower() for c in (masked_columns if masked_columns is not None else [NAME_COLUMN])}
    lowered = [c.lower() for c in columns]
    masked_idx = [i for i, c in enumerate(lowered) if c in targets]
    if not masked_idx:
        return [tuple(r) for r in rows]
    flag = (flag_column or FLAG_COLUMN).lower()
    flag_idx = lowered.index(flag) if flag in lowered else None

    out = []
    for row in rows:
        row = list(row)
        if flag_idx is None or row[flag_idx] is not False:
            for i in masked_idx:
                row[i] = MASK
        out.append(tuple(row))
    return out
