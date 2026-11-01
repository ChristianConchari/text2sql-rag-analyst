"""Masking plan for the SQL verifier: which output columns carry masked data, and keeping the flag with them.

The output guardrail masks `policy.mask_column` per row using `policy.flag_column`. For that, the flag must be
in the result set. This module (a) finds the output columns that expose the masked column and (b) appends the
flag to the query when that is safe.

Fail-closed: when the flag cannot be added safely (CTE, subquery, expression or aggregate over the column), the
query is left as is and a `mask_without_flag` warning is emitted; the guardrail then masks every row.
`COUNT(...)` over the column exposes no value and is ignored.
"""
from __future__ import annotations

from typing import NamedTuple

from sqlglot import exp

from src.policies.policy import Policy


class MaskingResult(NamedTuple):
    masked_columns: list[str]   # output names (aliases included) the guardrail must mask
    warnings: list[str]


def leaf_selects(node: exp.Expression) -> list[exp.Select]:
    """The SELECTs at the leaves of a (possibly nested) UNION/INTERSECT/EXCEPT; `[node]` for a plain SELECT."""
    if isinstance(node, exp.Select):
        return [node]
    if isinstance(node, exp.SetOperation):
        return leaf_selects(node.left) + leaf_selects(node.right)
    return []


def _exposed_projections(leaf: exp.Select, mask_column: str) -> list[exp.Expression]:
    """Projections that reference `mask_column` outside a COUNT."""
    return [
        p for p in leaf.expressions
        if any(c.name.lower() == mask_column and c.find_ancestor(exp.Count) is None for c in p.find_all(exp.Column))
    ]


def _is_bare_column(projection: exp.Expression) -> bool:
    inner = projection.this if isinstance(projection, exp.Alias) else projection
    return isinstance(inner, exp.Column)


def _is_flat_query(tree: exp.Expression) -> bool:
    """True when the query reads the base table directly (no CTE, no subquery)."""
    return tree.find(exp.With) is None and tree.find(exp.Subquery) is None


def _append_flag(leaf: exp.Select, flag_column: str) -> bool:
    """Add the flag to the projections (and GROUP BY) of one SELECT. False when it was already projected."""
    if any(e.alias_or_name.lower() == flag_column for e in leaf.expressions):
        return False
    leaf.append("expressions", exp.column(flag_column))
    group = leaf.args.get("group")
    if group is not None:
        group.append("expressions", exp.column(flag_column))
    return True


def prepare_masking(tree: exp.Expression, policy: Policy) -> MaskingResult:
    """Return the output columns to mask and, when safe, rewrite `tree` in place so the flag travels with them."""
    if not policy.mask_column or not policy.flag_column:
        return MaskingResult([], [])
    leaves = leaf_selects(tree)
    exposed = _exposed_projections(leaves[0], policy.mask_column) if leaves else []
    if not exposed:
        return MaskingResult([], [])

    masked = [p.alias_or_name for p in exposed]
    if not (all(_is_bare_column(p) for p in exposed) and _is_flat_query(tree)):
        return MaskingResult(masked, [f"mask_without_flag: {policy.mask_column} will be fully masked in the output"])
    warnings = [f"flag_added: {policy.flag_column}" for leaf in leaves if _append_flag(leaf, policy.flag_column)]
    return MaskingResult(masked, warnings)
