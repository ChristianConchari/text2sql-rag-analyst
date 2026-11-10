"""SQL policy: the domain-specific knobs the verifier enforces, loaded from configs/domain/*.yaml.

Nothing domain-specific is hardcoded in the verifier; adding a domain means adding a YAML file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_LIMIT = 100    # used only when the YAML omits `limits.default`
MAX_LIMIT = 1000       # used only when the YAML omits `limits.max`


@dataclass(frozen=True)
class Policy:
    """Immutable policy. All name sets are lower-cased; comparisons are case-insensitive."""

    allowed_tables: frozenset[str]
    default_limit: int = DEFAULT_LIMIT
    max_limit: int = MAX_LIMIT
    blocked_columns: frozenset[str] = frozenset()      # query is rejected if it mentions them
    restricted_columns: frozenset[str] = frozenset()   # allowed, but the verifier emits a warning
    filter_only_columns: frozenset[str] = frozenset()  # allowed only inside WHERE, in a whitelisted text predicate
    order_level_columns: frozenset[str] = frozenset()  # repeated on every line of an order: never SUM/AVG over lines
    order_key: str | None = None                         # column that identifies the order (deduplicate by it)
    mask_column: str | None = None   # column whose values are masked for natural persons
    flag_column: str | None = None   # per-row flag that decides the masking (travels with mask_column)


def _lower_set(values: list[str] | None) -> frozenset[str]:
    return frozenset(v.lower() for v in (values or []))


def load_policy(path: str | Path) -> Policy:
    """Read a domain YAML (see configs/domain/chilecompra.yaml) into a `Policy`. `allowed_tables` is required."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    limits = cfg.get("limits") or {}
    masking = cfg.get("masking") or {}
    grain = cfg.get("order_grain") or {}
    return Policy(
        allowed_tables=_lower_set(cfg["allowed_tables"]),
        default_limit=int(limits.get("default", DEFAULT_LIMIT)),
        max_limit=int(limits.get("max", MAX_LIMIT)),
        blocked_columns=_lower_set(cfg.get("blocked_columns")),
        restricted_columns=_lower_set(cfg.get("restricted_columns")),
        filter_only_columns=_lower_set(cfg.get("filter_only_columns")),
        order_level_columns=_lower_set(grain.get("order_level_columns")),
        order_key=grain.get("order_key", "").lower() or None,
        mask_column=masking.get("column", "").lower() or None,
        flag_column=masking.get("flag_column", "").lower() or None,
    )
