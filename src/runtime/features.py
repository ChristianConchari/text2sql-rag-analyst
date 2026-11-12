"""Feature flags of the SQL route, loaded from configs/pipeline/sql_features.yaml.

Defaults (all on) reproduce the full system. Turning a flag off removes a *helper* (glossary, value lookup, the
semantic critic, retries); it never removes a safety rule. See the YAML header for the safety/critic split.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "pipeline" / "sql_features.yaml"


@dataclass(frozen=True)
class SQLFeatures:
    glossary: bool = True
    value_lookup: bool = True
    critic: bool = True
    retries: bool = True
    max_retries: int = 3

    @property
    def effective_retries(self) -> int:
        """Retries the loop may use: 0 when `retries` is off."""
        return self.max_retries if self.retries else 0

    def with_overrides(self, **kw: object) -> SQLFeatures:
        return replace(self, **kw)


def features_from_dict(raw: dict | None) -> SQLFeatures:
    """Build from a mapping; unknown keys raise (a typo must not silently leave a flag on)."""
    raw = raw or {}
    unknown = set(raw) - set(SQLFeatures.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown SQL feature flags: {sorted(unknown)}")
    return SQLFeatures(**{k: (int(v) if k == "max_retries" else bool(v)) for k, v in raw.items()})


def load_features(path: str | Path = DEFAULT_PATH) -> SQLFeatures:
    """Read the YAML; a missing file means defaults (everything on)."""
    p = Path(path)
    if not p.exists():
        return SQLFeatures()
    return features_from_dict(yaml.safe_load(p.read_text(encoding="utf-8")))
