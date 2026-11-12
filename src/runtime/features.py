"""Feature flags of the SQL route, loaded from configs/pipeline/sql_features.yaml.

Defaults (all on) reproduce the full system. Turning a flag off removes a *helper* (glossary, value lookup, the
semantic critic, retries); it never removes a safety rule. See the YAML header for the safety/critic split.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "pipeline" / "sql_features.yaml"


# legacy: unchanged prompts (the only supported layout). The other two were MEASURED NOT VIABLE and exist only to
# reproduce the measurement (eval/results/prefix_cache_*, router_layout_comparison.jsonl): prefixing the 6.6k-token SQL
# block makes the composer answer with SQL instead of prose (cache_friendly) and weakens router refusals
# (cache_friendly_router). Never enable them in production.
PROMPT_LAYOUTS = ("legacy", "cache_friendly", "cache_friendly_router")


@dataclass(frozen=True)
class SQLFeatures:
    glossary: bool = True
    value_lookup: bool = True
    critic: bool = True
    retries: bool = True
    max_retries: int = 3
    prompt_layout: str = "legacy"          # see PROMPT_LAYOUTS
    max_completion_tokens: int = 700      # cap on the generator's completion

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
    conv = {"max_retries": int, "max_completion_tokens": int, "prompt_layout": str}
    feats = SQLFeatures(**{k: conv.get(k, bool)(v) for k, v in raw.items()})
    if feats.prompt_layout not in PROMPT_LAYOUTS:
        raise ValueError(f"prompt_layout must be one of {PROMPT_LAYOUTS}, got {feats.prompt_layout!r}")
    return feats


def load_features(path: str | Path = DEFAULT_PATH) -> SQLFeatures:
    """Read the YAML; a missing file means defaults (everything on)."""
    p = Path(path)
    if not p.exists():
        return SQLFeatures()
    return features_from_dict(yaml.safe_load(p.read_text(encoding="utf-8")))
