"""Tests for general NO_SQL fixes: schema context coverage, assumption rule, and IDF tie-breaking."""
from pathlib import Path

import duckdb

from src.rag.prompts.sql_prompt import SQL_RULES, build_schema_context
from src.rag.retrievers.value_index import ValueIndex, ValueLookupConfig

ROOT = Path(__file__).resolve().parents[2]


def test_schema_context_includes_cobertura():
    ctx = build_schema_context(ROOT / "domains/chilecompra/glossary.yaml")
    assert "cobertura:" in ctx


def test_sql_rules_prefer_assumption_over_declining():
    assert "NO declines" in SQL_RULES
    assert "NO_SQL:" in SQL_RULES


def test_value_index_small_column_not_starved_by_large_column(tmp_path):
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE oc_lineas (sector VARCHAR, RubroN3 VARCHAR)")
    rows = [("Salud", f"Rubro {i}") for i in range(60)] + [("Salud", "Servicios de Salud")]
    rows += [("Otros", "Otro")]
    con.executemany("INSERT INTO oc_lineas VALUES (?, ?)", rows)
    con.close()
    cfg = ValueLookupConfig(table="oc_lineas", columns=("sector", "RubroN3"), max_values_per_word=30)
    idx = ValueIndex.from_duckdb(db, cfg)
    hits = idx.lookup("proveedores del sector Salud")
    assert any(h.column == "sector" and h.value == "Salud" for h in hits)
