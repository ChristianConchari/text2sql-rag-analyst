"""Tests for general NO_SQL fixes: ValueIndex tie-breaking, column-cap, and schema context rendering."""
import duckdb
import yaml

from src.rag.prompts.sql_prompt import build_schema_context
from src.rag.retrievers.value_index import ValueIndex, ValueLookupConfig


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


def test_value_index_frequent_value_not_evicted_by_spurious_idf_hits(tmp_path):
    """A high-frequency value matching a specific token must not be evicted by rarer 1-line hits."""
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE oc_lineas (NombreProveedor VARCHAR)")
    rows = [("COPEC S.A.",) for _ in range(500)]
    rows += [(f"ESTACION COPEC {i}",) for i in range(6)]
    rows += [(f"PROVEEDOR DIRECTO {i}",) for i in range(5)]
    rows += [(f"OTRO {i}",) for i in range(30)]
    con.executemany("INSERT INTO oc_lineas VALUES (?)", rows)
    con.close()
    cfg = ValueLookupConfig(
        table="oc_lineas",
        columns=("NombreProveedor",),
        max_values_per_word=10,
        max_share_per_word=0.5,
        max_per_column=5,
    )
    idx = ValueIndex.from_duckdb(db, cfg)
    hits = idx.lookup("trato directo con Copec")
    assert hits, "Expected value hits for Copec"
    assert hits[0].value == "COPEC S.A."


def test_value_index_idf_breaks_ties_when_score_and_lines_equal(tmp_path):
    """When match score and line count are equal, the rarer (higher-IDF) token wins."""
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE oc_lineas (NombreProveedor VARCHAR)")
    rows = [("PROVEEDOR RARO",)] * 10
    rows += [("PROVEEDOR COMUN A",)] * 10 + [("PROVEEDOR COMUN B",)] * 10 + [("PROVEEDOR COMUN C",)] * 10
    rows += [(f"OTRO {i}",) for i in range(20)]
    con.executemany("INSERT INTO oc_lineas VALUES (?)", rows)
    con.close()
    cfg = ValueLookupConfig(
        table="oc_lineas",
        columns=("NombreProveedor",),
        max_values_per_word=10,
        max_share_per_word=0.5,
        max_per_column=5,
    )
    idx = ValueIndex.from_duckdb(db, cfg)
    hits = idx.lookup("raro comun")
    assert [h.value for h in hits[:1]] == ["PROVEEDOR RARO"]


def test_value_index_idf_breaks_ties_reverse_order(tmp_path):
    """Tie-breaker depends on IDF, not query token order or alphabetical order."""
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE oc_lineas (NombreProveedor VARCHAR)")
    rows = [("EMPRESA ZETA",)] * 10
    rows += [("EMPRESA ALFA 1",)] * 10 + [("EMPRESA ALFA 2",)] * 10 + [("EMPRESA ALFA 3",)] * 10
    rows += [(f"OTRO {i}",) for i in range(20)]
    con.executemany("INSERT INTO oc_lineas VALUES (?)", rows)
    con.close()
    cfg = ValueLookupConfig(
        table="oc_lineas",
        columns=("NombreProveedor",),
        max_values_per_word=10,
        max_share_per_word=0.5,
        max_per_column=5,
    )
    idx = ValueIndex.from_duckdb(db, cfg)
    hits = idx.lookup("alfa zeta")
    assert [h.value for h in hits[:1]] == ["EMPRESA ZETA"]


def test_build_schema_context_renders_reglas_generales(tmp_path):
    """build_schema_context renders tabla.reglas_generales before Columnas when present, and omits when empty."""
    g_with = {
        "tabla": {
            "nombre": "oc_lineas",
            "grano": "linea de orden",
            "reglas_generales": ["Regla general uno", "Regla general dos"],
        },
        "columnas": {"Codigo": {"desc": "Codigo de orden", "cobertura": "100%"}},
        "reglas_de_negocio": [],
    }
    p_with = tmp_path / "with_rules.yaml"
    p_with.write_text(yaml.safe_dump(g_with), encoding="utf-8")
    ctx_with = build_schema_context(p_with)
    assert "Reglas generales:" in ctx_with
    assert "- Regla general uno" in ctx_with
    assert "- Regla general dos" in ctx_with
    assert ctx_with.index("Reglas generales:") < ctx_with.index("Columnas:")

    g_without = {
        "tabla": {"nombre": "oc_lineas", "grano": "linea de orden"},
        "columnas": {"Codigo": {"desc": "Codigo de orden"}},
        "reglas_de_negocio": [],
    }
    p_without = tmp_path / "without_rules.yaml"
    p_without.write_text(yaml.safe_dump(g_without), encoding="utf-8")
    ctx_without = build_schema_context(p_without)
    assert "Reglas generales:" not in ctx_without
