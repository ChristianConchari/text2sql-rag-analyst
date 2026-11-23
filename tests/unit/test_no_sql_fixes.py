"""Tests for general NO_SQL fixes: ValueIndex tie-breaking and column-cap behavior."""
import duckdb

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
