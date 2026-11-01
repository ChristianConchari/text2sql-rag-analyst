from pathlib import Path

import duckdb

from src.policies.output_guardrail import MASK, mask_rows
from src.policies.sql_verifier import load_policy, verify

COLS = ["NombreProveedor", "es_persona_natural", "total"]


def test_masks_true_and_null_keeps_false():
    rows = [("Juan Perez", True, 1), ("Empresa SA", False, 2), ("Sin Rut", None, 3)]
    assert mask_rows(COLS, rows) == [(MASK, True, 1), ("Empresa SA", False, 2), (MASK, None, 3)]


def test_missing_flag_column_masks_everything():
    assert mask_rows(["NombreProveedor", "total"], [("Empresa SA", 5)]) == [(MASK, 5)]


def test_no_name_column_untouched():
    rows = [("a", 1), ("b", 2)]
    assert mask_rows(["x", "y"], rows) == rows


def test_column_name_case_insensitive_and_flag_order():
    assert mask_rows(["es_persona_natural", "nombreproveedor"], [(True, "Ana")]) == [(True, MASK)]


def test_only_literal_false_unmasks():
    # 0 / "false" / "" must not unmask: fail closed
    assert mask_rows(COLS, [("X", 0, 1)])[0][0] == MASK
    assert mask_rows(COLS, [("X", "false", 1)])[0][0] == MASK


def test_aliased_column_from_verifier_is_masked():
    assert mask_rows(["proveedor", "es_persona_natural"], [("Ana", True)], ["proveedor"]) == [(MASK, True)]
    assert mask_rows(["proveedor", "es_persona_natural"], [("SA", False)], ["proveedor"]) == [("SA", False)]


def test_empty_masked_list_means_nothing_to_mask():
    assert mask_rows(["NombreProveedor"], [("Ana",)], []) == [("Ana",)]


def test_end_to_end_verify_then_mask():
    pol = load_policy(Path(__file__).resolve().parents[2] / "configs/domain/chilecompra.yaml")
    v = verify("SELECT NombreProveedor AS prov, SUM(totalLineaNeto) AS t FROM oc_lineas GROUP BY NombreProveedor", pol)
    con = duckdb.connect()
    con.execute("create table oc_lineas(NombreProveedor varchar, totalLineaNeto double, es_persona_natural boolean)")
    con.execute("insert into oc_lineas values ('Juan Perez',10,true),('Empresa SA',20,false),('Sin Rut',5,null)")
    cur = con.execute(v.sql)
    columns = [d[0] for d in cur.description]
    rows = mask_rows(columns, cur.fetchall(), v.masked_columns, v.flag_column)
    # grouped by (name, flag): the natural person and the unknown-RUT supplier stay separate rows, both masked
    assert sorted(rows, key=lambda r: r[1]) == [(MASK, 5.0, None), (MASK, 10.0, True), ("Empresa SA", 20.0, False)]
