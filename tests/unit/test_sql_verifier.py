from pathlib import Path

import pytest
import yaml

from src.policies.sql_verifier import Policy, load_policy, verify

ROOT = Path(__file__).resolve().parents[2]
POLICY = load_policy(ROOT / "configs/domain/chilecompra.yaml")
CASES = yaml.safe_load((ROOT / "eval/attack_cases.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("c", CASES, ids=[c["id"] for c in CASES])
def test_attack_cases(c):
    r = verify(c["sql"], POLICY)
    if c["expect"] == "allow":
        assert r.ok, r.violations
        if "sql_out" in c:
            assert r.sql == c["sql_out"]
        if "warn" in c:
            assert any(w.startswith(c["warn"]) for w in r.warnings)
    else:
        assert not r.ok and r.sql is None
        assert any(v.startswith(c["code"]) for v in r.violations), r.violations


def test_policy_comes_from_config():
    assert POLICY.allowed_tables == {"oc_lineas"}
    assert (POLICY.default_limit, POLICY.max_limit) == (100, 1000)


def test_dialect_is_a_parameter():
    r = verify("SELECT a FROM oc_lineas", POLICY, dialect="postgres")
    assert r.ok and r.sql.endswith("LIMIT 100")


def test_custom_policy_limits():
    p = Policy(allowed_tables=frozenset({"x"}), default_limit=5, max_limit=10)
    assert verify("SELECT a FROM x", p).sql == "SELECT a FROM x LIMIT 5"
    assert verify("SELECT a FROM x LIMIT 50", p).sql == "SELECT a FROM x LIMIT 10"


@pytest.mark.parametrize("sql", ["", "   ", ";", "SELEC FROM", "SELECT (", None])
def test_garbage_rejected(sql):
    assert not verify(sql, POLICY).ok


# ---- masking: es_persona_natural must travel with NombreProveedor ----
def test_flag_added_when_name_selected():
    r = verify("SELECT NombreProveedor, totalLineaNeto FROM oc_lineas", POLICY)
    assert r.ok and r.masked_columns == ["NombreProveedor"] and r.flag_column == "es_persona_natural"
    assert r.sql == "SELECT NombreProveedor, totalLineaNeto, es_persona_natural FROM oc_lineas LIMIT 100"


def test_flag_added_to_group_by():
    r = verify("SELECT NombreProveedor, SUM(totalLineaNeto) AS t FROM oc_lineas GROUP BY NombreProveedor", POLICY)
    assert r.ok
    assert r.sql == ("SELECT NombreProveedor, SUM(totalLineaNeto) AS t, es_persona_natural FROM oc_lineas "
                     "GROUP BY NombreProveedor, es_persona_natural LIMIT 100")


def test_alias_is_tracked():
    r = verify("SELECT NombreProveedor AS proveedor FROM oc_lineas", POLICY)
    assert r.masked_columns == ["proveedor"] and "es_persona_natural" in r.sql


def test_flag_not_duplicated():
    r = verify("SELECT NombreProveedor, es_persona_natural FROM oc_lineas", POLICY)
    assert r.sql.count("es_persona_natural") == 1


def test_count_distinct_name_needs_no_mask():
    r = verify("SELECT COUNT(DISTINCT NombreProveedor) FROM oc_lineas", POLICY)
    assert r.ok and r.masked_columns == [] and "es_persona_natural" not in r.sql


def test_expression_over_name_is_masked_without_flag_rewrite():
    r = verify("SELECT UPPER(NombreProveedor) AS n FROM oc_lineas", POLICY)
    assert r.ok and r.masked_columns == ["n"] and "es_persona_natural" not in r.sql
    assert any(x.startswith("mask_without_flag") for x in r.warnings)


def test_cte_source_masked_without_flag_rewrite():
    r = verify("WITH t AS (SELECT NombreProveedor FROM oc_lineas) SELECT NombreProveedor FROM t", POLICY)
    assert r.ok and r.masked_columns == ["NombreProveedor"] and "es_persona_natural" not in r.sql


def test_union_flag_added_to_every_branch():
    r = verify("SELECT NombreProveedor FROM oc_lineas UNION ALL SELECT NombreProveedor FROM oc_lineas", POLICY)
    assert r.ok and r.sql.count("es_persona_natural") == 2


def test_no_masking_when_policy_has_none():
    p = Policy(allowed_tables=frozenset({"x"}))
    assert verify("SELECT NombreProveedor FROM x", p).masked_columns == []
