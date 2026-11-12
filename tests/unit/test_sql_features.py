"""Flags of the SQL route: helpers switch off, safety never does."""
from pathlib import Path

import duckdb
import pytest

from src.agents.orchestration.sql_loop import run_sql_loop
from src.policies.sql_verifier import load_policy, verify
from src.rag.prompts.sql_prompt import build_plain_schema_context
from src.runtime.features import SQLFeatures, features_from_dict, load_features
from src.runtime.pipelines.text2sql import Text2SQLPipeline
from src.runtime.providers.llm_client import FakeLLM, MeteredLLM
from src.toolkits.db.executor import run_query

ROOT = Path(__file__).resolve().parents[2]
POLICY = load_policy(ROOT / "configs/domain/chilecompra.yaml")
GRAIN_SQL = "SELECT SUM(MontoTotalOC) AS t FROM oc_lineas"
DROP = "DROP TABLE oc_lineas"


def block(s): return f"```sql\n{s}\n```"


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.duckdb"
    con = duckdb.connect(str(p))
    con.execute("create table oc_lineas(Codigo varchar, MontoTotalOC double, NombreProveedor varchar, "
                "es_persona_natural boolean, RutSucursal varchar)")
    con.execute("insert into oc_lineas values ('A',10,'Empresa SA',false,'76-1'),('A',10,'Empresa SA',false,'76-1')")
    con.close()
    return p


def test_default_config_is_everything_on():
    f = load_features()
    assert (f.glossary, f.value_lookup, f.critic, f.retries, f.max_retries) == (True, True, True, True, 3)
    assert SQLFeatures() == f and f.effective_retries == 3


def test_retries_off_means_zero_retries():
    assert features_from_dict({"retries": False}).effective_retries == 0


def test_unknown_flag_is_an_error():
    with pytest.raises(ValueError):
        features_from_dict({"critc": False})


# ---------- critic ----------
def test_critic_on_rejects_grain_violation():
    v = verify(GRAIN_SQL, POLICY)
    assert not v.ok and any("order_level_aggregate" in x for x in v.violations) and v.advisory == []


def test_critic_off_accepts_grain_violation_but_reports_it_as_advisory():
    v = verify(GRAIN_SQL, POLICY, critic=False)
    assert v.ok and v.violations == [] and any("order_level_aggregate" in x for x in v.advisory)


@pytest.mark.parametrize("sql", [
    DROP, "SELECT RutSucursal FROM oc_lineas", "SELECT * FROM oc_lineas", "SELECT 1 FROM otra_tabla",
    "SELECT Codigo FROM oc_lineas -- c", "SELECT \"Descripcion/Obervaciones\" FROM oc_lineas",
])
def test_safety_rules_stay_on_when_critic_is_off(sql):
    assert not verify(sql, POLICY, critic=False).ok


def test_critic_off_still_enforces_limit_and_masking_flag():
    v = verify("SELECT NombreProveedor FROM oc_lineas", POLICY, critic=False)
    assert v.ok and v.sql.endswith("LIMIT 100") and v.masked_columns and v.flag_column


def test_loop_critic_off_executes_grain_query_and_records_advisory(db):
    r = run_sql_loop("q", FakeLLM([block(GRAIN_SQL)]), POLICY, lambda s: run_query(db, s), "sys", 0, critic=False)
    assert r.status == "ok" and r.attempts[0].advisory and r.rows == [(20.0,)]


def test_loop_critic_on_retries_grain_query(db):
    fixed = "WITH o AS (SELECT Codigo, any_value(MontoTotalOC) AS m FROM oc_lineas GROUP BY Codigo) SELECT SUM(m) FROM o"
    r = run_sql_loop("q", FakeLLM([block(GRAIN_SQL), block(fixed)]), POLICY, lambda s: run_query(db, s), "sys", 3)
    assert r.status == "ok" and len(r.attempts) == 2 and r.rows == [(10.0,)] and r.attempts[1].advisory == []


# ---------- retries ----------
def test_retries_off_is_a_single_attempt(db):
    llm = FakeLLM([block(DROP), block("SELECT Codigo FROM oc_lineas")])
    r = run_sql_loop("q", llm, POLICY, lambda s: run_query(db, s), "sys", 0)
    assert r.status == "failed" and len(llm.calls) == 1


def test_safety_violation_without_retries_fails_and_never_executes():
    executed = []
    r = run_sql_loop("q", FakeLLM([block(DROP)]), POLICY, lambda s: executed.append(s), "sys", 0, critic=False)
    assert r.status == "failed" and executed == []


# ---------- glossary ----------
def test_plain_schema_has_no_glossary_and_hides_blocked_columns(db):
    ctx = build_plain_schema_context(db, "oc_lineas", POLICY.blocked_columns)
    assert "MontoTotalOC DOUBLE" in ctx and "RutSucursal" not in ctx
    assert "Reglas de negocio" not in ctx and "usar:" not in ctx


# ---------- pipeline ----------
class SpyIndex:
    def __init__(self): self.calls = 0
    def lookup(self, q): self.calls += 1; return []


def make(db, llm, **flags):
    return Text2SQLPipeline(llm, POLICY, db, "sys", 3, value_index=SpyIndex(), features=features_from_dict(flags))


def test_pipeline_value_lookup_flag(db):
    on, off = make(db, FakeLLM([block("SELECT Codigo FROM oc_lineas")])), make(
        db, FakeLLM([block("SELECT Codigo FROM oc_lineas")]), value_lookup=False)
    on.generate_sql("q"); off.generate_sql("q")
    assert on.value_index.calls == 1 and off.value_index.calls == 0


def test_pipeline_retries_flag(db):
    llm = FakeLLM([block(DROP), block("SELECT Codigo FROM oc_lineas")])
    assert make(db, llm, retries=False).generate_sql("q").status == "failed" and len(llm.calls) == 1
    llm = FakeLLM([block(DROP), block("SELECT Codigo FROM oc_lineas")])
    assert make(db, llm).generate_sql("q").status == "ok" and len(llm.calls) == 2


def test_pipeline_critic_flag(db):
    assert make(db, FakeLLM([block(GRAIN_SQL)] * 4)).generate_sql("q").status == "failed"
    assert make(db, FakeLLM([block(GRAIN_SQL)]), critic=False, retries=False).generate_sql("q").status == "ok"


# ---------- metering ----------
def test_metered_llm_records_usage_or_none():
    class Inner(FakeLLM):
        last_usage = {"prompt_tokens": 12, "completion_tokens": 3}
    m = MeteredLLM(Inner(["a"]))
    assert m.chat([{"role": "user", "content": "x"}]) == "a"
    assert (m.calls[0].prompt_tokens, m.calls[0].completion_tokens) == (12, 3) and m.calls[0].latency_s >= 0
    m2 = MeteredLLM(FakeLLM(["b"]))
    m2.chat([{"role": "user", "content": "x"}])
    assert m2.calls[0].prompt_tokens is None
