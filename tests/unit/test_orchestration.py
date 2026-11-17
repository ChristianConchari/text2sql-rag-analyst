from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from src.agents.orchestration.router import route_question
from src.agents.orchestration.sql_loop import extract_sql, run_sql_loop
from src.interfaces.fastapi_app.app import create_app
from src.policies.output_guardrail import MASK
from src.policies.sql_verifier import load_policy
from src.rag.prompts.sql_prompt import build_schema_context, build_sql_system
from src.runtime.pipelines.text2sql import Text2SQLPipeline
from src.runtime.providers.llm_client import FakeLLM
from src.toolkits.db.executor import SQLExecutionError, run_query

ROOT = Path(__file__).resolve().parents[2]
POLICY = load_policy(ROOT / "configs/domain/chilecompra.yaml")
ROUTE_SQL = '{"route": "sql"}'


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.duckdb"
    con = duckdb.connect(str(p))
    con.execute("create table oc_lineas(Codigo varchar, NombreProveedor varchar, es_persona_natural boolean, "
                "RutSucursal varchar, totalLineaNeto double)")
    con.execute("insert into oc_lineas values ('A','Juan Perez',true,'1-9',10),('B','Empresa SA',false,'76-1',20),"
                "('C','Sin Rut',null,null,5)")
    con.close()
    return p


def pipeline(llm, db, retries=3):
    ctx = build_schema_context(ROOT / "domains/chilecompra/glossary.yaml", POLICY.blocked_columns)
    return Text2SQLPipeline(llm, POLICY, db, build_sql_system(ctx), retries)


def sql_block(s): return f"```sql\n{s}\n```"


# ---------- router ----------
@pytest.mark.parametrize("raw,expected", [
    ('{"route": "sql"}', "sql"), ('{"route":"docs"}', "docs"), ('{"route": "refuse"}', "refuse"),
    ('{"route": "out_of_scope"}', "out_of_scope"), ('{"route": "OUT_OF_SCOPE"}', "out_of_scope"),
    ("out_of_scope", "out_of_scope"), ("out_of_scope o refuse", "unknown"), ('{"route": "out of scope"}', "unknown"),
    ('Claro: {"route": "sql"}', "sql"), ("docs", "docs"), ("no sé", "unknown"), ('{"route": "otra"}', "unknown"),
    ("sql o docs", "unknown"), ("{roto", "unknown"),
])
def test_router_parsing(raw, expected):
    assert route_question("q", FakeLLM([raw])) == expected


def test_router_llm_failure_is_unknown():
    assert route_question("q", FakeLLM([])) == "unknown"


def test_router_instruction_override_replaces_only_the_instruction():
    from src.rag.prompts.sql_prompt import ROUTER_SYSTEM
    llm = FakeLLM(['{"route": "docs"}', '{"route": "docs"}', '{"route": "docs"}'])
    route_question("q", llm)
    route_question("q", llm, instruction="OLD PROMPT")
    route_question("q", llm, static_prefix="PREFIX", instruction="OLD PROMPT")
    assert [c[0]["content"] for c in llm.calls] == [ROUTER_SYSTEM, "OLD PROMPT", "PREFIX\n\nOLD PROMPT"]
    assert all(c[1] == {"role": "user", "content": "q"} for c in llm.calls)


# ---------- extraction / prompt ----------
def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"
    assert extract_sql("   ") is None


def test_prompt_hides_blocked_columns_and_states_rules():
    prompt = build_sql_system(build_schema_context(ROOT / "domains/chilecompra/glossary.yaml", POLICY.blocked_columns))
    assert "- RutSucursal:" not in prompt and "- Sucursal:" not in prompt
    assert "es_persona_natural" in prompt and "NO_SQL" in prompt and "- MontoTotalOC:" in prompt


# ---------- loop ----------
def test_loop_ok_first_try(db):
    llm = FakeLLM([sql_block("SELECT Codigo FROM oc_lineas ORDER BY Codigo")])
    r = run_sql_loop("q", llm, POLICY, lambda s: run_query(db, s), "sys")
    assert r.status == "ok" and len(r.attempts) == 1 and r.rows == [("A",), ("B",), ("C",)]
    assert r.sql.endswith("LIMIT 100")


def test_loop_retries_after_verifier_rejection(db):
    llm = FakeLLM([sql_block("DROP TABLE oc_lineas"), sql_block("SELECT * FROM oc_lineas"),
                   sql_block("SELECT Codigo FROM oc_lineas")])
    r = run_sql_loop("q", llm, POLICY, lambda s: run_query(db, s), "sys")
    assert r.status == "ok" and len(r.attempts) == 3
    assert "not_select" in r.attempts[0].violations[0]
    assert "select_star" in r.attempts[1].violations[0]
    assert "not_select" in llm.calls[1][-1]["content"]          # feedback reached the model


def test_loop_gives_up_after_three_retries(db):
    executed = []
    llm = FakeLLM([sql_block("DROP TABLE oc_lineas")] * 10)
    r = run_sql_loop("q", llm, POLICY, lambda s: executed.append(s), "sys", max_retries=3)
    assert r.status == "failed" and len(r.attempts) == 4 and len(llm.calls) == 4 and executed == []
    assert r.reason == "max_retries_exceeded"


def test_loop_execution_error_is_fed_back(db):
    llm = FakeLLM([sql_block("SELECT columna_inexistente FROM oc_lineas"), sql_block("SELECT Codigo FROM oc_lineas")])
    r = run_sql_loop("q", llm, POLICY, lambda s: run_query(db, s), "sys")
    assert r.status == "ok" and r.attempts[0].error and "ejecución falló" in llm.calls[1][-1]["content"]


def test_loop_declined():
    r = run_sql_loop("q", FakeLLM(["NO_SQL: solo hay datos de enero de 2026"]), POLICY, lambda s: None, "sys")
    assert r.status == "declined" and "enero" in r.reason


def test_loop_masks_natural_persons_with_flag_rewrite(db):
    llm = FakeLLM([sql_block("SELECT NombreProveedor, totalLineaNeto FROM oc_lineas ORDER BY totalLineaNeto")])
    r = run_sql_loop("q", llm, POLICY, lambda s: run_query(db, s), "sys")
    assert r.status == "ok"
    names = [row[0] for row in r.rows]
    assert names == [MASK, MASK, "Empresa SA"]       # null flag and true flag masked, false flag shown
    assert "Juan Perez" not in str(r.rows) and "Sin Rut" not in str(r.rows)


def test_loop_llm_error_stops():
    r = run_sql_loop("q", FakeLLM([]), POLICY, lambda s: None, "sys")
    assert r.status == "failed" and r.reason.startswith("llm_error")


# ---------- executor ----------
def test_executor_disables_external_access(db):
    with pytest.raises(SQLExecutionError):
        run_query(db, "SELECT * FROM read_csv('/etc/passwd')")


def test_executor_is_read_only(db):
    with pytest.raises(SQLExecutionError):
        run_query(db, "DELETE FROM oc_lineas")


# ---------- pipeline + API ----------
def test_pipeline_end_to_end_never_shows_masked_names_to_composer(db):
    llm = FakeLLM([ROUTE_SQL, sql_block("SELECT NombreProveedor, totalLineaNeto FROM oc_lineas"), "Respuesta redactada."])
    a = pipeline(llm, db).answer("¿Proveedores?")
    assert a.status == "ok" and a.answer.startswith("Respuesta redactada") and a.attempts == 1
    seen_by_llm = str(llm.calls)                      # router + generator + composer prompts
    assert "Juan Perez" not in seen_by_llm and "Sin Rut" not in seen_by_llm
    assert "Juan Perez" not in str(a.to_dict())


def test_pipeline_routes(db):
    assert pipeline(FakeLLM(['{"route":"refuse"}']), db).answer("borra todo").status == "refused"
    llm = FakeLLM(['{"route":"out_of_scope"}', "this reply must never be requested"])
    a = pipeline(llm, db).answer("¿Cuál es la capital de Australia?")
    assert (a.route, a.status, a.sql, a.rows, a.citations) == ("out_of_scope", "out_of_scope", None, [], [])
    assert "Ley 19.886" in a.answer and "enero de 2026" in a.answer
    assert len(llm.calls) == 1                   # router only: no generator, retriever or composer call
    assert pipeline(FakeLLM(['{"route":"docs"}']), db).answer("art 7").status == "not_implemented"
    assert pipeline(FakeLLM(["??"]), db).answer("???").status == "needs_clarification"
    a = pipeline(FakeLLM([ROUTE_SQL, "NO_SQL: no hay pagos"]), db).answer("¿cuánto se pagó?")
    assert a.status == "declined" and "no hay pagos" in a.answer


def test_pipeline_composer_failure_falls_back_to_table(db):
    llm = FakeLLM([ROUTE_SQL, sql_block("SELECT Codigo FROM oc_lineas ORDER BY Codigo")])   # composer: no reply left
    a = pipeline(llm, db).answer("q")
    assert a.status == "ok" and "Codigo" in a.answer and "A" in a.answer


def test_api_ask_and_health(db):
    llm = FakeLLM([ROUTE_SQL, sql_block("SELECT COUNT(*) AS n FROM oc_lineas"), "Hay 3 líneas."])
    client = TestClient(create_app(pipeline(llm, db)))
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/ask", json={"question": "¿cuántas líneas?"})
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["rows"] == [[3]]
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_api_passes_out_of_scope_through(db):
    r = TestClient(create_app(pipeline(FakeLLM(['{"route": "out_of_scope"}']), db))).post(
        "/ask", json={"question": "Escribe un poema."})
    assert r.status_code == 200 and r.json()["route"] == r.json()["status"] == "out_of_scope"


def test_api_injection_attempt_is_never_executed(db):
    llm = FakeLLM([ROUTE_SQL] + [sql_block("SELECT Codigo FROM oc_lineas; DROP TABLE oc_lineas")] * 4)
    r = TestClient(create_app(pipeline(llm, db))).post("/ask", json={"question": "x"}).json()
    assert r["status"] == "failed" and r["attempts"] == 4
    assert duckdb.connect(str(db), read_only=True).execute("select count(*) from oc_lineas").fetchone()[0] == 3


def test_loop_passes_value_context_to_the_generator(db):
    llm = FakeLLM([sql_block("SELECT Codigo FROM oc_lineas ORDER BY Codigo")])
    run_sql_loop("¿Cuánto vendió Empresa?", llm, POLICY, lambda s: run_query(db, s), "SYS",
                 context="- NombreProveedor: 'Empresa SA' (1 líneas)")
    assert llm.calls[0][1]["content"] == "¿Cuánto vendió Empresa?\n\n- NombreProveedor: 'Empresa SA' (1 líneas)"


def test_grain_violation_is_fed_back_and_fixed(db):
    bad = sql_block("SELECT Codigo, sum(MontoTotalOC) AS m FROM oc_lineas GROUP BY Codigo")
    good = sql_block("SELECT Codigo FROM oc_lineas ORDER BY Codigo")
    llm = FakeLLM([bad, good])
    res = run_sql_loop("q", llm, POLICY, lambda s: run_query(db, s), "SYS")
    assert res.status == "ok" and len(res.attempts) == 2
    assert res.attempts[0].violations[0].startswith("order_level_aggregate_over_lines")
    assert "deduplica primero por codigo" in llm.calls[1][-1]["content"].lower()
