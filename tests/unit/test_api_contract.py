"""The /api contract (schema, enums) and the per-request agent trace, through the HTTP layer with a FakeLLM."""
import json
from pathlib import Path
from typing import get_args

import duckdb
import pytest
from fastapi.testclient import TestClient

from src.interfaces.fastapi_app.app import create_app
from src.interfaces.fastapi_app.schemas import AskResponse, Outcome, Route, Status
from src.policies.output_guardrail import MASK
from src.policies.sql_verifier import load_policy
from src.rag.prompts.sql_prompt import build_schema_context, build_sql_system
from src.rag.retrievers.corpus import load_chunks
from src.rag.retrievers.dense import DenseRetriever
from src.rag.retrievers.embedders import HashEmbedder
from src.rag.retrievers.embedding_cache import EmbeddingCache
from src.runtime.pipelines.text2sql import Text2SQLPipeline
from src.runtime.providers.llm_client import FakeLLM
from src.runtime.trace import describe_sql, describe_violations

ROOT = Path(__file__).resolve().parents[2]
POLICY = load_policy(ROOT / "configs/domain/chilecompra.yaml")
SECRET_NAME = "Juan Perez"


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.duckdb"
    con = duckdb.connect(str(p))
    con.execute("create table oc_lineas(Codigo varchar, NombreProveedor varchar, es_persona_natural boolean, "
                "RutSucursal varchar, totalLineaNeto double)")
    con.execute(f"insert into oc_lineas values ('A','{SECRET_NAME}',true,'1-9',10),('B','Empresa SA',false,'76-1',20),"
                "('A','Juan Perez',true,'1-9',5)")
    con.close()
    return p


@pytest.fixture()
def retriever(tmp_path):
    cache = EmbeddingCache(tmp_path, "hash-embedder")
    return DenseRetriever(load_chunks(ROOT / "tests/fixtures/rag_corpus.jsonl"), HashEmbedder(), cache, k=3, min_score=0.2)


def client(llm, db, retriever=None):
    ctx = build_schema_context(ROOT / "domains/chilecompra/glossary.yaml", POLICY.blocked_columns)
    return TestClient(create_app(Text2SQLPipeline(llm, POLICY, db, build_sql_system(ctx), 3, retriever=retriever)))


def ask(c, question="q"):
    r = c.post("/api/ask", json={"question": question})
    assert r.status_code == 200, r.text
    body = r.json()
    AskResponse.model_validate(body)
    return body


def sql_block(s): return f"```sql\n{s}\n```"


def steps(body): return [(s["agent"], s["outcome"]) for s in body["trace"]]


# ---------- schema ----------
def test_openapi_exposes_the_typed_contract(db):
    spec = client(FakeLLM([]), db).get("/openapi.json").json()
    assert set(spec["paths"]) == {"/api/ask", "/api/health"}
    schemas = spec["components"]["schemas"]
    ask_resp = schemas["AskResponse"]
    assert set(ask_resp["properties"]) == {"request_id", "route", "status", "answer", "sql", "columns", "rows",
                                           "warnings", "attempts", "citations", "trace", "latency_ms", "model"}
    assert ask_resp["properties"]["route"]["enum"] == list(get_args(Route))
    assert ask_resp["properties"]["status"]["enum"] == list(get_args(Status))
    assert schemas["TraceStep"]["properties"]["outcome"]["enum"] == list(get_args(Outcome)) == [
        "ok", "objection", "error", "skipped"]
    assert set(ask_resp["required"]) == set(ask_resp["properties"])          # all always sent
    assert set(schemas["TraceStep"]["required"]) == set(schemas["TraceStep"]["properties"])
    assert set(schemas["Citation"]["properties"]) == {"label", "norma", "articulo", "excerpt"}


def test_route_and_status_literals_cover_the_pipeline():
    assert set(get_args(Route)) == {"sql", "docs", "out_of_scope", "refuse", "unknown"}
    assert set(get_args(Status)) == {"ok", "declined", "failed", "refused", "out_of_scope", "no_evidence",
                                     "not_implemented", "needs_clarification"}


def test_health_and_old_paths(db):
    c = client(FakeLLM([]), db)
    assert c.get("/api/health").json() == {"status": "ok"}
    assert c.post("/ask", json={"question": "x"}).status_code == 404
    assert c.post("/api/ask", json={"question": ""}).status_code == 422


# ---------- one trace per route ----------
def test_sql_trace_with_verifier_objection_and_retry(db):
    llm = FakeLLM(['{"route": "sql"}',
                   sql_block("SELECT RutSucursal FROM oc_lineas"),                       # blocked column -> objection
                   sql_block("SELECT COUNT(DISTINCT Codigo) AS n FROM oc_lineas"),
                   "Se enviaron 2 órdenes."])
    body = ask(client(llm, db))
    assert (body["route"], body["status"], body["attempts"], body["rows"]) == ("sql", "ok", 2, [[2]])
    assert steps(body) == [("Enrutador", "ok"),
                           ("Generador SQL", "ok"), ("Verificador SQL", "objection"),
                           ("Generador SQL", "ok"), ("Verificador SQL", "ok"),
                           ("Ejecutor", "ok"), ("Redactor", "ok")]
    gen = [s for s in body["trace"] if s["agent"] == "Generador SQL"]
    assert [s["attempt"] for s in gen] == [1, 2]
    assert gen[1]["detail"] == "Intento 2: COUNT(DISTINCT) sobre oc_lineas"
    assert body["trace"][2]["detail"] == "Objeta: usa una columna bloqueada"
    assert body["trace"][5]["detail"].startswith("1 fila")
    assert all(s["duration_ms"] >= 0 for s in body["trace"])
    assert body["model"] == "FakeLLM" and body["latency_ms"] >= 0 and len(body["request_id"]) == 36


def test_docs_trace(db, retriever):
    llm = FakeLLM(['{"route": "docs"}', "Diez días [Norma Ficticia 100 art. 2]."])
    body = ask(client(llm, db, retriever), "plazo para reclamar ante el Tribunal de Cuentas Imaginario")
    assert (body["route"], body["status"]) == ("docs", "ok")
    assert steps(body) == [("Enrutador", "ok"), ("Recuperador", "ok"), ("Redactor", "ok"),
                           ("Verificador de citas", "ok")]
    assert body["trace"][1]["detail"].startswith("k=3 · ")
    assert body["trace"][3]["detail"] == "Aprueba: art. 2 fue recuperado"
    assert body["citations"][0]["articulo"] == "2"


def test_docs_trace_with_invented_citation_retry(db, retriever):
    llm = FakeLLM(['{"route": "docs"}', "Diez días [Ley 19.886 art. 999].", "Diez días [Norma Ficticia 100 art. 2]."])
    body = ask(client(llm, db, retriever), "plazo para reclamar ante el Tribunal de Cuentas Imaginario")
    assert steps(body)[1:] == [("Recuperador", "ok"), ("Redactor", "ok"), ("Verificador de citas", "objection"),
                               ("Redactor", "ok"), ("Verificador de citas", "ok")]
    assert "999" not in json.dumps(body["trace"])      # the model's invented reference is not echoed


def test_out_of_scope_trace(db):
    body = ask(client(FakeLLM(['{"route": "out_of_scope"}']), db))
    assert (body["route"], body["status"]) == ("out_of_scope", "out_of_scope")
    assert steps(body) == [("Enrutador", "ok"), ("Respuesta fija", "ok")]


def test_refuse_trace(db):
    body = ask(client(FakeLLM(['{"route": "refuse"}']), db))
    assert (body["route"], body["status"]) == ("refuse", "refused")
    assert steps(body) == [("Enrutador", "ok"), ("Respuesta fija", "ok")]
    assert body["sql"] is None and body["rows"] == [] and body["citations"] == []


def test_unknown_route_trace(db):
    body = ask(client(FakeLLM(["??"]), db))
    assert (body["route"], body["status"]) == ("unknown", "needs_clarification")
    assert steps(body) == [("Enrutador", "error"), ("Respuesta fija", "ok")]


# ---------- no personal data in the trace ----------
def test_masked_name_never_reaches_trace_or_warnings(db):
    llm = FakeLLM(['{"route": "sql"}',
                   # DuckDB's conversion error quotes the value: "Could not convert string 'Juan Perez' ..."
                   sql_block("SELECT CAST(NombreProveedor AS INTEGER) AS n FROM oc_lineas"),
                   sql_block("SELECT NombreProveedor, totalLineaNeto FROM oc_lineas WHERE NombreProveedor = 'Juan Perez'"),
                   "Respuesta."])
    body = ask(client(llm, db))
    assert SECRET_NAME in str(llm.calls[2])           # the generator did see the DB error (feedback) ...
    assert body["status"] == "ok" and all(r[0] == MASK for r in body["rows"])
    assert ("Ejecutor", "error") in steps(body) and ("Guardia de salida", "ok") in steps(body)
    leaked = json.dumps({"trace": body["trace"], "warnings": body["warnings"]}, ensure_ascii=False)
    assert "Juan" not in leaked and "Perez" not in leaked   # ... but neither the trace nor the warnings carry it
    guard = next(s for s in body["trace"] if s["agent"] == "Guardia de salida")
    assert guard["detail"] == "2 valores enmascarados (personas naturales)"


def test_describe_sql_keeps_no_literals():
    sql = "SELECT SUM(totalLineaNeto) FROM oc_lineas WHERE NombreProveedor = 'Juan Perez' AND x IN (SELECT 1 FROM t)"
    assert describe_sql(sql, POLICY.allowed_tables) == "SUM sobre oc_lineas"
    assert describe_sql("SELEC 'Juan Perez' FRM", POLICY.allowed_tables).startswith("consulta SQL")
    assert describe_sql(None, POLICY.allowed_tables) == "respuesta sin SQL"


def test_describe_violations_uses_codes_only():
    v = ["parse_error: Invalid expression near 'Juan Perez'", "order_level_aggregate_over_lines: SUM(x) agrega ...",
         "weird text: Juan"]
    text = describe_violations(v)
    assert "Juan" not in text and "SQL no válido" in text and "deduplicar por orden" in text
