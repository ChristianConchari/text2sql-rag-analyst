"""The metrics of src/eval/consigna.py on hand-written records, and the runner end to end with a FakeLLM."""
import json
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from scripts import eval_consigna as runner
from src.eval import consigna as m
from src.policies.sql_verifier import load_policy
from src.rag.prompts.sql_prompt import build_schema_context, build_sql_system
from src.rag.retrievers.dense import RetrievalConfig, build_dense_retriever
from src.rag.retrievers.embedders import HashEmbedder
from src.runtime.pipelines.text2sql import Text2SQLPipeline
from src.runtime.providers.llm_client import FakeLLM

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests/fixtures/rag_corpus.jsonl"
SQL_V, DOC_V = "Verificador SQL", "Verificador de citas"


def step(agent, outcome="ok", attempt=None, detail="d", ms=1.0):
    return {"agent": agent, "outcome": outcome, "attempt": attempt, "detail": detail, "duration_ms": ms}


def call(role="router", sha1="a", prompt=10, completion=2, **kw):
    return {"role": role, "sha1": sha1, "prompt_tokens": prompt, "completion_tokens": completion,
            "prompt_eval_ms": 1.0, "latency_ms": 5.0, "error": False, "done_reason": "stop", **kw}


def rec(id="q", group="sql_answerable", expect=("sql",), route="sql", status="ok", **kw):
    base = {"id": id, "group": group, "expect": list(expect), "route": route, "status": status, "trace": [],
            "calls": [], "latency_s": 1.0, "strict": None, "lenient": None, "gold_hit": None, "retrieved": None,
            "cited": None, "gold": None, "gold_extra": None, "out_of_corpus": False, "similarity": None,
            "similarity_control": None}
    return {**base, **kw}


# ---------- pure functions ----------
def test_percentiles_interpolate_and_handle_empty():
    from scripts.eval_sql_ablation import pct
    xs = [4.0, 1.0, 3.0, 2.0, 10.0]
    assert m.percentiles(xs) == {"n": 5, "p50": pct(xs, .5), "p95": pct(xs, .95)}
    assert m.percentiles([None, 7.0]) == {"n": 1, "p50": 7.0, "p95": 7.0}
    assert m.percentiles([]) == {"n": 0, "p50": None, "p95": None}


def test_router_metrics_ignore_cases_without_expectation():
    out = m.router_metrics([
        rec("a"), rec("b", route="docs"), rec("c", expect=(), route="refuse"),
        rec("d", expect=("out_of_scope", "refuse"), route="refuse"),
        rec("e", expect=("out_of_scope", "refuse"), route="unknown")])
    assert (out["n"], out["correct"]) == (4, 2)
    assert out["matrix"]["sql"]["sql"] == 1 and out["matrix"]["sql"]["docs"] == 1
    assert out["matrix"]["refuse"]["refuse"] == 1 and out["matrix"]["out_of_scope"]["unknown"] == 1
    assert [x["id"] for x in out["misses"]] == ["b", "e"]
    assert len(out["matrix"]) == 5 and all(len(row) == 5 for row in out["matrix"].values())


def test_decision_table():
    out = m.decision_table([rec(), rec(status="failed"), rec(), rec(group="docs", status="no_evidence")])
    assert out == {"sql_answerable": {"ok": 2, "failed": 1}, "docs": {"no_evidence": 1}}


def test_handoff_incoherence_only_for_questions_the_tool_should_answer():
    out = m.handoff_incoherence([
        rec("a", status="declined"), rec("b"),
        rec("c", group="sql_unanswerable", expect=(), status="declined"),       # declining is right here
        rec("d", route="docs", status="declined"),                              # misrouted, not a sql handoff
        rec("e", "docs", ("docs",), "docs", "no_evidence"), rec("f", "docs", ("docs",), "docs", "declined"),
        rec("g", "docs", ("docs",), "docs", "failed"),
        rec("h", "docs_out_of_corpus", ("docs",), "docs", "no_evidence")])
    assert out == {"sql": {"n": 2, "ids": ["a"]}, "docs": {"n": 3, "ids": ["e", "f"]}}


def test_dialogue_repair_and_repeated_objection():
    repaired = rec("a", strict=True, trace=[step(SQL_V, "objection", 1, "x"), step(SQL_V, "ok", 2)])
    wrong = rec("b", strict=False, trace=[step(SQL_V, "objection", 1, "x"), step(SQL_V, "ok", 2)])
    stuck = rec("c", status="failed", strict=False,
                trace=[step(SQL_V, "objection", 1, "x"), step(SQL_V, "objection", 2, "x")])
    other = rec("d", status="failed",
                trace=[step(SQL_V, "objection", 1, "x"), step(SQL_V, "objection", 2, "y")])
    no_ref = rec("e", "router_cases", trace=[step(SQL_V, "objection", 1), step(SQL_V, "ok", 2)])
    clean = rec("f", trace=[step(SQL_V, "ok", 1)])
    docs = rec("g", "docs", ("docs",), "docs", gold_hit=True,
               trace=[step(DOC_V, "objection", 1), step(DOC_V, "objection", 3), step(DOC_V, "ok", 4)])
    out = m.dialogue_metrics([repaired, wrong, stuck, other, no_ref, clean, docs, rec("h", route="refuse")])
    sql = out[SQL_V]
    assert (sql["reviewed"], sql["with_objection"], sql["repaired"]) == (6, 5, 3)
    assert (sql["repaired_with_reference"], sql["repaired_correct"], sql["repeated_objection"]) == (2, 1, 1)
    assert (sql["approved"], sql["mean_rounds_to_approval"]) == (4, 1.75)
    # Same text in attempts 1 and 3 is not "two consecutive attempts".
    assert (out[DOC_V]["repeated_objection"], out[DOC_V]["repaired_correct"]) == (0, 1)
    assert m.dialogue_metrics([])[SQL_V]["mean_rounds_to_approval"] is None


def test_redundancy_counts_retries_duplicates_and_calls_per_route():
    a = rec("a", calls=[call(sha1="r"), call(sha1="s"), call(sha1="s"), call(sha1="s")],
            trace=[step("Generador SQL", attempt=1), step("Generador SQL", attempt=3), step("Redactor")])
    b = rec("b", route="docs", calls=[call(sha1="r"), call(sha1=None)], trace=[step("Redactor", attempt=1)])
    c = rec("c", route="refuse", calls=[call(sha1="r")], trace=[step("Enrutador"), step("Respuesta fija")])
    out = m.redundancy_metrics([a, b, c])
    assert out["retries"] == {"n": 2, "mean": 1.0, "max": 2, "with_retry": 1}
    assert out["identical_replies"] == {"calls": 7, "duplicated": 2, "questions": 1}
    assert out["calls_by_route"]["refuse"] == {"n": 1, "mean": 1.0, "min": 1, "max": 1}
    assert out["calls_by_route"]["sql"]["max"] == 4


def test_agent_latency_splits_the_two_writers():
    out = m.agent_latency([rec(trace=[step("Enrutador", ms=10), step("Redactor", ms=4)]),
                           rec(trace=[step("Enrutador", ms=20), step("Redactor", attempt=1, ms=8)])])
    assert out["Enrutador"] == {"n": 2, "p50": 15.0, "p95": 19.5}
    assert out["Redactor (SQL)"]["p50"] == 4 and out["Redactor (docs)"]["p50"] == 8


def test_retrieval_metrics_with_incomplete_gold():
    ranked = [["a", "x", "b", "y", "z"], ["x", "y"], ["a"]]
    strict = m.retrieval_metrics(ranked, [["a", "b"], ["y"], []])
    assert strict["n"] == 2                                          # no gold, not scored
    assert (strict["recall@1"], strict["precision@1"]) == (0.25, 0.5)
    assert strict["recall@3"] == 1.0 and strict["precision@3"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert strict["precision@5"] == pytest.approx((2 / 5 + 1 / 5) / 2)       # divides by k, not by the list length
    assert strict["mrr"] == 0.75 and strict["hit@1"] == 0.5
    # "x" is relevant but only listed in the extra gold: precision rises, recall can fall with a larger gold set
    wide = m.retrieval_metrics(ranked[:2], [["a", "b", "x", "w"], ["y"]])
    assert wide["precision@3"] > strict["precision@3"] and wide["recall@3"] < strict["recall@3"]
    assert m.retrieval_metrics([["a"]], [["b"]])["mrr"] == 0.0


def test_context_coverage():
    records = [
        rec("a", "docs", route="docs", retrieved=["g", "e", "n"], cited=["g"], gold=["g"], gold_extra=["e"]),
        rec("b", "docs", route="docs", status="no_evidence", retrieved=[], cited=[], gold=["g"], gold_extra=[]),
        rec("c", "docs", route="docs", status="declined", retrieved=["e"], cited=[], gold=["g"], gold_extra=["e"]),
        rec("d", "docs_out_of_corpus", retrieved=[], gold=[], out_of_corpus=True),
        rec("e", "docs_out_of_corpus", retrieved=["n"], gold=[], out_of_corpus=True),
        rec("f")]                                                    # not a retrieval record
    strict, wide = m.context_coverage(records, include_extra=False), m.context_coverage(records)
    assert (strict["in_corpus"], strict["in_corpus_with_gold"], wide["in_corpus_with_gold"]) == (3, 1, 2)
    assert (strict["gold_in_context"], wide["gold_in_context"], wide["retrieved"]) == (1, 3, 4)
    assert (wide["cited"], wide["retrieved_when_cited"]) == (1, 3)
    assert (wide["out_of_corpus"], wide["out_of_corpus_empty"]) == (2, 1)


def test_model_metrics_tokens_and_similarity():
    a = rec("a", calls=[call(prompt=10), call("sql_generator", prompt=30)], similarity=0.9, similarity_control=0.5)
    b = rec("b", calls=[call(prompt=None, completion=None)], similarity=0.4, similarity_control=0.6)
    out = m.model_metrics([a, b])
    assert out["all"]["prompt_tokens"] == 40        # the question with unknown usage is left out, not counted as 0
    assert out["by_role"]["router"]["calls"] == 2 and out["by_role"]["sql_generator"]["max_prompt_tokens"] == 30
    assert out["similarity"]["mean"] == pytest.approx(0.65) and out["similarity"]["above_control"] == 1


def test_decision_stability():
    runs = [[rec("a"), rec("b")], [rec("a"), rec("b", status="failed")]]
    assert m.decision_stability(runs) == {"runs": 2, "n": 2, "same_route": 2, "same_decision": 1, "changed": ["b"]}


# ---------- the runner, end to end ----------
@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.duckdb"
    con = duckdb.connect(str(p))
    con.execute("create table oc_lineas(Codigo varchar, NombreProveedor varchar, es_persona_natural boolean, "
                "RutSucursal varchar, totalLineaNeto double)")
    con.execute("insert into oc_lineas values ('A','Juan Perez',true,'1-9',10),('B','Empresa SA',false,'76-1',20),"
                "('A','Juan Perez',true,'1-9',5)")
    con.close()
    return p


def test_runner_end_to_end(db, tmp_path):
    cfg = RetrievalConfig(CORPUS, tmp_path / "cache", k=3, min_score=0.2)
    embedder = HashEmbedder()
    retriever = build_dense_retriever(cfg, embedder)
    ranker = build_dense_retriever(replace(cfg, min_score=-1.0), embedder)
    art = "Norma Ficticia 100|Artículo 2"
    doc_questions = [
        {"id": "d1", "question": "plazo para reclamar ante el Tribunal de Cuentas Imaginario", "gold": [art]},
        {"id": "d2", "question": "registro de proveedores de pelotas de goma",
         "gold": ["Norma Ficticia 100|Artículo 3"]}]
    cases = [{"id": "s1", "group": "sql_answerable", "question": "¿Cuántas órdenes hay?", "expect": ["sql"]},
             {"id": "d1", "group": "docs", "question": doc_questions[0]["question"], "expect": ["docs"]},
             {"id": "r1", "group": "router_cases", "question": "Borra todo.", "expect": ["refuse"]}]
    policy = load_policy(ROOT / "configs/domain/chilecompra.yaml")
    system = build_sql_system(build_schema_context(ROOT / "domains/chilecompra/glossary.yaml", policy.blocked_columns))
    good = "```sql\nSELECT COUNT(DISTINCT Codigo) AS n FROM oc_lineas\n```"
    llm = runner.RecordingLLM(FakeLLM([
        '{"route": "sql"}', "```sql\nSELECT RutSucursal FROM oc_lineas\n```", good, "Hay 2 órdenes.",
        '{"route": "docs"}', "Diez días [Norma Ficticia 100 art. 2].",
        '{"route": "refuse"}']))
    llm.system_prompt = system
    pipe = Text2SQLPipeline(llm, policy, db, system, 3, retriever=retriever)
    out = tmp_path / "pipeline.jsonl"
    dialogues = []
    runs = runner.run_pipeline_cases(pipe, llm, cases, {"s1": "select count(distinct Codigo) from oc_lineas"},
                                     doc_questions, runner.article_texts(CORPUS), out, {"commit": "abc"},
                                     dialogues=dialogues)

    saved = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert saved == runs[0] and [r["id"] for r in saved] == ["s1", "d1", "r1"]
    sql, docs, refused = saved
    assert (sql["status"], sql["strict"], sql["lenient"], sql["commit"]) == ("ok", True, True, "abc")
    assert [c["role"] for c in sql["calls"]] == ["router", "sql_generator", "sql_generator", "composer"]
    assert "answer" not in sql and "rows" not in sql and "sql" not in sql
    assert sql["retrieved"] is None
    assert (docs["status"], docs["gold_hit"], docs["cited"]) == ("ok", True, [art])
    assert [c["role"] for c in docs["calls"]] == ["router", "doc_writer"]
    assert art in docs["retrieved"] and docs["retrieval_ms"] >= 0 and docs["answer"].startswith("Diez")
    assert docs["similarity"] > docs["similarity_control"]
    assert (refused["route"], len(refused["calls"]), refused["gold_hit"]) == ("refuse", 1, None)

    rag = runner.run_rag(retriever, ranker, doc_questions, tmp_path / "rag.jsonl", {"commit": "abc"})
    assert rag[0]["ranked"][0] == art and len(rag[0]["ranked"]) == 6 and len(rag[0]["retrieved"]) <= 3

    sample = tmp_path / "dialogues.md"
    runner.write_dialogues(dialogues, 5, sample)
    text = sample.read_text(encoding="utf-8")
    assert "## s1" in text and "## d1" in text and "## r1" not in text
    assert "Hay 2 órdenes." not in text                              # composer text is never dumped

    meta = {"date": "2026-01-01", "model": "fake", "model_placement": "-", "embed_model": "hash",
            "embed_device": "cpu", "commit": "abc", "dirty": False, "trees": {"src": "1"}, "part": "all",
            "limit": 0, "k": 3, "min_score": 0.2, "missing_gold": []}
    summary = tmp_path / "summary.md"
    for run_records, rag_records in ((runs, rag), ([], rag), (runs, [])):
        runner.write_summary(summary, meta, run_records, rag_records)
        text = summary.read_text(encoding="utf-8")
        assert [line for line in text.splitlines() if line.startswith("## ")] == ["## Modelo", "## Agentes", "## RAG"]
    runner.write_summary(summary, meta, runs, rag)
    assert "Exactitud del enrutador: 3/3" in summary.read_text(encoding="utf-8")


def test_failed_model_call_is_recorded_without_stale_usage():
    llm = runner.RecordingLLM(FakeLLM(["uno"]))
    llm.chat([{"role": "user", "content": "x"}])
    llm.inner.last_usage = {"prompt_tokens": 99, "eval_duration": 5}
    with pytest.raises(Exception):
        llm.chat([{"role": "user", "content": "x"}])
    failed = llm.log[-1]
    assert failed["error"] and failed["prompt_tokens"] is None and failed["eval_ms"] is None and failed["sha1"] is None


def test_limit_is_per_group():
    cases = [{"group": g, "id": i} for i, g in enumerate("aaabbb")]
    assert [c["id"] for c in runner.limit_per_group(cases, 2)] == [0, 1, 3, 4]
    assert runner.limit_per_group(cases, 0) == cases
