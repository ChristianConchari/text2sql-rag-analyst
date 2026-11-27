#!/usr/bin/env python3
"""Metrics asked by the course brief (model, agents, RAG), measured on the production wiring. Measures only.

Usage: EMBED_DEVICE=cuda uv run --group rag python scripts/eval_consigna.py [--part all|pipeline|rag] [--limit N]
       [--data-dir data] [--model NAME] [--runs 1] [--dump-dialogues 0] [--out-dir eval/results]

pipeline: `pipe.answer` on every case of `router_case_set()` (eval_router_layout.py) with the real model through
Ollama's native API at temperature 0. rag: the retriever alone on eval/doc_questions.yaml, no model. Writes one
JSONL per part (one line per case) and a Markdown summary; the metrics are the pure functions of
src/eval/consigna.py. Records never hold result rows or SQL; answer text is kept only for document questions
answered by the document route (law text). Exit code 1 if any model call failed (the files are still written).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_doc_e2e import gold_keys  # noqa: E402
from scripts.eval_router_layout import ABL, router_case_set  # noqa: E402
from src.eval import consigna as m  # noqa: E402
from src.eval.scoring import score_rows  # noqa: E402
from src.rag.citations import citation_key, passage_key  # noqa: E402
from src.rag.prompts.doc_prompt import DOC_SYSTEM  # noqa: E402
from src.rag.prompts.sql_prompt import COMPOSER_SYSTEM, ROUTER_SYSTEM  # noqa: E402
from src.rag.retrievers.corpus import load_chunks  # noqa: E402
from src.rag.retrievers.dense import build_dense_retriever, load_retrieval_config  # noqa: E402
from src.runtime.pipelines.text2sql import build_default_pipeline  # noqa: E402
from src.runtime.providers.llm_client import ChatModel, LLMError, MeteredLLM, OllamaNativeClient, load_config  # noqa: E402
from src.toolkits.db.executor import run_query  # noqa: E402

DOC_GROUPS = ("docs", "docs_out_of_corpus")
RANK_K = 10
KS = (1, 3, 5)


class RecordingLLM(MeteredLLM):
    """`MeteredLLM` that also keeps, per call, the agent role, token counts, server timings and a reply hash.

    The role comes from the start of the system message. `system_prompt` (the SQL generator's) is set once the
    pipeline exists. `replies` holds the raw texts of the current question, only for --dump-dialogues.
    """

    def __init__(self, inner: ChatModel):
        super().__init__(inner)
        self.system_prompt: str | None = None
        self.log: list[dict] = []
        self.replies: list[tuple[str, str]] = []

    def reset(self) -> None:
        super().reset()
        self.log, self.replies = [], []

    def _role(self, messages: list[dict]) -> str:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        for role, prefix in (("router", ROUTER_SYSTEM), ("composer", COMPOSER_SYSTEM), ("doc_writer", DOC_SYSTEM),
                             ("sql_generator", self.system_prompt)):
            if prefix and system.startswith(prefix):
                return role
        return "other"

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float | None = None) -> str:
        role, reply = self._role(messages), None
        try:
            reply = super().chat(messages, max_tokens=max_tokens, temperature=temperature)
            return reply
        finally:
            call = self.calls[-1]
            ok = reply is not None
            stats = (call.stats or {}) if ok else {}      # after a failure `last_usage` is the previous call's

            def ms(key: str) -> float | None:
                return None if stats.get(key) is None else round(stats[key] / 1e6, 1)
            self.log.append({
                "role": role, "latency_ms": round(call.latency_s * 1000, 1),
                "prompt_tokens": call.prompt_tokens if ok else None,
                "completion_tokens": call.completion_tokens if ok else None,
                "prompt_eval_ms": ms("prompt_eval_duration"), "eval_ms": ms("eval_duration"),
                "done_reason": stats.get("done_reason"),
                "sha1": hashlib.sha1(reply.encode("utf-8")).hexdigest() if ok else None, "error": not ok,
            })
            if ok:
                self.replies.append((role, reply))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_yaml(path: str) -> list[dict]:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def reference_data() -> tuple[dict[str, str], list[dict]]:
    """(reference SQL by question id, document questions)."""
    sql = {q["id"]: q["sql"] for p in ABL["questions"] for q in load_yaml(p) if q.get("kind") == "answerable"}
    return sql, load_yaml("eval/doc_questions.yaml")


def limit_per_group(cases: list[dict], limit: int) -> list[dict]:
    if not limit:
        return cases
    taken: Counter = Counter()
    out = []
    for c in cases:
        if taken[c["group"]] < limit:
            out.append(c)
            taken[c["group"]] += 1
    return out


def article_texts(chunks_path: Path) -> dict[str, str]:
    """Article id -> the text of all its chunks, in corpus order."""
    parts: dict[str, list[str]] = {}
    for c in load_chunks(chunks_path):
        parts.setdefault(c.art_id, []).append(c.text)
    return {art: "\n".join(texts) for art, texts in parts.items()}


def similarity(embedder, answer: str, gold: list[str], texts: dict[str, str]) -> float | None:
    """Highest cosine between the answer and each gold article's text (the embedders return normalized vectors)."""
    known = [texts[a] for a in gold if a in texts]
    if not known:
        return None
    vectors = embedder.encode_documents([answer, *known])
    return round(float(max(vectors[1:] @ vectors[0])), 4)


def run_case(pipe, llm: RecordingLLM, case: dict, ref_sql: dict[str, str], doc_questions: list[dict],
             texts: dict[str, str]) -> tuple[dict, list[tuple[str, str]]]:
    """Answer one case and build its record. Returns (record, raw model replies)."""
    llm.reset()
    t0 = time.perf_counter()
    a = pipe.answer(case["question"])
    latency = time.perf_counter() - t0
    calls, replies = list(llm.log), list(llm.replies)
    rec = {"id": case["id"], "group": case["group"], "expect": case["expect"], "route": a.route, "status": a.status,
           "attempts": a.attempts, "strict": None, "lenient": None, "gold_hit": None, "score_error": None,
           "latency_s": round(latency, 3), "trace": a.trace, "calls": calls}
    for key in ("prompt_tokens", "completion_tokens"):
        values = [c[key] for c in calls]
        rec[key] = None if None in values else sum(values)

    if case["group"] == "sql_answerable":
        rec["strict"] = rec["lenient"] = False
        if a.status == "ok" and a.sql:
            try:
                ref = ref_sql[case["id"]]
                rec["strict"], rec["lenient"] = score_rows(ref, run_query(pipe.db_path, ref)[1],
                                                           run_query(pipe.db_path, a.sql)[1])
            except Exception as e:  # noqa: BLE001 - a scoring failure is a miss; the message may quote data
                rec["score_error"] = type(e).__name__

    is_doc_case = case["group"] in DOC_GROUPS
    rec.update(out_of_corpus=case["group"] == "docs_out_of_corpus", retrieved=None, retrieved_scores=None,
               retrieval_ms=None, cited=None, gold=None, gold_extra=None, similarity=None, similarity_control=None)
    if pipe.retriever is not None and (is_doc_case or a.route == "docs"):
        t0 = time.perf_counter()
        passages = pipe.retriever.search(case["question"])
        rec["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        rec["retrieved"] = [p.art_id for p in passages]
        rec["retrieved_scores"] = [round(p.score, 4) for p in passages]
        ids = {passage_key(p): p.art_id for p in passages}
        keys = [citation_key(c["norma"], c["article"]) for c in a.citations]
        rec["cited"] = [ids.get(k, k) for k in keys]
    if is_doc_case:
        q = next(q for q in doc_questions if q["id"] == case["id"])
        rec["gold"], rec["gold_extra"] = q.get("gold", []), q.get("gold_extra", [])
        if a.route == "docs":
            rec["answer"] = a.answer
        if not rec["out_of_corpus"]:
            keys = {citation_key(c["norma"], c["article"]) for c in a.citations}
            rec["gold_hit"] = bool(gold_keys(q) & keys)
            if a.status == "ok" and pipe.retriever is not None:
                embedder = pipe.retriever.embedder
                rec["similarity"] = similarity(embedder, a.answer, rec["gold"] + rec["gold_extra"], texts)
                # Baseline: the same answer against the gold of the next in-corpus question of the file (cyclic).
                inside = [x for x in doc_questions if not x.get("out_of_corpus")]
                other = inside[(next(i for i, x in enumerate(inside) if x["id"] == q["id"]) + 1) % len(inside)]
                rec["similarity_control"] = similarity(
                    embedder, a.answer, other.get("gold", []) + other.get("gold_extra", []), texts)
    return rec, replies


def run_pipeline_cases(pipe, llm: RecordingLLM, cases: list[dict], ref_sql: dict[str, str],
                       doc_questions: list[dict], texts: dict[str, str], out: Path, meta: dict, runs: int = 1,
                       dialogues: list | None = None) -> list[list[dict]]:
    """Run every case `runs` times, writing one JSONL line per case and run. Returns the records of each run."""
    results: list[list[dict]] = []
    with out.open("w", encoding="utf-8") as fh:
        for run in range(1, runs + 1):
            records = []
            for i, case in enumerate(cases, 1):
                rec, replies = run_case(pipe, llm, case, ref_sql, doc_questions, texts)
                rec = {**meta, "run": run, **rec}
                records.append(rec)
                if dialogues is not None and run == 1:
                    dialogues.append((case, rec, replies))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[run {run} {i}/{len(cases)}] {case['id']:5} {case['group']:20} route={rec['route']:12} "
                      f"status={rec['status']:19} calls={len(rec['calls'])} {rec['latency_s']:6.1f}s", flush=True)
            results.append(records)
    return results


def run_rag(retriever, ranker, doc_questions: list[dict], out: Path, meta: dict) -> list[dict]:
    """Retriever alone: unthresholded ranking (`ranker`) and the production search, timed warm."""
    records = []
    if doc_questions:
        retriever.search(doc_questions[0]["question"])      # warm-up, discarded
    for q in doc_questions:
        t0 = time.perf_counter()
        context = retriever.search(q["question"])
        ms = (time.perf_counter() - t0) * 1000
        ranked = ranker.search(q["question"], k=RANK_K)
        records.append({**meta, "id": q["id"], "out_of_corpus": bool(q.get("out_of_corpus")),
                        "gold": q.get("gold", []), "gold_extra": q.get("gold_extra", []),
                        "ranked": [p.art_id for p in ranked], "ranked_scores": [round(p.score, 4) for p in ranked],
                        "retrieved": [p.art_id for p in context],
                        "retrieved_scores": [round(p.score, 4) for p in context], "search_ms": round(ms, 2)})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return records


def write_dialogues(dialogues: list, n: int, path: Path) -> None:
    """A sample of generator <-> verifier dialogues to grade by hand: those with objections first.

    Only the replies of the SQL generator and the document writer, plus the trace (safe by contract). The composer's
    text (built from rows) and the raw feedback (may quote database errors) are never written.
    """
    spoken = [d for d in dialogues if any(role in ("sql_generator", "doc_writer") for role, _ in d[2])]
    objected = [d for d in spoken if any(s["outcome"] in ("objection", "error") for s in d[1]["trace"])]
    rest = [d for d in spoken if d not in objected]
    # Alternate SQL and document questions among the ones without objections.
    seen: Counter = Counter()
    order = []
    for d in rest:
        order.append(seen[d[1]["route"]])
        seen[d[1]["route"]] += 1
    rest = [d for _, d in sorted(zip(order, rest), key=lambda pair: pair[0])]
    lines = ["# Muestra de diálogos generador-verificador", "", "Nota (1-5) y comentario: a mano.", ""]
    for case, rec, replies in (objected + rest)[:n]:
        lines += [f"## {case['id']} ({case['group']}, ruta {rec['route']}, estado {rec['status']})", "",
                  f"Pregunta: {case['question']}", "", "Traza:", ""]
        lines += [f"- {s['agent']} [{s['outcome']}]: {s['detail']}" for s in rec["trace"]]
        for i, (_, text) in enumerate([r for r in replies if r[0] in ("sql_generator", "doc_writer")], 1):
            lines += ["", f"Respuesta {i} del generador:", "", "~~~~", text.strip(), "~~~~"]
        lines += ["", "Nota: ", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- summary ----------
def num(x, digits: int = 2) -> str:
    return "n/d" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def frac(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f} %)" if b else "0/0"


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(str(v) for v in row) + " |" for row in rows), ""]


def model_section(runs: list[list[dict]]) -> list[str]:
    out = ["## Modelo", "",
           "Cómo se midió y límites: `pipe.answer` cronometrado en una sola máquina, una pregunta a la vez, con el "
           "modelo ya cargado; llamadas por la API nativa de Ollama (la API en producción usa `/v1`). El orden de las "
           "preguntas influye en la caché de prefijo. Los tokens de prompt incluyen los servidos desde la caché; "
           "`prompt_eval` es el tiempo realmente gastado en procesar el prompt. La similitud es el coseno bge-m3 entre "
           "la respuesta y el texto del artículo gold más parecido (gold + gold_extra): la referencia es texto legal, "
           "no una respuesta escrita por una persona, solo existe para la ruta docs con estado ok, no mide si la "
           "respuesta es correcta y las citas van incluidas en el texto. El control compara la misma respuesta con el "
           "gold de la pregunta siguiente (mismo corpus, así que no parte de cero). Los gold no están validados por un "
           "experto.", ""]
    if not runs:
        return [*out, "No se corrió (`--part rag`).", ""]
    records = runs[0]
    mm = m.model_metrics(records)
    rows = [[route, v["latency_s"]["n"], num(v["latency_s"]["p50"]), num(v["latency_s"]["p95"]),
             num(v["prompt_tokens"], 0), num(v["completion_tokens"], 0)]
            for route, v in [*sorted(mm["by_route"].items()), ("todas", mm["all"])]]
    out += ["### Tiempo de respuesta y tokens por pregunta", ""]
    out += table(["ruta", "n", "latencia p50 s", "latencia p95 s", "tokens prompt (media)",
                  "tokens respuesta (media)"], rows)
    rows = [[role, v["calls"], num(v["prompt_tokens"], 0), num(v["max_prompt_tokens"], 0),
             num(v["completion_tokens"], 0), num(v["latency_ms"]["p50"], 0), num(v["latency_ms"]["p95"], 0),
             num(v["prompt_eval_ms"]["p50"], 0), num(v["prompt_eval_ms"]["p95"], 0)]
            for role, v in sorted(mm["by_role"].items())]
    out += ["### Por llamada al modelo y agente", ""]
    out += table(["agente", "llamadas", "tok prompt (media)", "tok prompt (máx)", "tok respuesta (media)",
                  "latencia p50 ms", "latencia p95 ms", "prompt_eval p50 ms", "prompt_eval p95 ms"], rows)
    s = mm["similarity"]
    out += ["### Similitud semántica (docs en corpus con estado ok)", ""]
    out += table(["comparación", "n", "media", "p50", "p95"],
                 [["gold propio", s["n"], num(s["mean"], 3), num(s["p50"], 3), num(s["p95"], 3)],
                  ["control (gold de otra pregunta)", s["control_n"], num(s["control_mean"], 3), "", ""]])
    out += [f"Respuestas más parecidas a su gold que al de control: {frac(s['above_control'], s['control_n'])}.", ""]
    sql = [r for r in records if r["group"] == "sql_answerable"]
    docs = [r for r in records if r["group"] == "docs"]
    calls = [c for r in records for c in r["calls"]]
    out += ["### Contexto: exactitud en esta corrida", "",
            f"- SQL answerable: strict {frac(sum(r['strict'] for r in sql), len(sql))}, lenient "
            f"{frac(sum(r['lenient'] for r in sql), len(sql))}; errores al puntuar: "
            f"{sum(bool(r['score_error']) for r in sql)}.",
            f"- Docs en corpus: cita algún artículo gold {frac(sum(bool(r['gold_hit']) for r in docs), len(docs))}.",
            f"- Llamadas al modelo fallidas: {sum(c['error'] for c in calls)}/{len(calls)}; respuestas cortadas por "
            f"el tope de tokens: {sum(c['done_reason'] == 'length' for c in calls)}.", ""]
    return out


def agents_section(runs: list[list[dict]]) -> list[str]:
    out = ["## Agentes", "",
           "Cómo se midió y límites: todo sale de la traza y de las llamadas al modelo de la misma corrida. La "
           "coherencia es la exactitud del enrutador contra la ruta esperada (los candidatos ambiguos y sin respuesta "
           "no declaran ruta y no puntúan), la tabla de decisión final y los traspasos incoherentes; no mide el "
           "razonamiento. La calidad del diálogo es indirecta (objeciones reparadas y repetidas), sin juicio humano "
           "ni de un modelo juez; los errores de ejecución de SQL reintentan sin objeción del verificador y no "
           "cuentan como objeción. «Correcta» solo existe donde hay referencia. «Respuesta fija» y «Guardia de "
           "salida» son pasos deterministas (la respuesta fija no se cronometra).", ""]
    if not runs:
        return [*out, "No se corrió (`--part rag`).", ""]
    records = runs[0]
    rm = m.router_metrics(records)
    out += ["### Coherencia de decisión", "", f"Exactitud del enrutador: {frac(rm['correct'], rm['n'])}.", ""]
    out += table(["esperado \\ obtenido", *m.ROUTE_LABELS],
                 [[e, *(rm["matrix"][e][o] for o in m.ROUTE_LABELS)] for e in m.ROUTE_LABELS])
    if rm["misses"]:
        out += ["Fallos: " + "; ".join(f"{x['id']} (esperado {'|'.join(x['expect'])}, obtenido {x['route']})"
                                       for x in rm["misses"]) + ".", ""]
    decisions = m.decision_table(records)
    statuses = sorted({s for v in decisions.values() for s in v})
    out += table(["grupo \\ estado final", *statuses],
                 [[g, *(v.get(s, 0) for s in statuses)] for g, v in decisions.items()])
    for route, v in m.handoff_incoherence(records).items():
        out.append(f"- Traspaso incoherente {route}: {frac(len(v['ids']), v['n'])}"
                   + (f" ({', '.join(v['ids'])})" if v["ids"] else ""))
    if len(runs) > 1:
        st = m.decision_stability(runs)
        out.append(f"- Estabilidad en {st['runs']} corridas: misma ruta {frac(st['same_route'], st['n'])}, misma ruta "
                   f"y estado {frac(st['same_decision'], st['n'])}"
                   + (f"; cambian: {', '.join(st['changed'])}" if st["changed"] else ""))
    out += ["", "### Diálogo generador-verificador", ""]
    dm = m.dialogue_metrics(records)
    out += table(["verificador", "preguntas revisadas", "con objeción", "reparadas", "reparadas y correctas",
                  "objeción repetida", "aprobadas", "rondas medias hasta aprobar"],
                 [[v, d["reviewed"], d["with_objection"], d["repaired"],
                   f"{d['repaired_correct']}/{d['repaired_with_reference']}", d["repeated_objection"], d["approved"],
                   num(d["mean_rounds_to_approval"])] for v, d in dm.items()])
    red = m.redundancy_metrics(records)
    r, same = red["retries"], red["identical_replies"]
    out += ["### Redundancia", "",
            f"- Reintentos por pregunta con generador (n={r['n']}): media {num(r['mean'])}, máximo {num(r['max'])}; "
            f"con al menos un reintento: {r['with_retry']}.",
            f"- Respuestas del modelo idénticas dentro de una misma pregunta: {same['duplicated']} de "
            f"{same['calls']} llamadas, en {same['questions']} preguntas.",
            "- En `refuse` y `out_of_scope` 1 llamada significa que se evitó el resto del flujo.", ""]
    out += table(["ruta", "preguntas", "llamadas (media)", "mín", "máx"],
                 [[k, v["n"], num(v["mean"]), v["min"], v["max"]] for k, v in sorted(red["calls_by_route"].items())])
    out += ["### Latencia por agente", ""]
    out += table(["agente", "pasos", "p50 ms", "p95 ms"],
                 [[a, v["n"], num(v["p50"], 1), num(v["p95"], 1)] for a, v in m.agent_latency(records).items()])
    return out


def rag_section(rag: list[dict], runs: list[list[dict]], meta: dict) -> list[str]:
    out = ["## RAG", "",
           "Cómo se midió y límites: relevancia por artículo contra los gold de `eval/doc_questions.yaml` (elegidos "
           "por un asistente, sin validación experta; `gold_extra` se añadió después de ver la primera corrida). Con "
           "pocos artículos gold por pregunta, precision@k tiene un techo bajo, y un artículo pertinente que no esté "
           "anotado cuenta como fallo. Varios gold pueden ser alternativas, así que recall@k es más exigente que "
           "hit@k. El ranking usa k=10 sin umbral; el contexto usa k y umbral de producción, calibrado con estas "
           "mismas preguntas. La latencia es en caliente, con el modelo de lenguaje sin generar a la vez.", ""]
    if rag:
        inside = [r for r in rag if not r["out_of_corpus"]]
        results = {name: m.retrieval_metrics([r["ranked"] for r in inside], gold, KS) for name, gold in (
            ("gold", [r["gold"] for r in inside]),
            ("gold + gold_extra", [r["gold"] + r["gold_extra"] for r in inside]))}
        keys = [f"{kind}@{k}" for kind in ("recall", "precision", "hit") for k in KS] + ["mrr"]
        out += ["### Recuperación (preguntas dentro del corpus)", ""]
        out += table(["métrica", *(f"{name} (n={v['n']})" for name, v in results.items())],
                     [[k, *(num(v[k], 3) for v in results.values())] for k in keys])
        t = m.percentiles([r["search_ms"] for r in rag])
        out += [f"Tiempo de recuperación (n={t['n']}): p50 {num(t['p50'], 1)} ms, p95 {num(t['p95'], 1)} ms.", ""]
    else:
        out += ["Ranking y latencia no medidos (`--part pipeline`).", ""]
    # With the pipeline run there are citations; without it, the rag records give the context only.
    source = [r for r in runs[0] if r["group"] in DOC_GROUPS] if runs else rag
    both = [m.context_coverage(source, include_extra=x) for x in (False, True)]
    out += [f"### Cobertura de contexto útil (k={meta['k']}, umbral {meta['min_score']})", ""]
    out += table(["medida", "gold", "gold + gold_extra"], [
        ["preguntas en corpus con ≥1 gold en el contexto",
         *(frac(c["in_corpus_with_gold"], c["in_corpus"]) for c in both)],
        ["artículos del contexto que son gold", *(frac(c["gold_in_context"], c["retrieved"]) for c in both)],
        ["artículos del contexto citados (respuestas ok)",
         *(frac(c["cited"], c["retrieved_when_cited"]) if runs else "n/d" for c in both)],
        ["preguntas fuera del corpus con contexto vacío",
         *(frac(c["out_of_corpus_empty"], c["out_of_corpus"]) for c in both)]])
    if meta["missing_gold"]:
        out += [f"Artículos gold que no están en el corpus: {', '.join(meta['missing_gold'])}.", ""]
    return out


def write_summary(path: Path, meta: dict, runs: list[list[dict]], rag: list[dict]) -> None:
    counts = Counter(r["group"] for r in runs[0]) if runs else Counter()
    head = [f"# Métricas de la consigna {meta['date']}", "",
            f"- Modelo: `{meta['model']}`, temperatura 0 ({meta['model_placement']})",
            f"- Embeddings: `{meta['embed_model']}` en `{meta['embed_device']}`",
            f"- Commit `{meta['commit']}`{' (dirty)' if meta['dirty'] else ''}",
            *(f"- Árbol `{name}`: `{h}`" for name, h in meta["trees"].items()),
            f"- Parte: `{meta['part']}`; límite por grupo: {meta['limit'] or 'sin límite'}; corridas del pipeline: "
            f"{len(runs)} (las tablas usan la primera)"]
    if runs:
        head.append(f"- Casos del pipeline ({len(runs[0])}): " + ", ".join(f"{g} {n}" for g, n in counts.items()))
    if rag:
        outside = sum(r["out_of_corpus"] for r in rag)
        head.append(f"- Preguntas de recuperación ({len(rag)}): {len(rag) - outside} en corpus, {outside} fuera")
    lines = [*head, "", *model_section(runs), *agents_section(runs), *rag_section(rag, runs, meta)]
    path.write_text("\n".join(lines), encoding="utf-8")


def model_placement(llm: OllamaNativeClient) -> str:
    """How much of the loaded model sits in GPU memory (it matters for the latency figures)."""
    for x in loaded_models(llm):
        if x.get("name") == llm.config.model and x.get("size"):
            share = x.get("size_vram", 0) / x["size"]
            note = "" if share >= 0.999 else "; las latencias del modelo incluyen la parte que corre en CPU"
            return f"{100 * share:.0f} % del modelo en GPU según Ollama{note}"
    return "ubicación del modelo desconocida"


def loaded_models(llm: OllamaNativeClient) -> list[dict]:
    return llm._request("GET", "/api/ps", base_url=llm._native_base()).get("models", [])


def unload_model(llm: OllamaNativeClient, wait_s: float = 30.0) -> None:
    """Ask Ollama to free the model before the embedder loads, so the order is the one of an API start.

    With the model already resident on a 12 GB GPU, loading bge-m3 runs out of memory. Without a reachable Ollama
    (`--part rag` needs none) this does nothing.
    """
    try:
        if not any(x.get("name") == llm.config.model for x in loaded_models(llm)):
            return
        llm._request("POST", "/api/generate", {"model": llm.config.model, "keep_alive": 0}, base_url=llm._native_base())
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and any(x.get("name") == llm.config.model for x in loaded_models(llm)):
            time.sleep(0.5)
    except LLMError as e:
        print(f"could not unload the model: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("all", "pipeline", "rag"), default="all")
    ap.add_argument("--limit", type=int, default=0, help="N cases per group (smoke run)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=1, help="repeat the pipeline cases to measure decision stability")
    ap.add_argument("--dump-dialogues", type=int, default=0, metavar="N", help="sample of N dialogues to grade by hand")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval/results")
    args = ap.parse_args(argv)
    if args.runs < 1:
        ap.error("--runs must be at least 1")

    data = args.data_dir.resolve()
    cfg = replace(load_retrieval_config(), chunks_path=data / "docs/chunks.jsonl", cache_dir=data / "cache")
    db = data / "chilecompra.duckdb"
    with_pipeline = args.part in ("all", "pipeline")
    for path in [cfg.chunks_path, *([db] if with_pipeline else [])]:
        if not path.exists():
            raise SystemExit(f"missing {path}; see 'Reproduce the data and the evaluation' in README.md")

    from src.rag.retrievers.embedders import SentenceTransformerEmbedder
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", "src", "configs", "domains", "scripts"))
    trees = {d: git("rev-parse", f"HEAD:{d}") for d in ("src", "configs", "domains")}
    now = dt.datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S") + ("-sample" if args.limit else "")
    model = args.model or ABL["model"]
    ref_sql, doc_questions = reference_data()
    texts = article_texts(cfg.chunks_path)
    client = OllamaNativeClient(replace(load_config(), model=model, temperature=0.0))
    unload_model(client)
    embedder = SentenceTransformerEmbedder(cfg.model)
    retriever = build_dense_retriever(cfg, embedder)
    meta = {"commit": commit, "dirty": dirty, "model": model, "date": now.strftime("%Y-%m-%d"), "temperature": 0}
    summary_meta = {**meta, "trees": trees, "part": args.part, "limit": args.limit, "embed_model": cfg.model,
                    "embed_device": embedder.device, "k": cfg.k, "min_score": cfg.min_score,
                    "model_placement": "no se usó el modelo de lenguaje",
                    "missing_gold": sorted({a for q in doc_questions for a in q.get("gold", []) + q.get("gold_extra", [])
                                            if a not in texts})}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = lambda part, ext: args.out_dir / f"consigna_{part}_{stamp}_{commit}.{ext}"  # noqa: E731
    t0 = time.perf_counter()

    runs: list[list[dict]] = []
    if with_pipeline:
        os.environ["DB_PATH"] = str(db)
        client.smoke()
        llm = RecordingLLM(client)
        pipe = build_default_pipeline(llm=llm, with_retriever=False)
        if pipe.features.value_lookup and pipe.value_index is None:
            raise SystemExit("value lookup unavailable: this would not be the production wiring")
        pipe.retriever = retriever
        llm.system_prompt = pipe.system_prompt
        llm.chat([{"role": "user", "content": "Responde OK."}], max_tokens=4)    # warm-up: keep model load out
        retriever.search("plazo de pago")                                        # same for the embedder
        summary_meta["model_placement"] = model_placement(client)
        cases = limit_per_group(router_case_set(), args.limit)
        dialogues: list = []
        runs = run_pipeline_cases(pipe, llm, cases, ref_sql, doc_questions, texts, name("pipeline", "jsonl"), meta,
                                  args.runs, dialogues)
        if args.dump_dialogues:
            write_dialogues(dialogues, args.dump_dialogues, name("dialogues", "md"))

    rag: list[dict] = []
    if args.part in ("all", "rag"):
        questions = [{**q, "group": "docs_out_of_corpus" if q.get("out_of_corpus") else "docs"} for q in doc_questions]
        ranker = build_dense_retriever(replace(cfg, min_score=-1.0), embedder)
        rag = run_rag(retriever, ranker, limit_per_group(questions, args.limit), name("rag", "jsonl"), meta)

    summary = name("summary", "md")
    write_summary(summary, summary_meta, runs, rag)
    print(f"total {time.perf_counter() - t0:.0f}s; summary: {summary}")
    failed = sum(c["error"] for run in runs for r in run for c in r["calls"])
    if failed:
        print(f"WARNING: {failed} model calls failed; the metrics include those failures", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
