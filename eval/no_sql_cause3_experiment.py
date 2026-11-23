#!/usr/bin/env python3
"""Cause 3 experiment (length vs content) for the 10 NO_SQL declines + 3 unanswerable leakage controls.

Conditions (both at temperature=0, value_lookup=False, critic=False, max_retries=0, qwen2.5-coder:14b-8k):
  - A_length: V0 plain schema prompt + neutral padding to ~6,600 prompt tokens (isolates length).
  - B_pruned_glossary: V1 rules + pruned glossary to ~2,000 prompt tokens keeping business rules (isolates content).

Runs one evaluation at a time (shared local GPU) and writes raw JSONL to eval/results/no_sql_cause3_20261122.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.orchestration.sql_loop import run_sql_loop  # noqa: E402
from src.eval.scoring import score_rows  # noqa: E402
from src.policies.sql_verifier import load_policy  # noqa: E402
from src.rag.prompts.sql_prompt import (  # noqa: E402
    SQL_RULES,
    _clip,
    build_plain_schema_context,
    build_sql_system,
)
from src.runtime.providers.llm_client import LLMClient, LLMConfig, MeteredLLM  # noqa: E402
from src.toolkits.db.executor import run_query  # noqa: E402

TARGET_IDS = [
    "q12", "q14", "q15", "q16", "v05", "v06", "v15", "v16", "v18", "v19",
    "q18", "q19", "v25",
]
MODEL = "qwen2.5-coder:14b-8k"
OUT_PATH = ROOT / "eval/results/no_sql_cause3_20261122.jsonl"


def find_db_path() -> Path:
    local = ROOT / "data/chilecompra.duckdb"
    if local.exists():
        return local
    main_repo = ROOT.parent / "text2sql-rag-analyst/data/chilecompra.duckdb"
    if main_repo.exists():
        return main_repo
    raise FileNotFoundError("Could not locate data/chilecompra.duckdb")


def load_target_questions() -> list[dict]:
    by_id: dict[str, dict] = {}
    for rel in ("eval/candidate_questions.yaml", "eval/candidate_questions_v2.yaml"):
        for q in yaml.safe_load((ROOT / rel).read_text(encoding="utf-8")):
            if q.get("id") in TARGET_IDS:
                by_id[q["id"]] = q
    missing = [qid for qid in TARGET_IDS if qid not in by_id]
    if missing:
        raise ValueError(f"Missing target questions: {missing}")
    return [by_id[qid] for qid in TARGET_IDS]


def build_condition_a_prompt(db_path: Path, hidden_columns: frozenset[str], target_chars: int = 23700) -> str:
    """V0 plain schema + neutral padding up to ~6,600 tokens."""
    base = build_sql_system(build_plain_schema_context(db_path, "oc_lineas", hidden_columns))
    header = "\n\nNotas de archivo neutro (sin instrucciones ni datos del dominio):\n"
    pad_lines = [base + header]
    total = len(pad_lines[0])
    i = 1
    while total < target_chars:
        line = (
            f"- Nota neutra {i:03d}: Este párrafo es texto de relleno administrativo sin relación con "
            "consultas, esquemas ni reglas de negocio; se incluye únicamente para igualar la longitud del contexto.\n"
        )
        pad_lines.append(line)
        total += len(line)
        i += 1
    return "".join(pad_lines)


def build_condition_b_prompt(glossary_path: Path, hidden_columns: frozenset[str]) -> str:
    """V1 pruned glossary (~2,000 tokens) keeping SQL_RULES and business rules."""
    g = yaml.safe_load(glossary_path.read_text(encoding="utf-8"))
    hidden = {c.lower() for c in hidden_columns}
    lines = [f"Tabla {g['tabla']['nombre']}: {g['tabla']['grano']}.", "Columnas:"]
    for name, spec in g["columnas"].items():
        if name.lower() in hidden:
            continue
        quoted = f'"{name}"' if not name.isidentifier() else name
        desc = _clip(spec.get("desc", ""), 30)
        lines.append(f"- {quoted}: {desc}")
    lines.append("Reglas de negocio:")
    lines += [f"- {r['tema']}: {_clip(r['regla'], 120)}" for r in g.get("reglas_de_negocio", [])]
    return SQL_RULES + "\n\n" + "\n".join(lines)


def run_condition(name: str, system_prompt: str, questions: list[dict], llm: LLMClient, db_path: Path,
                  policy, fh) -> list[dict]:
    meter = MeteredLLM(llm)
    meter.chat([{"role": "user", "content": "Responde OK."}], max_tokens=4)
    rows = []
    for i, q in enumerate(questions, 1):
        meter.reset()
        t0 = time.perf_counter()
        res = run_sql_loop(
            q["question"],
            meter,
            policy,
            lambda s: run_query(db_path, s),
            system_prompt,
            max_retries=0,
            critic=False,
            max_completion_tokens=700,
        )
        total_s = time.perf_counter() - t0
        strict = lenient = False
        score_error = None
        if res.status == "ok" and q.get("kind") == "answerable" and q.get("sql"):
            try:
                strict, lenient = score_rows(q["sql"], run_query(db_path, q["sql"])[1], run_query(db_path, res.sql)[1])
            except Exception as e:  # noqa: BLE001
                score_error = str(e)[:200]
        calls = meter.calls
        ptoks = [c.prompt_tokens for c in calls]
        ctoks = [c.completion_tokens for c in calls]
        rec = {
            "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "model": MODEL,
            "condition": name,
            "system_chars": len(system_prompt),
            "id": q["id"],
            "kind": q["kind"],
            "question": q["question"],
            "status": res.status,
            "strict": strict,
            "lenient": lenient,
            "attempts": len(res.attempts),
            "final_sql": res.sql,
            "reason": res.reason,
            "prompt_tokens": None if None in ptoks else sum(ptoks),
            "completion_tokens": None if None in ctoks else sum(ctoks),
            "latency_s": round(total_s, 3),
            "score_error": score_error,
        }
        rows.append(rec)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        print(
            f"[{name} {i:2d}/{len(questions)}] {q['id']} ({q['kind']:12s}) "
            f"status={res.status:8s} strict={str(strict):5s} ptok={rec['prompt_tokens']} {total_s:5.1f}s",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    args = ap.parse_args()

    db_path = find_db_path()
    policy = load_policy(ROOT / "configs/domain/chilecompra.yaml")
    questions = load_target_questions()
    prompt_a = build_condition_a_prompt(db_path, policy.blocked_columns)
    prompt_b = build_condition_b_prompt(ROOT / "domains/chilecompra/glossary.yaml", policy.blocked_columns)

    print(f"Prompt A chars: {len(prompt_a)}")
    print(f"Prompt B chars: {len(prompt_b)}")

    llm = LLMClient(LLMConfig(provider="local", base_url=args.base_url, model=MODEL, temperature=0.0, timeout_s=120.0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        rows_a = run_condition("A_length", prompt_a, questions, llm, db_path, policy, fh)
        rows_b = run_condition("B_pruned_glossary", prompt_b, questions, llm, db_path, policy, fh)

    for label, rs in (("A_length", rows_a), ("B_pruned_glossary", rows_b)):
        ans = [r for r in rs if r["kind"] == "answerable"]
        unans = [r for r in rs if r["kind"] == "unanswerable"]
        print(
            f"\nSummary {label}: "
            f"answerable ok={sum(r['status']=='ok' for r in ans)}/{len(ans)}, "
            f"declined={sum(r['status']=='declined' for r in ans)}/{len(ans)}, "
            f"strict={sum(r['strict'] for r in ans)}/{len(ans)} | "
            f"unanswerable declined={sum(r['status']=='declined' for r in unans)}/{len(unans)}, "
            f"leaked(ok)={sum(r['status']=='ok' for r in unans)}/{len(unans)}"
        )


if __name__ == "__main__":
    main()
