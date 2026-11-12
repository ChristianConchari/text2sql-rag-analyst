#!/usr/bin/env python3
"""Ablation of the SQL route with the real model: cumulative variants V0..V3 (configs/eval/sql_ablation.yaml).

Usage: uv run python scripts/eval_sql_ablation.py [--variants V0,V1] [--limit 5] [--model NAME] [--out-dir eval/results]

Runs the `answerable` questions of the configured files through `Text2SQLPipeline.generate_sql` (no router, no
composer) at temperature 0, scores execution accuracy with src/eval/scoring.py (strict and lenient, same as
score_baseline.py) and writes one JSONL per variant plus a CSV and Markdown summary into eval/results/.
Scoring runs the verified SQL (LIMIT added, masking flag column possibly added) unmasked against the reference, so it
measures SQL correctness, not the masked output. Safety rules are active in every variant (see
configs/pipeline/sql_features.yaml); `advisory` lists order-grain findings that a disabled critic did not enforce.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.scoring import score_rows  # noqa: E402
from src.runtime.features import features_from_dict  # noqa: E402
from src.runtime.pipelines.text2sql import build_default_pipeline  # noqa: E402
from src.runtime.providers.llm_client import LLMClient, MeteredLLM, load_config  # noqa: E402

CONFIG = ROOT / "configs/eval/sql_ablation.yaml"
CTX_LIMIT = 8192


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_questions(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        out += [q for q in yaml.safe_load((ROOT / p).read_text(encoding="utf-8")) if q.get("kind") == "answerable"]
    return out


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_variant(name: str, flags: dict, questions: list[dict], meta: dict, llm: LLMClient, db: Path, out: Path) -> list[dict]:
    meter = MeteredLLM(llm)
    pipe = build_default_pipeline(features_from_dict(flags), llm=meter, with_retriever=False)
    meter.chat([{"role": "user", "content": "Responde OK."}], max_tokens=4)    # warm-up: keep model load out of latency
    con = duckdb.connect(str(db), read_only=True)
    rows = []
    with out.open("w", encoding="utf-8") as fh:
        for i, q in enumerate(questions, 1):
            meter.reset()
            t0 = time.perf_counter()
            res = pipe.generate_sql(q["question"])
            total_s = time.perf_counter() - t0
            strict = lenient = False
            score_error = None
            if res.status == "ok":
                try:
                    strict, lenient = score_rows(q["sql"], con.execute(q["sql"]).fetchall(), con.execute(res.sql).fetchall())
                except Exception as e:  # noqa: BLE001 - a scoring failure is a miss, recorded
                    score_error = str(e)[:200]
            calls = meter.calls
            ptoks = [c.prompt_tokens for c in calls]
            ctoks = [c.completion_tokens for c in calls]
            rec = {
                **meta, "variant": name, "flags": flags, "id": q["id"], "difficulty": q["difficulty"],
                "question": q["question"], "status": res.status, "strict": strict, "lenient": lenient,
                "attempts": len(res.attempts), "llm_calls": len(calls), "final_sql": res.sql, "reason": res.reason,
                "rejections": [a.violations or ([f"execution_error: {a.error[:200]}"] if a.error else [])
                               for a in res.attempts],
                "advisory": [a.advisory for a in res.attempts],       # order-grain findings NOT enforced (critic off)
                "attempt_sql": [a.sql for a in res.attempts],
                "prompt_tokens": None if None in ptoks else sum(ptoks),
                "completion_tokens": None if None in ctoks else sum(ctoks),
                "max_prompt_tokens": None if None in ptoks or not ptoks else max(ptoks),
                "latency_s": round(total_s, 3), "llm_latency_s": round(sum(c.latency_s for c in calls), 3),
                "score_error": score_error,
            }
            rows.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{name} {i}/{len(questions)}] {q['id']} {res.status:8} strict={strict!s:5} lenient={lenient!s:5} "
                  f"att={len(res.attempts)} {total_s:5.1f}s", flush=True)
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def summarize(results: dict[str, list[dict]], meta: dict, stem: Path) -> None:
    diffs = ["easy", "medium", "hard"]
    lines = []
    head = ["variant", "n", "strict", "lenient", *[f"strict_{d}" for d in diffs], "lat_mean_s", "lat_p50_s", "lat_p95_s",
            "llm_lat_mean_s", "prompt_tok_mean", "completion_tok_mean", "retries_mean", "failed", "declined",
            "would_reject_grain"]
    for v, rs in results.items():
        n = len(rs)
        lat = [r["latency_s"] for r in rs]
        row = {
            "variant": v, "n": n, "strict": sum(r["strict"] for r in rs), "lenient": sum(r["lenient"] for r in rs),
            "lat_mean_s": round(statistics.fmean(lat), 2), "lat_p50_s": round(pct(lat, .5), 2),
            "lat_p95_s": round(pct(lat, .95), 2), "llm_lat_mean_s": round(statistics.fmean(r["llm_latency_s"] for r in rs), 2),
            "prompt_tok_mean": round(mean(r["prompt_tokens"] for r in rs) or 0), "completion_tok_mean":
            round(mean(r["completion_tokens"] for r in rs) or 0),
            "retries_mean": round(statistics.fmean(max(r["attempts"] - 1, 0) for r in rs), 2),
            "failed": sum(r["status"] == "failed" for r in rs), "declined": sum(r["status"] == "declined" for r in rs),
            "would_reject_grain": sum(any(a for a in r["advisory"]) for r in rs),
        }
        for d in diffs:
            sub = [r for r in rs if r["difficulty"] == d]
            row[f"strict_{d}"] = f"{sum(r['strict'] for r in sub)}/{len(sub)}"
        lines.append(row)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(lines)
    md = [f"# SQL ablation {meta['date']}", "", f"Model `{meta['model']}`, commit `{meta['commit']}`"
          f"{' (dirty)' if meta['dirty'] else ''}, temperature 0, {lines[0]['n'] if lines else 0} answerable questions.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in lines]
    names = list(results)
    first = results[names[0]] if names else []
    md += ["", "## Per question (S = strict, L = lenient only, - = miss)", "",
           "| id | difficulty | " + " | ".join(names) + " | changes |", "|" + "---|" * (len(names) + 3)]
    by = {v: {r["id"]: r for r in rs} for v, rs in results.items()}
    sym = lambda r: "S" if r["strict"] else ("L" if r["lenient"] else "-")  # noqa: E731
    for r0 in first:
        marks = [sym(by[v][r0["id"]]) for v in names if r0["id"] in by[v]]
        ch = "" if len(set(marks)) == 1 else " → ".join(marks)
        md.append(f"| {r0['id']} | {r0['difficulty']} | " + " | ".join(marks) + f" | {ch} |")
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (time estimate)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval/results")
    ap.add_argument("--db", type=Path, default=ROOT / "data/chilecompra.duckdb")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    names = [v for v in args.variants.split(",") if v] or list(cfg["variants"])
    questions = load_questions(cfg["questions"])
    if args.limit:
        questions = questions[: args.limit]
    llm = LLMClient(replace(load_config(), model=args.model or cfg["model"], temperature=0.0))
    llm.smoke()
    now = dt.datetime.now()
    meta = {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src", "configs", "scripts")),
            "model": llm.config.model, "date": now.strftime("%Y-%m-%d"), "run_ts": now.isoformat(timespec="seconds"),
            "temperature": 0}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d-%H%M%S") + ("-sample" if args.limit else "")
    results = {}
    t0 = time.perf_counter()
    for v in names:
        path = args.out_dir / f"sql_ablation_{v}_{stamp}_{meta['commit']}.jsonl"
        results[v] = run_variant(v, cfg["variants"][v], questions, meta, llm, args.db, path)
    summarize(results, meta, args.out_dir / f"sql_ablation_summary_{stamp}_{meta['commit']}")
    print(f"total {time.perf_counter() - t0:.0f}s; summary: sql_ablation_summary_{stamp}_{meta['commit']}.md")


if __name__ == "__main__":
    main()
