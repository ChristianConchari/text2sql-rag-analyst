#!/usr/bin/env python3
"""Router safety check for prompt_layout: legacy vs cache_friendly_router_router must route every question the same way.

Usage: uv run python scripts/eval_router_layout.py [--out-dir eval/results]
Questions: the 40 answerable SQL questions, the 30 document questions (25 in corpus + 5 out of corpus), the unsafe /
ambiguous / unanswerable ones of the candidate files and eval/router_cases.yaml. Any case that legacy routes to
"refuse" and cache_friendly does not is BLOCKING (exit code 2). Native Ollama client, temperature 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.orchestration.router import route_question  # noqa: E402
from src.runtime.pipelines.text2sql import build_default_pipeline  # noqa: E402
from src.runtime.providers.llm_client import OllamaNativeClient, load_config  # noqa: E402

ABL = yaml.safe_load((ROOT / "configs/eval/sql_ablation.yaml").read_text(encoding="utf-8"))


def load(path: str) -> list[dict]:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval/results")
    args = ap.parse_args()
    cases = []
    for p in ABL["questions"]:
        cases += [{"id": q["id"], "group": f"sql_{q['kind']}", "question": q["question"]} for q in load(p)]
    cases += [{"id": q["id"], "group": "docs_out_of_corpus" if q.get("out_of_corpus") else "docs",
               "question": q["question"]} for q in load("eval/doc_questions.yaml")]
    cases += [{"id": q["id"], "group": "router_cases", "question": q["question"]} for q in load("eval/router_cases.yaml")]
    llm = OllamaNativeClient(replace(load_config(), model=ABL["model"], temperature=0.0))
    prefix = build_default_pipeline(llm=llm, with_retriever=False).system_prompt
    rows, blocking = [], []
    for c in cases:
        leg = route_question(c["question"], llm)
        fri = route_question(c["question"], llm, prefix)
        rows.append({**c, "legacy": leg, "cache_friendly": fri})
        if leg == "refuse" and fri != "refuse":
            blocking.append(c["id"])
        if leg != fri:
            print(f"DIFF {c['id']} [{c['group']}] legacy={leg} cf_router={fri}: {c['question'][:70]}")
    out = args.out_dir / "router_layout_comparison.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    groups = sorted({r["group"] for r in rows})
    print(f"{'group':22} n  same  legacy_refuse cf_refuse")
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        print(f"{g:22} {len(rs):2} {sum(r['legacy'] == r['cache_friendly'] for r in rs):4}  "
              f"{sum(r['legacy'] == 'refuse' for r in rs):4}  {sum(r['cache_friendly'] == 'refuse' for r in rs):4}")
    print("BLOCKING (legacy refuse, cache_friendly not):", blocking or "none")
    sys.exit(2 if blocking else 0)


if __name__ == "__main__":
    main()
