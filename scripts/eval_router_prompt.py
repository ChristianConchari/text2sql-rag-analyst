#!/usr/bin/env python3
"""Router prompt comparison: the ROUTER_SYSTEM of a git ref (default `main`) vs the current one, same case set.

Usage: uv run python scripts/eval_router_prompt.py [--old-ref main] [--runs 2] [--out-dir eval/results]
Cases: scripts/eval_router_layout.py `router_case_set` (SQL candidates, the 30 document questions, router_cases.yaml).
Legacy layout (no shared prefix), native Ollama client, model of configs/eval/sql_ablation.yaml, temperature 0.
The old prompt is read from the ref's src/rag/prompts/sql_prompt.py as a string literal (parsed, never executed).

BLOCKING (exit code 2), checked on every run of the NEW prompt:
  - an unsafe case (expect refuse only) routed to sql or docs;
  - an answerable SQL question not routed to sql;
  - a document question that the OLD prompt (same run) sent to docs and the new one does not.
refuse <-> out_of_scope confusions are reported, not blocking: neither runs a tool.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_router_layout import ABL, router_case_set  # noqa: E402
from src.agents.orchestration.router import ROUTES, route_question  # noqa: E402
from src.rag.prompts.sql_prompt import ROUTER_SYSTEM  # noqa: E402
from src.runtime.providers.llm_client import OllamaNativeClient, load_config  # noqa: E402

CLASSES = list(ROUTES) + ["unknown"]
TOOL_ROUTES = {"sql", "docs"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def router_prompt_at(ref: str) -> str:
    """ROUTER_SYSTEM as written in `ref`'s sql_prompt.py (the literal assignment; nothing is executed)."""
    tree = ast.parse(git("show", f"{ref}:src/rag/prompts/sql_prompt.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ROUTER_SYSTEM" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"ROUTER_SYSTEM not found in {ref}")


def confusion(rows: list[dict], row_key, col_key, row_classes: list[str], title: str) -> None:
    counts = Counter((row_key(r), col_key(r)) for r in rows)
    print(f"\n{title}")
    print(f"  {'':20}" + "".join(f"{c:>13}" for c in CLASSES))
    for rc in row_classes:
        if any(counts[(rc, c)] for c in CLASSES):
            print(f"  {rc:20}" + "".join(f"{counts[(rc, c)]:>13}" for c in CLASSES))


def blocking_for_run(rows: list[dict], k: int) -> list[str]:
    old, new = f"old_run{k}", f"new_run{k}"
    out = []
    for r in rows:
        if r["expect"] == ["refuse"] and r[new] in TOOL_ROUTES:
            out.append(f"run{k} UNSAFE->{r[new]} {r['id']}")
        if r["group"] == "sql_answerable" and r[new] != "sql":
            out.append(f"run{k} SQL lost {r['id']} -> {r[new]}")
        if r["group"] == "docs" and r[old] == "docs" and r[new] != "docs":
            out.append(f"run{k} DOCS lost {r['id']} -> {r[new]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-ref", default="main")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval/results")
    args = ap.parse_args()
    old_prompt = router_prompt_at(args.old_ref)
    if old_prompt == ROUTER_SYSTEM:
        print(f"WARNING: the router prompt at {args.old_ref} equals the current one")
    commit = git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")
    cases = router_case_set()
    llm = OllamaNativeClient(replace(load_config(), model=ABL["model"], temperature=0.0))
    rows = [dict(c) for c in cases]
    for k in range(1, args.runs + 1):
        for r in rows:
            r[f"old_run{k}"] = route_question(r["question"], llm, instruction=old_prompt)
            r[f"new_run{k}"] = route_question(r["question"], llm)
        print(f"run {k} done ({len(rows)} cases x 2 prompts)")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out_dir / f"router_prompt_comparison_{stamp}_{commit}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {out}  (old prompt: {args.old_ref}; model {ABL['model']}; temperature 0)")

    for k in range(1, args.runs + 1):
        for p in ("old", "new"):
            unstable = [r["id"] for r in rows if r[f"{p}_run{k}"] != r[f"{p}_run1"]]
            if k > 1:
                print(f"{p} prompt, run{k} vs run1: {len(unstable)} differences {unstable or ''}")
        confusion(rows, lambda r: r[f"old_run{k}"], lambda r: r[f"new_run{k}"], CLASSES,
                  f"run{k}: OLD route (rows) -> NEW route (columns), all {len(rows)} cases")
        for p in ("old", "new"):
            confusion([r for r in rows if r["expect"]], lambda r: "|".join(r["expect"]), lambda r: r[f"{p}_run{k}"],
                      sorted({"|".join(r["expect"]) for r in rows if r["expect"]}),
                      f"run{k}: {p.upper()} prompt, EXPECTED (rows) -> routed (columns), cases with an expectation")

    print("\nRoute changes old -> new (any run):")
    print(f"  {'id':5} {'group':19} {'expect':22} {'old r1/r2':20} {'new r1/r2':20} question")
    for r in rows:
        olds = [r[f"old_run{k}"] for k in range(1, args.runs + 1)]
        news = [r[f"new_run{k}"] for k in range(1, args.runs + 1)]
        if olds != news:
            ok = "" if not r["expect"] or all(n in r["expect"] for n in news) else "  <-- not expected"
            print(f"  {r['id']:5} {r['group']:19} {'|'.join(r['expect']) or '-':22} {'/'.join(olds):20} "
                  f"{'/'.join(news):20} {r['question'][:60]}{ok}")
    wrong = [r["id"] for r in rows if r["expect"] and any(r[f"new_run{k}"] not in r["expect"]
                                                          for k in range(1, args.runs + 1))]
    print(f"\nNEW prompt, cases not matching their expectation in some run: {wrong or 'none'}")
    blocking = [b for k in range(1, args.runs + 1) for b in blocking_for_run(rows, k)]
    print("BLOCKING:", blocking or "none")
    sys.exit(2 if blocking else 0)


if __name__ == "__main__":
    main()
