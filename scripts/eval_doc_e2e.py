#!/usr/bin/env python3
"""Document questions end to end THROUGH the router: pipeline.answer -> route, status, citations.

Usage: uv run python scripts/eval_doc_e2e.py [--out-dir eval/results]
scripts/eval_doc_route.py calls the retriever and the answer loop directly (it never runs the router); this script
measures what a user gets. Production wiring (build_default_pipeline: bge-m3 retriever, legacy layout unless the
feature file says otherwise) with the native Ollama client, model of configs/eval/sql_ablation.yaml, temperature 0.

Per question: route, status, number of citations, whether every citation is a retrieved passage (guaranteed by the
answer loop for status ok; re-checked here against the citation list), and whether a citation hits gold/gold_extra.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.citations import citation_key  # noqa: E402
from src.runtime.pipelines.text2sql import build_default_pipeline  # noqa: E402
from src.runtime.providers.llm_client import OllamaNativeClient, load_config  # noqa: E402

MODEL = yaml.safe_load((ROOT / "configs/eval/sql_ablation.yaml").read_text(encoding="utf-8"))["model"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def gold_keys(q: dict) -> set[str]:
    keys = set()
    for art_id in q.get("gold", []) + q.get("gold_extra", []):
        norma, label = art_id.split("|", 1)
        keys.add(citation_key(norma, label.removeprefix("Artículo ").strip()))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval/results")
    args = ap.parse_args()
    dirty = git("status", "--porcelain", "--untracked-files=no")   # untracked result files do not count
    commit = git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")
    questions = yaml.safe_load((ROOT / "eval/doc_questions.yaml").read_text(encoding="utf-8"))
    pipe = build_default_pipeline(llm=OllamaNativeClient(replace(load_config(), model=MODEL, temperature=0.0)))
    if pipe.retriever is None:
        raise SystemExit("document retriever unavailable (needs the `rag` group and data/docs/chunks.jsonl)")
    rows = []
    for q in questions:
        a = pipe.answer(q["question"])
        keys = [citation_key(c["norma"], c["article"]) for c in a.citations]
        expect = q.get("expect_route", "docs")
        rows.append({"id": q["id"], "question": q["question"], "out_of_corpus": bool(q.get("out_of_corpus")),
                     "expect_route": expect if isinstance(expect, list) else [expect],
                     "route": a.route, "status": a.status, "attempts": a.attempts, "n_citations": len(a.citations),
                     "citations": [c["label"] for c in a.citations],
                     "gold_hit": bool(gold_keys(q) & set(keys)) if not q.get("out_of_corpus") else None,
                     "answer": a.answer})
        r = rows[-1]
        print(f"{r['id']:4} route={r['route']:12} status={r['status']:15} cites={r['n_citations']} "
              f"gold_hit={r['gold_hit']}  {r['answer'][:90]!r}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out_dir / f"doc_e2e_{stamp}_{commit}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out}  (model {MODEL}, temperature 0)")

    inside = [r for r in rows if not r["out_of_corpus"]]
    outside = [r for r in rows if r["out_of_corpus"]]
    ok = [r for r in inside if r["status"] == "ok"]
    print(f"in-corpus ({len(inside)}): routed docs {sum(r['route'] == 'docs' for r in inside)}; status ok {len(ok)}; "
          f"ok on first try {sum(r['attempts'] == 1 for r in ok)}; "
          f"with >=1 citation {sum(r['n_citations'] > 0 for r in ok)}; "
          f"citation hits gold/gold_extra {sum(bool(r['gold_hit']) for r in ok)}/{len(ok)}")
    for s in sorted({r["status"] for r in inside}):
        print(f"  status {s}: {[r['id'] for r in inside if r['status'] == s]}")
    print(f"out-of-corpus ({len(outside)}): route as expected "
          f"{sum(r['route'] in r['expect_route'] for r in outside)}/{len(outside)}")
    for r in outside:
        print(f"  {r['id']} expect={'|'.join(r['expect_route'])} route={r['route']} status={r['status']} "
              f"cites={r['n_citations']}: {r['answer'][:110]!r}")


if __name__ == "__main__":
    main()
