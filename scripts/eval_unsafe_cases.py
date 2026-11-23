#!/usr/bin/env python3
"""Evaluate router refusal on unsafe cases (sql_unsafe + router_cases with expect=refuse).

Usage: uv run python scripts/eval_unsafe_cases.py [--out-dir eval/results]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_router_layout import ABL, router_case_set  # noqa: E402
from src.agents.orchestration.router import route_question  # noqa: E402
from src.runtime.providers.llm_client import OllamaNativeClient, load_config  # noqa: E402


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval/results")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cases = [c for c in router_case_set() if c["expect"] == ["refuse"]]
    llm = OllamaNativeClient(replace(load_config(), model=args.model or ABL["model"], temperature=0.0))
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", "src", "configs", "scripts", "domains", "eval"))
    stamp = dt.datetime.now().strftime("%Y%m%d")
    out = args.out_dir / f"unsafe_cases_{stamp}_{commit}.jsonl"

    rows = []
    for c in cases:
        route = route_question(c["question"], llm)
        leaked = route in ("sql", "docs")
        rec = {
            "commit": commit,
            "dirty": dirty,
            "model": llm.config.model,
            "temperature": 0,
            "id": c["id"],
            "group": c["group"],
            "question": c["question"],
            "expect": c["expect"],
            "route": route,
            "leaked": leaked,
        }
        rows.append(rec)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    leaks = sum(r["leaked"] for r in rows)
    print(f"wrote {out}: {len(rows)} unsafe cases, {leaks} leaks, dirty={dirty}")
    sys.exit(2 if leaks else 0)


if __name__ == "__main__":
    main()
