# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb", "pyyaml"]
# ///
"""Run every reference SQL in eval/candidate_questions.yaml against the DuckDB file and print results.

Usage: uv run scripts/run_reference_queries.py [--db data/chilecompra.duckdb] [--questions eval/candidate_questions.yaml]
"""
import argparse
import time
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--db", type=Path, default=ROOT / "data" / "chilecompra.duckdb")
ap.add_argument("--questions", type=Path, default=ROOT / "eval" / "candidate_questions.yaml")
args = ap.parse_args()

con = duckdb.connect(str(args.db), read_only=True)
for q in yaml.safe_load(args.questions.read_text()):
    print(f"\n[{q['id']}] ({q['kind']}, {q['difficulty']}) {q['question']}")
    if "sql" not in q:
        print("  no reference SQL; expected:", q["expected"])
        continue
    t0 = time.time()
    try:
        rows = con.execute(q["sql"]).fetchall()
        print(f"  {len(rows)} row(s) in {time.time() - t0:.2f}s:", rows[:5])
    except duckdb.Error as e:  # report a failing reference query and keep going
        print("  ERROR:", e)
