# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb", "pyyaml"]
# ///
"""Score predicted SQL against reference SQL by execution accuracy (result-set equality).

Usage: uv run scripts/score_baseline.py <predictions.yaml> [--questions eval/candidate_questions.yaml] [--db data/chilecompra.duckdb]

Two criteria, both ignoring column names and rounding floats (to integers when |v| >= 100, else 2 decimals):
  strict  rows must match in order when the reference has ORDER BY, else as a multiset.
  lenient order is ignored, strings are stripped, and extra predicted columns are tolerated
          (some subset of the predicted columns, in order, must reproduce the reference rows).
Predicted SQL runs on a read-only connection. Non-SQL kinds (ambiguous / unanswerable / unsafe)
are scored on whether the prediction starts with an acceptable decision prefix.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import duckdb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.scoring import ACCEPT, lenient_eq, norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("predictions", type=Path)
ap.add_argument("--questions", type=Path, default=ROOT / "eval" / "candidate_questions.yaml")
ap.add_argument("--db", type=Path, default=ROOT / "data" / "chilecompra.duckdb")
args = ap.parse_args()

con = duckdb.connect(str(args.db), read_only=True)
preds = yaml.safe_load(args.predictions.read_text())
tally = Counter()
for q in yaml.safe_load(args.questions.read_text()):
    p = preds[q["id"]].strip()
    if q["kind"] == "answerable":
        try:
            ref = con.execute(q["sql"]).fetchall()
            got = con.execute(p).fetchall()
            ordered = "order by" in q["sql"].lower()
            strict = norm(ref, ordered) == norm(got, ordered)
            lenient = lenient_eq(ref, got)
            detail = "" if strict else f"ref={ref[:2]} got={got[:2]}"
        except Exception as e:  # noqa: BLE001 - any per-question failure is scored as a miss and reported
            strict = lenient = False
            detail = f"ERROR {str(e)[:120]}"
    else:
        strict = lenient = p.startswith(ACCEPT[q["kind"]])
        detail = "" if strict else p[:100]
    tally[(q["kind"], "strict", strict)] += 1
    tally[(q["kind"], "lenient", lenient)] += 1
    flag = "OK  " if strict else ("lenient" if lenient else "FAIL")
    print(f"{q['id']} {flag:7} [{q['kind']}/{q['difficulty']}] {detail[:230]}")
print()
for kind in ("answerable", "ambiguous", "unanswerable", "unsafe"):
    n = tally[(kind, "strict", True)] + tally[(kind, "strict", False)]
    if n:
        print(f"{kind}: strict {tally[(kind, 'strict', True)]}/{n}, lenient {tally[(kind, 'lenient', True)]}/{n}")
