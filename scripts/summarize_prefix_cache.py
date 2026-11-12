#!/usr/bin/env python3
"""Summarize eval/results/prefix_cache_*.jsonl: latency, prompt processing time and tokens per agent.

Usage: uv run python scripts/summarize_prefix_cache.py FILE [FILE ...] > summary.md
"""
import json
import statistics as st
import sys
from pathlib import Path


def pct(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def m(xs):
    xs = [x for x in xs if x is not None]
    return round(st.fmean(xs), 1) if xs else None


print("| run | n | strict | lenient | e2e mean s | p50 | p95 | prompt tokens sent | prompt ms total | decode ms total |")
print("|---|---|---|---|---|---|---|---|---|---|")
per_agent = []
for f in sys.argv[1:]:
    rs = [json.loads(line) for line in Path(f).read_text().splitlines()]
    lat = [r["latency_s"] for r in rs]
    calls = [c for r in rs for c in r["calls"]]
    name = Path(f).name.split("_20")[0].replace("prefix_cache_", "")
    print(f"| {name} | {len(rs)} | {sum(r['strict'] for r in rs)} | {sum(r['lenient'] for r in rs)} | {st.fmean(lat):.2f} | "
          f"{pct(lat, .5):.2f} | {pct(lat, .95):.2f} | {sum(c['prompt_eval_count'] or 0 for c in calls)} | "
          f"{sum(c['prompt_eval_ms'] or 0 for c in calls):.0f} | {sum(c['eval_ms'] or 0 for c in calls):.0f} |")
    for role in ("router", "sql", "composer", "docs"):
        cs = [c for c in calls if c["role"] == role]
        if cs:
            first = [c for c in cs if c["attempt"] in (None, 1)]
            retry = [c for c in cs if (c["attempt"] or 1) > 1]
            per_agent.append(f"| {name} | {role} | {len(cs)} | {m(c['prompt_eval_count'] for c in cs)} | "
                             f"{m(c['prompt_eval_ms'] for c in cs)} | {m(c['eval_count'] for c in cs)} | "
                             f"{m(c['eval_ms'] for c in cs)} | {m(c['latency_s'] * 1000 for c in cs)} | "
                             f"{m(c['prompt_eval_ms'] for c in first)} / {m(c['prompt_eval_ms'] for c in retry)} ({len(retry)}) |")
print("\n| run | agent | calls | prompt tok (incl. cached) | prompt_eval ms | completion tok | decode ms | call latency ms | prompt ms first / retry (n retry) |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(per_agent))
