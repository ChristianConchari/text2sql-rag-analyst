#!/usr/bin/env python3
"""Measures Ollama prefix (KV) cache reuse on the SQL route with the full system (V3 flags), per call.

Usage: uv run python scripts/eval_prefix_cache.py --mode {sql,sql_busted,full} [--limit N] [--layout legacy|cache_friendly]
                                                  [--max-tokens N] [--keep-alive 30m] [--out-dir eval/results]

Modes (all at temperature 0, native /api/chat so Ollama's timings are available, warm-up before the first question):
  sql         SQL route only (value lookup + generator/verifier loop), the layout under test.
  sql_busted  CONTROL, not a product feature: a unique nonce is prepended to the system prompt of every question, so
              the first call of each question cannot reuse the cache (the cold cost). Retries of a question share it.
  full        router -> SQL route -> composer through `Text2SQLPipeline.answer` (what the API runs): shows whether the
              router and composer calls evict the SQL prefix from the server's KV cache.
Per call it records prompt_eval_count (includes cached tokens), prompt_eval_duration (time really spent processing
the prompt: the cache metric), eval_count/eval_duration, load_duration, role and attempt number.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from eval_sql_ablation import git, load_questions, pct  # noqa: E402

from src.eval.scoring import score_rows  # noqa: E402
from src.runtime.features import features_from_dict  # noqa: E402
from src.runtime.pipelines.text2sql import build_default_pipeline  # noqa: E402
from src.runtime.providers.llm_client import MeteredLLM, OllamaNativeClient, load_config  # noqa: E402
from src.toolkits.db.executor import run_query  # noqa: E402

ABL = yaml.safe_load((ROOT / "configs/eval/sql_ablation.yaml").read_text(encoding="utf-8"))
NS = 1e9


class Busted:
    """Control: prefix the system message with a nonce so the server cannot reuse its cache for that call."""

    def __init__(self, inner: MeteredLLM):
        self.inner, self.nonce, self.roles = inner, "", []

    def chat(self, messages, max_tokens=512, temperature=None):
        messages = [dict(m) for m in messages]
        self.roles.append(role_of(messages[0]["content"]))
        if self.nonce:
            messages[0]["content"] = f"[{self.nonce}]\n" + messages[0]["content"]
        return self.inner.chat(messages, max_tokens=max_tokens, temperature=temperature)


def role_of(system_text: str) -> str:
    from src.rag.prompts.sql_prompt import COMPOSER_SYSTEM, ROUTER_SYSTEM, SQL_RULES
    if ROUTER_SYSTEM in system_text:
        return "router"
    if COMPOSER_SYSTEM in system_text:
        return "composer"
    return "sql" if system_text.startswith(SQL_RULES[:30]) else "docs"


class KeywordRetriever:
    """Stand-in for the dense retriever (needs torch): top articles by word overlap over data/docs/chunks.jsonl.
    Only its role is to put a real document-route prompt through the model; its quality is irrelevant here."""

    def __init__(self, k: int = 3):
        from src.rag.retrievers.base import Passage
        self.k, self.P = k, Passage
        self.chunks = [json.loads(line) for line in (ROOT / "data/docs/chunks.jsonl").read_text(encoding="utf-8").splitlines()]

    def search(self, query, k=None):
        words = {w for w in query.lower().replace("?", " ").split() if len(w) > 4}
        scored = sorted(((sum(w in c["text"].lower() for w in words), c) for c in self.chunks), key=lambda t: -t[0])
        return [self.P(c["art_id"], c["norma"], c["label"], c["article_number"], c["text"], 0.9) for n, c in scored[: self.k] if n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sql", "sql_busted", "full", "mixed"], required=True)
    ap.add_argument("--layout", default=None, help="prompt_layout override (default: from configs/pipeline)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="completion cap for the SQL generator (0 = loop default)")
    ap.add_argument("--keep-alive", default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval/results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    flags = dict(ABL["variants"]["V3"])
    if args.layout:
        flags["prompt_layout"] = args.layout
    if args.max_tokens:
        flags["max_completion_tokens"] = args.max_tokens
    features = features_from_dict(flags)
    cfg = replace(load_config(), model=ABL["model"], temperature=0.0)
    client = OllamaNativeClient(cfg, keep_alive=args.keep_alive)
    meter = MeteredLLM(client)
    bust = Busted(meter)
    pipe = build_default_pipeline(features, llm=bust, with_retriever=False)
    if args.mode == "mixed":
        pipe.retriever = KeywordRetriever()
    questions = load_questions(ABL["questions"])
    if args.limit:
        questions = questions[: args.limit]
    if args.mode == "mixed":   # alternate: a document question, then a SQL question
        docs = [q for q in yaml.safe_load((ROOT / "eval/doc_questions.yaml").read_text(encoding="utf-8")) if not q.get("out_of_corpus")]
        n = min(len(questions), len(docs), 12)
        questions = [x for i in range(n) for x in ({**docs[i], "kind": "docs", "difficulty": "-", "sql": None}, questions[i])]
    # warm-up with the real static prefix and a throwaway question, so load time and first-fill are out of the data
    meter.chat([{"role": "system", "content": pipe.system_prompt}, {"role": "user", "content": "Responde OK."}], max_tokens=4)
    now = dt.datetime.now()
    stem = f"prefix_cache_{args.mode}_{features.prompt_layout}{args.tag}_{now.strftime('%Y%m%d-%H%M%S')}_{git('rev-parse', '--short', 'HEAD')}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{stem}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for i, q in enumerate(questions, 1):
            meter.reset()
            bust.roles = []
            bust.nonce = uuid.uuid4().hex[:12] if args.mode == "sql_busted" else ""
            t0 = time.perf_counter()
            if args.mode in ("full", "mixed"):
                ans = pipe.answer(q["question"])
                status, sql, answer = ans.status, ans.sql, ans.answer
            else:
                res = pipe.generate_sql(q["question"])
                status, sql, answer = res.status, res.sql, None
            total = time.perf_counter() - t0
            strict = lenient = False
            if status == "ok" and sql and q.get("sql"):
                try:
                    strict, lenient = score_rows(q["sql"], run_query(pipe.db_path, q["sql"])[1], run_query(pipe.db_path, sql)[1])
                except Exception:  # noqa: BLE001 - scored as a miss
                    pass
            sql_idx = 0
            calls = []
            for c, role in zip(meter.calls, bust.roles):
                st = c.stats or {}
                attempt = None
                if role == "sql":
                    sql_idx += 1
                    attempt = sql_idx
                calls.append({"role": role, "attempt": attempt, "latency_s": round(c.latency_s, 3), "prompt_eval_count": st.get("prompt_eval_count"),
                              "prompt_eval_ms": round(st["prompt_eval_duration"] / 1e6, 1) if st.get("prompt_eval_duration") is not None else None,
                              "eval_count": st.get("eval_count"),
                              "eval_ms": round(st["eval_duration"] / 1e6, 1) if st.get("eval_duration") is not None else None,
                              "load_ms": round(st["load_duration"] / 1e6, 1) if st.get("load_duration") is not None else None,
                              "done_reason": st.get("done_reason")})
            rec = {"mode": args.mode, "layout": features.prompt_layout, "id": q["id"], "difficulty": q["difficulty"],
                   "status": status, "strict": strict, "lenient": lenient, "final_sql": sql, "answer": answer,
                   "latency_s": round(total, 3), "calls": calls}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{args.mode} {i}/{len(questions)}] {q['id']} {status:8} strict={strict!s:5} calls={len(calls)} {total:5.1f}s", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
