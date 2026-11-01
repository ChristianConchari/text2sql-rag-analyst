# /// script
# requires-python = ">=3.11"
# dependencies = ["sentence-transformers", "rank-bm25", "numpy", "pyyaml"]
# ///
"""Compare retrievers on the document questions: BM25, dense embedding models, BM25+dense (RRF) and
dense + cross-encoder re-ranking. Embeddings are cached in data/cache/ (delete to recompute).

Usage: UV_TORCH_BACKEND=cpu uv run scripts/eval_retrieval.py [--models BAAI/bge-m3 ...]
           [--rerankers BAAI/bge-reranker-v2-m3 ...] [--rerank-top 20 10 5] [--misses]

Retrieval is at chunk level; results are collapsed to the ranked list of distinct articles, and a
question is a hit at k when any gold article appears in the top-k articles. Reports Recall@{1,3,5,10},
MRR (each metric shown as strict-gold|gold+gold_extra) and mean per-query latency (query encoding / re-ranking), on CPU.
"""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10)
DEFAULT_MODELS = ["BAAI/bge-m3"]
DEFAULT_RERANKERS = ["BAAI/bge-reranker-v2-m3", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"]


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", text)


def prefixes(model: str) -> tuple[str, str]:
    return ("query: ", "passage: ") if "e5" in model else ("", "")


def dedupe(chunk_order: list[int], chunks: list[dict]) -> list[str]:
    seen, ranked = set(), []
    for i in chunk_order:
        a = chunks[i]["art_id"]
        if a not in seen:
            seen.add(a)
            ranked.append(a)
    return ranked


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    score: dict[str, float] = {}
    for lst in rank_lists:
        for r, a in enumerate(lst):
            score[a] = score.get(a, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda a: -score[a])


def metrics(ranked_by_q, questions, extended):
    hits = {k: 0 for k in KS}
    mrr = 0.0
    for q in questions:
        gold = set(q["gold"]) | (set(q.get("gold_extra", [])) if extended else set())
        pos = next((i for i, a in enumerate(ranked_by_q[q["id"]]) if a in gold), None)
        if pos is not None:
            mrr += 1.0 / (pos + 1)
            for k in KS:
                hits[k] += pos < k
    n = len(questions)
    return {k: hits[k] / n for k in KS}, mrr / n


def evaluate(name, ranked_by_q, questions, show_misses, extra=""):
    strict, mrr_s = metrics(ranked_by_q, questions, False)
    ext, mrr_e = metrics(ranked_by_q, questions, True)
    cols = " ".join(f"R@{k}={strict[k]:.2f}|{ext[k]:.2f}" for k in KS)
    print(f"{name:62s} {cols} MRR={mrr_s:.3f}|{mrr_e:.3f} {extra}")
    if show_misses:
        for q in questions:
            gold = set(q["gold"]) | set(q.get("gold_extra", []))
            ranked = ranked_by_q[q["id"]]
            pos = next((i for i, a in enumerate(ranked) if a in gold), None)
            if pos is None or pos >= 1:
                print(f"      {q['id']} [{'not found' if pos is None else f'gold(+extra) at rank {pos + 1}'}] {q['question']}\n          top3: {ranked[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--rerankers", nargs="*", default=DEFAULT_RERANKERS)
    ap.add_argument("--rerank-top", type=int, nargs="+", default=[20, 10, 5])
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in (ROOT / "data/docs/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = [q for q in yaml.safe_load((ROOT / "eval/doc_questions.yaml").read_text(encoding="utf-8"))
                 if not q.get("out_of_corpus")]   # out-of-corpus questions have no gold: see scripts/eval_doc_route.py
    cache = ROOT / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"{len(chunks)} chunks, {len(questions)} questions\n")

    bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
    bm25_rank = {q["id"]: dedupe(list(np.argsort(-np.array(bm25.get_scores(tokens(q["question"]))))), chunks) for q in questions}
    evaluate("BM25", bm25_rank, questions, False)

    for model_name in args.models:
        short = model_name.split("/")[-1]
        model = SentenceTransformer(model_name, device="cpu")
        qp, pp = prefixes(model_name)
        f = cache / f"emb_{short}_{len(chunks)}.npy"
        if f.exists():
            doc_emb = np.load(f)
        else:
            t0 = time.time()
            doc_emb = model.encode([pp + c["text"] for c in chunks], batch_size=16, normalize_embeddings=True)
            np.save(f, doc_emb)
            print(f"  indexed {len(chunks)} chunks with {short} in {time.time() - t0:.0f}s (cached)")
        enc_times, order = [], {}
        for q in questions:
            t0 = time.time()
            qe = model.encode([qp + q["question"]], normalize_embeddings=True)[0]
            scores = doc_emb @ qe
            enc_times.append(time.time() - t0)
            order[q["id"]] = list(np.argsort(-scores))
        dense_rank = {qid: dedupe(o, chunks) for qid, o in order.items()}
        evaluate(f"dense {short}", dense_rank, questions, args.misses, f"[query {1000 * np.mean(enc_times):.0f} ms]")
        evaluate(f"hybrid BM25 + {short} (RRF)", {qid: rrf([bm25_rank[qid], dense_rank[qid]]) for qid in dense_rank}, questions, False)

        for rr_name in args.rerankers:
            try:
                rr = CrossEncoder(rr_name, device="cpu", max_length=512)
            except Exception as e:  # model unavailable: report and continue
                print(f"  reranker {rr_name} FAILED: {str(e)[:100]}")
                continue
            for top in args.rerank_top:
                rr_times, rr_rank = [], {}
                for q in questions:
                    cand = order[q["id"]][:top]
                    t0 = time.time()
                    s = rr.predict([(q["question"], chunks[i]["text"]) for i in cand], batch_size=top)
                    rr_times.append(time.time() - t0)
                    rr_rank[q["id"]] = dedupe([cand[j] for j in np.argsort(-np.array(s))], chunks)
                evaluate(f"dense {short} + rerank {rr_name.split('/')[-1]} (top {top})", rr_rank, questions,
                         args.misses and top == args.rerank_top[0], f"[rerank {1000 * np.mean(rr_times):.0f} ms/query]")


if __name__ == "__main__":
    main()
