"""Evaluate the document route: retrieval quality, "no evidence" behaviour and citation validity.

Usage (project environment):  uv run python scripts/eval_doc_route.py [--embedder real|hash] [--llm fake|real] [--misses]
  --embedder real   bge-m3 on CPU (needs the `rag` group: UV_TORCH_BACKEND=cpu uv sync --group rag); `hash` is a
                    toy embedder only useful to smoke-test the script
  --llm fake        deterministic stand-ins (a "faithful" model that cites the top fragment, an "adversarial" one
                    that cites a made-up article) to check the citation verifier; `real` uses configs/model/llm.yaml

Reports:
  1. Recall@{1,3,5,10} and MRR, strict gold | gold + gold_extra (same definition as scripts/eval_retrieval.py).
  2. Score separation between in-corpus and out-of-corpus questions, and the rate of correct "no evidence" at the
     configured threshold (configs/rag/retrieval.yaml, `--min-score` overrides it).
  3. Citation validity of the answer loop on the in-corpus questions.
"""
import argparse
import re
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.orchestration.doc_loop import run_doc_loop  # noqa: E402
from src.rag.retrievers.dense import build_dense_retriever, load_retrieval_config  # noqa: E402
from src.rag.retrievers.embedders import HashEmbedder  # noqa: E402

KS = (1, 3, 5, 10)
FAKE_FAITHFUL = "faithful"
FAKE_ADVERSARIAL = "adversarial"


class FakeAnswerer:
    """Deterministic model: cites the first fragment it is given (faithful) or a made-up article (adversarial)."""

    def __init__(self, mode: str):
        self.mode = mode

    def chat(self, messages, max_tokens=512, temperature=None):
        if self.mode == FAKE_ADVERSARIAL:
            return "Según la ley [Ley 19.886 art. 999]."
        ref = re.search(r'ref="([^"]+)"', messages[1]["content"]).group(1)
        return f"Según el texto recuperado [{ref}]."


def ranked_articles(retriever, question: str) -> list[str]:
    return [p.art_id for p in retriever.search(question["question"], k=max(KS))]


def recall_table(retriever, questions, show_misses: bool) -> None:
    for label, extended in (("strict gold", False), ("gold + gold_extra", True)):
        hits = {k: 0 for k in KS}
        mrr = 0.0
        for q in questions:
            gold = set(q["gold"]) | (set(q.get("gold_extra", [])) if extended else set())
            ranked = ranked_articles(retriever, q)
            pos = next((i for i, a in enumerate(ranked) if a in gold), None)
            if pos is not None:
                mrr += 1.0 / (pos + 1)
                for k in KS:
                    hits[k] += pos < k
            elif show_misses and extended:
                print(f"   miss {q['id']}: {q['question']}  top3={ranked[:3]}")
        n = len(questions)
        print(f"  {label:18s} " + " ".join(f"R@{k}={hits[k] / n:.2f}" for k in KS) + f"  MRR={mrr / n:.3f}")


def top_score(retriever, question: str) -> float:
    found = retriever.search(question, k=1)
    return found[0].score if found else float("-inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["real", "hash"], default="real")
    ap.add_argument("--llm", choices=["fake", "real"], default="fake")
    ap.add_argument("--min-score", type=float)
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()

    config = load_retrieval_config()
    min_score = config.min_score if args.min_score is None else args.min_score
    if args.embedder == "real":
        from src.rag.retrievers.embedders import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(config.model)
    else:
        embedder = HashEmbedder()
    questions = yaml.safe_load((ROOT / "eval/doc_questions.yaml").read_text(encoding="utf-8"))
    inside = [q for q in questions if not q.get("out_of_corpus")]
    outside = [q for q in questions if q.get("out_of_corpus")]

    ranking = build_dense_retriever(replace(config, k=max(KS), min_score=-1.0), embedder)   # no threshold: pure ranking
    print(f"{len(ranking.chunks)} chunks, {len(inside)} in-corpus + {len(outside)} out-of-corpus questions\n")
    print("1. Retrieval")
    recall_table(ranking, inside, args.misses)

    in_scores = [top_score(ranking, q["question"]) for q in inside]
    out_scores = [top_score(ranking, q["question"]) for q in outside]
    print("\n2. Evidence threshold")
    print(f"  top-1 score in-corpus : min {min(in_scores):.3f}  median {statistics.median(in_scores):.3f}")
    print(f"  top-1 score out-corpus: max {max(out_scores):.3f}  median {statistics.median(out_scores):.3f}")
    gap = min(in_scores) - max(out_scores)
    midpoint = (min(in_scores) + max(out_scores)) / 2
    print(f"  separable at one threshold: {'yes' if gap > 0 else 'NO'} (gap {gap:+.3f}; midpoint {midpoint:.3f})")
    correct_none = sum(s < min_score for s in out_scores)
    wrongly_none = sum(s < min_score for s in in_scores)
    print(f"  at min_score={min_score}: out-of-corpus correctly 'no evidence' {correct_none}/{len(outside)}; "
          f"in-corpus wrongly 'no evidence' {wrongly_none}/{len(inside)}")

    print(f"\n3. Citation validity (answer loop, --llm {args.llm}, in-corpus questions that pass the threshold)")
    answering = build_dense_retriever(replace(config, min_score=min_score), embedder)
    if args.llm == "real":
        from src.runtime.providers.llm_client import LLMClient
        models = {"real": LLMClient.from_config()}
    else:
        models = {FAKE_FAITHFUL: FakeAnswerer(FAKE_FAITHFUL), FAKE_ADVERSARIAL: FakeAnswerer(FAKE_ADVERSARIAL)}
    for name, llm in models.items():
        results = [run_doc_loop(q["question"], llm, answering) for q in inside]
        answered = [r for r in results if r.status != "no_evidence"]
        ok = [r for r in answered if r.status == "ok"]
        first_try = sum(r.attempts == 1 for r in ok)
        print(f"  {name:12s} answered {len(answered)}/{len(inside)}; valid citations {len(ok)}/{len(answered)}; "
              f"first try {first_try}/{len(answered)}; failed {sum(r.status == 'failed' for r in answered)}; "
              f"declined {sum(r.status == 'declined' for r in answered)}")


if __name__ == "__main__":
    main()
