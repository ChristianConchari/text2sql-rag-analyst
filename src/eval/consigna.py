"""Metrics asked by the course brief (model, agents, RAG), computed from per-question records.

Pure functions over plain dicts: no model, no I/O. scripts/eval_consigna.py builds the records.

A pipeline record has: id, group, expect (accepted routes, may be empty), route, status, attempts, latency_s,
strict / lenient / gold_hit (None when there is no reference), trace (TraceStep dicts), calls (one dict per model
call: role, latency_ms, prompt_tokens, completion_tokens, prompt_eval_ms, sha1, ...), retrieved and cited (article
ids, or None), gold, gold_extra, similarity, similarity_control.

A rate is reported as counts (numerator and denominator); a mean or percentile of nothing is None, never 0.
"""
from __future__ import annotations

import statistics
from collections import Counter

ROUTE_LABELS = ("sql", "docs", "out_of_scope", "refuse", "unknown")
VERIFIERS = ("Verificador SQL", "Verificador de citas")
GENERATORS = ("Generador SQL", "Redactor")


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _interpolate(s: list[float], p: float) -> float:
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def percentiles(xs) -> dict:
    """{n, p50, p95} with linear interpolation (same as `pct` in scripts/eval_sql_ablation.py). None values ignored."""
    s = sorted(x for x in xs if x is not None)
    if not s:
        return {"n": 0, "p50": None, "p95": None}
    return {"n": len(s), "p50": _interpolate(s, .5), "p95": _interpolate(s, .95)}


def router_metrics(records: list[dict]) -> dict:
    """Accuracy over the cases with a non-empty `expect`, and the expected x obtained matrix.

    When several routes are accepted, a hit counts on the diagonal of the obtained route and a miss on the row of
    the first accepted route.
    """
    matrix = {e: {o: 0 for o in ROUTE_LABELS} for e in ROUTE_LABELS}
    n = correct = 0
    misses = []
    for r in records:
        if not r["expect"]:
            continue
        n += 1
        hit = r["route"] in r["expect"]
        correct += hit
        matrix[r["route"] if hit else r["expect"][0]][r["route"]] += 1
        if not hit:
            misses.append({"id": r["id"], "expect": list(r["expect"]), "route": r["route"]})
    return {"n": n, "correct": correct, "matrix": matrix, "misses": misses}


def decision_table(records: list[dict]) -> dict[str, dict[str, int]]:
    """group -> final status -> count."""
    table: dict[str, Counter] = {}
    for r in records:
        table.setdefault(r["group"], Counter())[r["status"]] += 1
    return {g: dict(c) for g, c in table.items()}


def handoff_incoherence(records: list[dict]) -> dict:
    """The router hands a question to a tool and the next agent says the tool cannot answer a question it should.

    sql: answerable question routed to sql, generator answered NO_SQL (status "declined").
    docs: in-corpus question routed to docs, ending in "no_evidence" or "declined".
    """
    rules = {"sql": ("sql_answerable", {"declined"}), "docs": ("docs", {"no_evidence", "declined"})}
    out = {}
    for route, (group, bad) in rules.items():
        routed = [r for r in records if r["group"] == group and r["route"] == route]
        out[route] = {"n": len(routed), "ids": [r["id"] for r in routed if r["status"] in bad]}
    return out


def _dialogue(records: list[dict], verifier: str) -> dict:
    reviewed = objected = repaired = repaired_ref = repaired_correct = repeated = 0
    rounds = []
    for r in records:
        steps = [s for s in r["trace"] if s["agent"] == verifier]
        if not steps:
            continue
        reviewed += 1
        approvals = [s["attempt"] for s in steps if s["outcome"] == "ok"]
        if r["status"] == "ok" and approvals:
            rounds.append(approvals[-1])
        objections = {s["attempt"]: s["detail"] for s in steps if s["outcome"] == "objection"}
        if not objections:
            continue
        objected += 1
        repeated += any(objections.get(n + 1) == detail for n, detail in objections.items())
        if r["status"] != "ok":
            continue
        repaired += 1
        verdicts = [v for v in (r.get("strict"), r.get("gold_hit")) if v is not None]
        if verdicts:
            repaired_ref += 1
            repaired_correct += any(verdicts)
    return {"reviewed": reviewed, "with_objection": objected, "repaired": repaired,
            "repaired_with_reference": repaired_ref, "repaired_correct": repaired_correct,
            "repeated_objection": repeated, "approved": len(rounds), "mean_rounds_to_approval": _mean(rounds)}


def dialogue_metrics(records: list[dict]) -> dict[str, dict]:
    """Generator <-> verifier dialogue per verifier, from the trace steps.

    reviewed: questions the verifier saw; repaired: final status "ok" after at least one objection;
    repaired_correct: of those with a reference (`repaired_with_reference`), strict or gold_hit;
    repeated_objection: same objection text in two consecutive attempts.
    """
    return {v: _dialogue(records, v) for v in VERIFIERS}


def _generator_rounds(record: dict) -> int:
    """Attempts of the generator or document writer, from the trace (a NO_SQL attempt counts too)."""
    # The SQL composer shares the name "Redactor" with the document writer but has no attempt number.
    return max((s["attempt"] for s in record["trace"] if s["agent"] in GENERATORS and s["attempt"] is not None), default=0)


def redundancy_metrics(records: list[dict]) -> dict:
    """Retries, identical model replies within a question and model calls per route."""
    retries = [n - 1 for n in map(_generator_rounds, records) if n]
    duplicated = [len(h) - len(set(h)) for h in
                  ([c["sha1"] for c in r["calls"] if c.get("sha1")] for r in records)]
    by_route: dict[str, list[int]] = {}
    for r in records:
        by_route.setdefault(r["route"], []).append(len(r["calls"]))
    return {
        "retries": {"n": len(retries), "mean": _mean(retries), "max": max(retries, default=None),
                    "with_retry": sum(x > 0 for x in retries)},
        "identical_replies": {"calls": sum(len(r["calls"]) for r in records), "duplicated": sum(duplicated),
                              "questions": sum(d > 0 for d in duplicated)},
        "calls_by_route": {k: {"n": len(v), "mean": _mean(v), "min": min(v), "max": max(v)}
                           for k, v in by_route.items()},
    }


def agent_latency(records: list[dict]) -> dict[str, dict]:
    """p50/p95 of `duration_ms` per trace agent. "Redactor" is split: the document writer has an attempt number."""
    durations: dict[str, list[float]] = {}
    for r in records:
        for s in r["trace"]:
            name = s["agent"]
            if name == "Redactor":
                name = "Redactor (SQL)" if s["attempt"] is None else "Redactor (docs)"
            durations.setdefault(name, []).append(s["duration_ms"])
    return {name: percentiles(v) for name, v in durations.items()}


def retrieval_metrics(ranked: list[list[str]], gold: list[list[str]], ks=(1, 3, 5)) -> dict:
    """Article-level recall@k, precision@k, hit@k and MRR, averaged over the questions that have gold articles.

    `ranked[i]`: retrieved article ids of question i, best first, without repeats; `gold[i]`: its gold ids.
    precision@k divides by k even if fewer articles were retrieved. MRR is 0 when no gold article is in the list.
    """
    pairs = [(r, set(g)) for r, g in zip(ranked, gold, strict=True) if g]
    out: dict = {"n": len(pairs)}
    for k in ks:
        found = [(len(set(r[:k]) & g), len(g)) for r, g in pairs]
        out[f"recall@{k}"] = _mean([f / total for f, total in found])
        out[f"precision@{k}"] = _mean([f / k for f, _ in found])
        out[f"hit@{k}"] = _mean([float(f > 0) for f, _ in found])
    out["mrr"] = _mean([next((1 / i for i, a in enumerate(r, 1) if a in g), 0.0) for r, g in pairs])
    return out


def context_coverage(records: list[dict], include_extra: bool = True) -> dict:
    """Useful context, over the passages retrieved with the production k and threshold.

    in_corpus_with_gold: in-corpus questions with at least one gold article in the context;
    gold_in_context / retrieved: pooled share of context articles that are gold;
    cited / retrieved_when_cited: pooled share of context articles the answer cited (status "ok" only);
    out_of_corpus_empty: out-of-corpus questions that got an empty context.
    """
    inside = with_gold = gold_n = retrieved_n = cited_n = cited_base = outside = empty = 0
    for r in records:
        if r.get("retrieved") is None:
            continue
        context = set(r["retrieved"])
        if r["out_of_corpus"]:
            outside += 1
            empty += not context
            continue
        gold = set(r["gold"]) | (set(r.get("gold_extra") or []) if include_extra else set())
        inside += 1
        with_gold += bool(context & gold)
        gold_n += len(context & gold)
        retrieved_n += len(context)
        if r.get("cited") is not None and r.get("status") == "ok":
            cited_n += len(context & set(r["cited"]))
            cited_base += len(context)
    return {"in_corpus": inside, "in_corpus_with_gold": with_gold, "gold_in_context": gold_n,
            "retrieved": retrieved_n, "cited": cited_n, "retrieved_when_cited": cited_base,
            "out_of_corpus": outside, "out_of_corpus_empty": empty}


def model_metrics(records: list[dict]) -> dict:
    """Response time and tokens per route, tokens and prompt processing time per agent role, similarity."""
    routes: dict[str, list[dict]] = {}
    roles: dict[str, list[dict]] = {}
    for r in records:
        routes.setdefault(r["route"], []).append(r)
        for c in r["calls"]:
            roles.setdefault(c["role"], []).append(c)

    def per_question(rs: list[dict]) -> dict:
        def total(r: dict, key: str) -> int | None:
            values = [c.get(key) for c in r["calls"]]
            return None if None in values else sum(values)
        return {"latency_s": percentiles([r["latency_s"] for r in rs]),
                "prompt_tokens": _mean([total(r, "prompt_tokens") for r in rs]),
                "completion_tokens": _mean([total(r, "completion_tokens") for r in rs])}

    similar = [r for r in records if r.get("similarity") is not None]
    control = [r for r in similar if r.get("similarity_control") is not None]
    return {
        "all": per_question(records),
        "by_route": {k: per_question(v) for k, v in routes.items()},
        "by_role": {k: {"calls": len(v),
                        "prompt_tokens": _mean([c.get("prompt_tokens") for c in v]),
                        "max_prompt_tokens": max((c["prompt_tokens"] for c in v if c.get("prompt_tokens") is not None),
                                                 default=None),
                        "completion_tokens": _mean([c.get("completion_tokens") for c in v]),
                        "prompt_eval_ms": percentiles([c.get("prompt_eval_ms") for c in v]),
                        "latency_ms": percentiles([c.get("latency_ms") for c in v])}
                    for k, v in roles.items()},
        "similarity": {"n": len(similar), "mean": _mean([r["similarity"] for r in similar]),
                       **percentiles([r["similarity"] for r in similar]),
                       "control_n": len(control),
                       "control_mean": _mean([r["similarity_control"] for r in control]),
                       "above_control": sum(r["similarity"] > r["similarity_control"] for r in control)},
    }


def decision_stability(runs: list[list[dict]]) -> dict:
    """Cases present in every run whose route and final status are the same in all of them."""
    seen: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for run in runs:
        for r in run:
            seen.setdefault((r["group"], r["id"]), []).append((r["route"], r["status"]))
    complete = {k: v for k, v in seen.items() if len(v) == len(runs)}
    return {"runs": len(runs), "n": len(complete),
            "same_route": sum(len({route for route, _ in v}) == 1 for v in complete.values()),
            "same_decision": sum(len(set(v)) == 1 for v in complete.values()),
            "changed": [k[1] for k, v in complete.items() if len(set(v)) > 1]}
