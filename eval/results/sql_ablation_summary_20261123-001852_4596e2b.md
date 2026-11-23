# SQL ablation 2026-11-23

Model `qwen2.5-coder:14b-8k`, commit `4596e2b`, temperature 0, 40 answerable questions.

| variant | n | strict | lenient | strict_easy | strict_medium | strict_hard | lat_mean_s | lat_p50_s | lat_p95_s | llm_lat_mean_s | prompt_tok_mean | completion_tok_mean | retries_mean | failed | declined | would_reject_grain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V3 | 40 | 23 | 24 | 5/6 | 8/15 | 10/19 | 2.94 | 1.55 | 7.99 | 2.91 | 9404 | 113 | 0.25 | 0 | 0 | 0 |

## Per question (S = strict, L = lenient only, - = miss)

| id | difficulty | V3 | changes |
|---|---|---|---|
| q01 | easy | S |  |
| q02 | easy | S |  |
| q03 | easy | S |  |
| q04 | easy | - |  |
| q04b | easy | S |  |
| q05 | medium | - |  |
| q06 | hard | S |  |
| q07 | hard | - |  |
| q08 | hard | S |  |
| q09 | medium | S |  |
| q10 | easy | S |  |
| q11 | hard | - |  |
| q12 | hard | - |  |
| q13 | medium | S |  |
| q14 | hard | - |  |
| q15 | hard | - |  |
| q16 | medium | - |  |
| v01 | medium | S |  |
| v02 | medium | S |  |
| v03 | hard | S |  |
| v04 | hard | S |  |
| v05 | hard | S |  |
| v06 | hard | - |  |
| v07 | medium | - |  |
| v08 | medium | - |  |
| v09 | hard | S |  |
| v10 | medium | S |  |
| v11 | hard | S |  |
| v12 | hard | S |  |
| v13 | medium | S |  |
| v14 | medium | S |  |
| v15 | hard | - |  |
| v16 | hard | - |  |
| v17 | hard | S |  |
| v18 | medium | S |  |
| v19 | hard | - |  |
| v20 | hard | S |  |
| v21 | medium | L |  |
| v22 | medium | - |  |
| v23 | medium | - |  |
