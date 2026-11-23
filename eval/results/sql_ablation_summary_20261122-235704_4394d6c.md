# SQL ablation 2026-11-22

Model `qwen2.5-coder:14b-8k`, commit `4394d6c`, temperature 0, 40 answerable questions.

| variant | n | strict | lenient | strict_easy | strict_medium | strict_hard | lat_mean_s | lat_p50_s | lat_p95_s | llm_lat_mean_s | prompt_tok_mean | completion_tok_mean | retries_mean | failed | declined | would_reject_grain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V3 | 40 | 20 | 22 | 5/6 | 8/15 | 7/19 | 2.78 | 1.61 | 7.21 | 2.75 | 9011 | 100 | 0.2 | 1 | 0 | 0 |

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
| q08 | hard | - |  |
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
| v09 | hard | L |  |
| v10 | medium | S |  |
| v11 | hard | S |  |
| v12 | hard | S |  |
| v13 | medium | S |  |
| v14 | medium | S |  |
| v15 | hard | - |  |
| v16 | hard | - |  |
| v17 | hard | - |  |
| v18 | medium | S |  |
| v19 | hard | - |  |
| v20 | hard | S |  |
| v21 | medium | L |  |
| v22 | medium | - |  |
| v23 | medium | - |  |
