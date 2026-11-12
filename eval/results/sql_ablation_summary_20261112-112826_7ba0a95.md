# SQL ablation 2026-11-12

Model `qwen2.5-coder:14b-8k`, commit `7ba0a95`, temperature 0, 40 answerable questions.

| variant | n | strict | lenient | strict_easy | strict_medium | strict_hard | lat_mean_s | lat_p50_s | lat_p95_s | llm_lat_mean_s | prompt_tok_mean | completion_tok_mean | retries_mean | failed | declined | would_reject_grain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 40 | 2 | 3 | 0/6 | 2/15 | 0/19 | 1.16 | 1.04 | 1.73 | 1.15 | 800 | 52 | 0.0 | 1 | 9 | 10 |
| V1 | 40 | 12 | 13 | 5/6 | 5/15 | 2/19 | 1.34 | 1.23 | 2.23 | 1.33 | 6631 | 51 | 0.0 | 1 | 15 | 6 |
| V2 | 40 | 15 | 16 | 5/6 | 6/15 | 4/19 | 1.58 | 1.34 | 4.01 | 1.56 | 6797 | 58 | 0.0 | 0 | 10 | 7 |
| V3 | 40 | 21 | 23 | 5/6 | 6/15 | 10/19 | 2.47 | 1.34 | 6.6 | 2.45 | 8212 | 95 | 0.2 | 0 | 10 | 0 |

## Per question (S = strict, L = lenient only, - = miss)

| id | difficulty | V0 | V1 | V2 | V3 | changes |
|---|---|---|---|---|---|---|
| q01 | easy | - | S | S | S | - → S → S → S |
| q02 | easy | - | S | S | S | - → S → S → S |
| q03 | easy | - | S | S | S | - → S → S → S |
| q04 | easy | - | S | - | - | - → S → - → - |
| q04b | easy | - | S | S | S | - → S → S → S |
| q05 | medium | - | - | - | - |  |
| q06 | hard | - | - | - | S | - → - → - → S |
| q07 | hard | - | - | - | S | - → - → - → S |
| q08 | hard | - | - | - | S | - → - → - → S |
| q09 | medium | S | S | S | S |  |
| q10 | easy | - | - | S | S | - → - → S → S |
| q11 | hard | - | - | - | L | - → - → - → L |
| q12 | hard | - | - | - | - |  |
| q13 | medium | S | S | S | S |  |
| q14 | hard | - | - | - | - |  |
| q15 | hard | - | - | - | - |  |
| q16 | medium | - | - | - | - |  |
| v01 | medium | - | S | S | S | - → S → S → S |
| v02 | medium | - | - | S | S | - → - → S → S |
| v03 | hard | - | S | S | S | - → S → S → S |
| v04 | hard | - | S | S | S | - → S → S → S |
| v05 | hard | - | - | - | - |  |
| v06 | hard | - | - | - | - |  |
| v07 | medium | - | - | - | - |  |
| v08 | medium | - | - | - | - |  |
| v09 | hard | - | - | - | S | - → - → - → S |
| v10 | medium | - | - | S | S | - → - → S → S |
| v11 | hard | - | - | S | S | - → - → S → S |
| v12 | hard | - | - | S | S | - → - → S → S |
| v13 | medium | - | S | - | - | - → S → - → - |
| v14 | medium | - | S | S | S | - → S → S → S |
| v15 | hard | - | - | - | - |  |
| v16 | hard | - | - | - | - |  |
| v17 | hard | - | - | - | S | - → - → - → S |
| v18 | medium | - | - | - | - |  |
| v19 | hard | - | - | - | - |  |
| v20 | hard | - | - | - | S | - → - → - → S |
| v21 | medium | L | L | L | L |  |
| v22 | medium | - | - | - | - |  |
| v23 | medium | - | - | - | - |  |
