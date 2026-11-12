| run | n | strict | lenient | e2e mean s | p50 | p95 | prompt tokens sent | prompt ms total | decode ms total |
|---|---|---|---|---|---|---|---|---|---|
| full_cache_friendly | 40 | 21 | 23 | 6.77 | 5.40 | 12.39 | 553698 | 109669 | 146720 |
| full_cache_friendly_router | 40 | 18 | 20 | 4.37 | 2.79 | 9.12 | 805354 | 14409 | 145694 |
| full_legacy | 40 | 21 | 23 | 7.06 | 5.84 | 13.93 | 349191 | 108687 | 158877 |
| mixed_cache_friendly | 24 | 9 | 10 | 5.69 | 4.70 | 10.21 | 216650 | 39532 | 89138 |
| mixed_legacy | 24 | 9 | 10 | 5.85 | 5.42 | 13.21 | 137486 | 39174 | 93177 |
| sql_busted_legacy | 40 | 20 | 22 | 4.62 | 3.72 | 8.83 | 328885 | 103528 | 74414 |
| sql_legacy | 40 | 21 | 23 | 2.44 | 1.35 | 6.34 | 328485 | 4834 | 85542 |
| sql_legacy_cap320 | 40 | 21 | 23 | 2.35 | 1.43 | 6.73 | 321052 | 4821 | 82029 |

| run | agent | calls | prompt tok (incl. cached) | prompt_eval ms | completion tok | decode ms | call latency ms | prompt ms first / retry (n retry) |
|---|---|---|---|---|---|---|---|---|
| full_cache_friendly | router | 40 | 194.8 | 75.5 | 13.3 | 243.5 | 393.6 | 75.5 / None (0) |
| full_cache_friendly | sql | 49 | 6851.4 | 2113.8 | 80.9 | 1803.3 | 4051.6 | 2570.4 / 84.5 (9) |
| full_cache_friendly | composer | 31 | 6780.2 | 99.1 | 70.7 | 1568.3 | 1793.1 | 99.1 / None (0) |
| full_cache_friendly_router | router | 40 | 6791.8 | 161.0 | 11.5 | 237.8 | 487.6 | 161.0 / None (0) |
| full_cache_friendly_router | sql | 49 | 6878.1 | 103.6 | 86.9 | 1939.3 | 2184.1 | 107.7 / 87.8 (10) |
| full_cache_friendly_router | composer | 29 | 6781.2 | 99.6 | 64.1 | 1419.3 | 1639.6 | 99.6 / None (0) |
| full_legacy | router | 40 | 194.8 | 74.2 | 13.4 | 245.0 | 393.6 | 74.2 / None (0) |
| full_legacy | sql | 49 | 6851.4 | 2111.5 | 80.9 | 1800.9 | 4046.6 | 2567.6 / 84.1 (9) |
| full_legacy | composer | 31 | 183.2 | 72.9 | 100.7 | 1962.4 | 2173.3 | 72.9 / None (0) |
| mixed_cache_friendly | router | 24 | 193.4 | 74.6 | 13.8 | 256.6 | 406.2 | 74.6 / None (0) |
| mixed_cache_friendly | sql | 17 | 6835.9 | 1852.6 | 79.6 | 1804.5 | 3797.5 | 2587.6 / 88.5 (5) |
| mixed_cache_friendly | composer | 12 | 6797.6 | 108.1 | 95.2 | 2159.9 | 2410.8 | 108.1 / None (0) |
| mixed_cache_friendly | docs | 12 | 1185.6 | 412.6 | 108.3 | 2198.7 | 2749.2 | 412.6 / None (0) |
| mixed_legacy | router | 24 | 193.4 | 73.8 | 13.8 | 256.2 | 404.8 | 73.8 / None (0) |
| mixed_legacy | sql | 17 | 6835.9 | 1853.1 | 79.6 | 1805.0 | 3797.8 | 2588.4 / 88.4 (5) |
| mixed_legacy | composer | 12 | 200.6 | 79.6 | 125.3 | 2498.0 | 2734.2 | 79.6 / None (0) |
| mixed_legacy | docs | 12 | 1185.6 | 412.1 | 108.3 | 2197.3 | 2746.2 | 412.1 / None (0) |
| sql_busted_legacy | sql | 48 | 6851.8 | 2156.8 | 69.8 | 1550.3 | 3834.8 | 2570.7 / 87.3 (8) |
| sql_legacy | sql | 48 | 6843.4 | 100.7 | 79.5 | 1782.1 | 2018.2 | 103.9 / 84.6 (8) |
| sql_legacy_cap320 | sql | 47 | 6830.9 | 102.6 | 77.1 | 1745.3 | 1982.3 | 105.0 / 88.6 (7) |
