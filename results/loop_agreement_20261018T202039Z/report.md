# Agreement of shallow and deep Dory predictions on the same histories

64 chat prompts (bench/compare_outputs.py), 2-loop greedy completions of 256 tokens with thinking on; every completion position scored by 1-, 2- and 4-loop models (prompt log-probabilities, top 20).

Sanity check: the 2-loop model's top token equals the 2-loop greedy history at 99.3% of positions (prefill and decode use different kernels, so near-ties can flip).

| Draft loops | Verify loops | Top-token agreement | Mean distribution overlap (lower bound) | Mean accepted, 1-token greedy draft | Mean accepted, 2-token greedy draft | Mean accepted, 3-token greedy draft | Mean accepted, 4-token greedy draft |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 94.2% | 0.944 | 0.94 | 1.83 | 2.66 | 3.45 |
| 1 | 4 | 93.5% | 0.936 | 0.93 | 1.81 | 2.62 | 3.38 |
| 2 | 4 | 97.9% | 0.976 | 0.98 | 1.93 | 2.87 | 3.77 |

