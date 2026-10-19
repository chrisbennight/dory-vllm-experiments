# Multi-token Dory sublayers (NVIDIA GeForce RTX 4090, 640-token context, 4 tokens in their own caches)

## Accuracy against the single-token sublayers run once per token

Relative difference per token (BF16 outputs).

| Sublayer | Token | Output | Cached key | Cached value |
|---|---:|---:|---:|---:|
| attention layer 16 (sliding) | 0 | 5.18e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 16 (sliding) | 1 | 5.59e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 16 (sliding) | 2 | 4.61e-04 | 3.15e-05 | 1.10e-09 |
| attention layer 16 (sliding) | 3 | 4.54e-04 | 0.00e+00 | 3.05e-06 |
| attention layer 22 (full) | 0 | 9.38e-04 | 4.77e-07 | 0.00e+00 |
| attention layer 22 (full) | 1 | 8.04e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 22 (full) | 2 | 7.89e-04 | 3.81e-06 | 0.00e+00 |
| attention layer 22 (full) | 3 | 6.84e-04 | 0.00e+00 | 0.00e+00 |
| MLP layer 17 | 0 | 6.17e-04 | | |
| MLP layer 17 | 1 | 6.15e-04 | | |
| MLP layer 17 | 2 | 5.46e-04 | | |
| MLP layer 17 | 3 | 5.66e-04 | | |

## Time per attention+MLP layer pair (20 recurrent pairs, CUDA graph)

| Path | Tokens | us per pair | Relative to single-token, 1 token |
|---|---:|---:|---:|
| single-token sublayers | 1 | 75.0 | 1.00 |
| multi-token sublayers | 1 | 94.0 | 1.25 |
| multi-token sublayers | 2 | 90.7 | 1.21 |
| multi-token sublayers | 4 | 97.9 | 1.31 |
