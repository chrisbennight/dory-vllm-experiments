# Multi-token Dory sublayers (NVIDIA GeForce RTX 4090, 640-token context, 4 tokens in their own caches)

## Accuracy against the single-token sublayers run once per token

Relative difference per token (BF16 outputs).

| Sublayer | Token | Output | Cached key | Cached value |
|---|---:|---:|---:|---:|
| attention layer 16 (sliding) | 0 | 4.68e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 16 (sliding) | 1 | 5.38e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 16 (sliding) | 2 | 7.16e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 16 (sliding) | 3 | 5.77e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 22 (full) | 0 | 7.22e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 22 (full) | 1 | 8.66e-04 | 0.00e+00 | 0.00e+00 |
| attention layer 22 (full) | 2 | 9.13e-04 | 1.53e-05 | 0.00e+00 |
| attention layer 22 (full) | 3 | 7.57e-04 | 0.00e+00 | 0.00e+00 |
| MLP layer 17 | 0 | 7.61e-04 | | |
| MLP layer 17 | 1 | 5.53e-04 | | |
| MLP layer 17 | 2 | 5.12e-04 | | |
| MLP layer 17 | 3 | 7.54e-04 | | |

## Time per attention+MLP layer pair (20 recurrent pairs, CUDA graph)

| Path | Tokens | us per pair | Relative to single-token, 1 token |
|---|---:|---:|---:|
| single-token sublayers | 1 | 75.3 | 1.00 |
| multi-token sublayers | 1 | 91.9 | 1.22 |
| multi-token sublayers | 2 | 94.5 | 1.25 |
| multi-token sublayers | 4 | 101.5 | 1.35 |
