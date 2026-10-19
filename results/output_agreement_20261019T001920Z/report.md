# Greedy output agreement with nvfp4-triton (64 prompts, 256 tokens, 4 loops)

| Configuration | Identical outputs | Mean shared tokens | Fewest shared tokens | Mean abs logprob difference | Largest abs logprob difference | Divergences | Median divergence margin | Largest divergence margin | Divergent token outside top 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-triton-fused-wavefront | 9/64 | 120 | 23 | 0.0074 | 0.2019 | 55 | 0.028 | 0.172 | 0 |
| nvfp4-triton-fused-int8-head | 9/64 | 121 | 22 | 0.0072 | 0.2058 | 55 | 0.031 | 0.248 | 0 |
| nvfp4-flashinfer | 17/64 | 133 | 4 | 0.0076 | 0.1503 | 47 | 0.056 | 0.179 | 0 |
