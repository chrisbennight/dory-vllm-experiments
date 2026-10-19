# Greedy output agreement with nvfp4-triton (64 prompts, 256 tokens, 2 loops)

| Configuration | Identical outputs | Mean shared tokens | Fewest shared tokens | Mean abs logprob difference | Largest abs logprob difference | Divergences | Median divergence margin | Largest divergence margin | Divergent token outside top 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-flashinfer | 16/64 | 121 | 2 | 0.0069 | 0.1855 | 48 | 0.069 | 0.247 | 0 |
| nvfp4-triton-fused | 14/64 | 128 | 2 | 0.0066 | 0.1879 | 50 | 0.059 | 0.247 | 0 |
