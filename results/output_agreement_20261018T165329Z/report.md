# Greedy output agreement with nvfp4-triton (16 prompts, 256 tokens, 2 loops)

| Configuration | Identical outputs | Mean shared tokens | Fewest shared tokens | Mean abs logprob difference | Largest abs logprob difference | Divergences | Median divergence margin | Largest divergence margin | Divergent token outside top 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-flashinfer | 4/16 | 111 | 33 | 0.0059 | 0.1344 | 12 | 0.088 | 0.247 | 0 |
| nvfp4-triton-fused | 3/16 | 139 | 41 | 0.0058 | 0.0947 | 13 | 0.081 | 0.222 | 0 |
