# Gap to bandwidth minimum: nvfp4, 2 loops, batch size 1, 7936-token prompt (profile_20261018T135516Z_nvfp4_loops2_prompt7936_triton_attn)

- Measured: 7.64 ms per token (131 tokens/s)
- Bandwidth minimum: 4.17 ms per token (240 tokens/s) at 1008 GB/s
- KV cache read per token: 633 MB over 56 attention layer applications

| Component | Measured ms | Minimum ms | Gap ms | Share of total gap | Efficiency |
|---|---:|---:|---:|---:|---:|
| Attention | 1.870 | 0.628 | 1.242 | 36% | 34% |
| Weight GEMMs | 4.091 | 2.867 | 1.224 | 35% | 70% |
| Normalization and gating | 0.817 | 0.010 | 0.807 | 23% | 1% |
| Sampling, bookkeeping, copies | 0.100 | 0.000 | 0.100 | 3% | n/a |
| Idle GPU | 0.063 | 0.000 | 0.063 | 2% | n/a |
| Output projection | 0.702 | 0.666 | 0.036 | 1% | 95% |

| Weight projection | Calls per token | Mean us per call | Minimum us per call | Gap ms per token | Efficiency |
|---|---:|---:|---:|---:|---:|
| q/k/v | 56 | 13.0 | 8.8 | 0.235 | 68% |
| output gate | 56 | 9.6 | 5.9 | 0.209 | 61% |
| attention output | 56 | 10.1 | 5.9 | 0.238 | 58% |
| MLP gate+up | 56 | 25.8 | 20.5 | 0.300 | 79% |
| MLP down | 56 | 14.6 | 10.2 | 0.242 | 70% |
