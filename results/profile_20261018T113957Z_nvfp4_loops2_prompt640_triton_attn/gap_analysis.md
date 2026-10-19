# Gap to bandwidth minimum: nvfp4, 2 loops, batch size 1, 640-token prompt (profile_20261018T113957Z_nvfp4_loops2_prompt640_triton_attn)

- Measured: 6.32 ms per token (158 tokens/s)
- Bandwidth minimum: 3.70 ms per token (271 tokens/s) at 1008 GB/s
- KV cache read per token: 154 MB over 56 attention layer applications

| Component | Measured ms | Minimum ms | Gap ms | Share of total gap | Efficiency |
|---|---:|---:|---:|---:|---:|
| Weight GEMMs | 4.089 | 2.867 | 1.222 | 47% | 70% |
| Normalization and gating | 0.817 | 0.010 | 0.807 | 31% | 1% |
| Attention | 0.553 | 0.153 | 0.400 | 15% | 28% |
| Sampling, bookkeeping, copies | 0.098 | 0.000 | 0.098 | 4% | n/a |
| Idle GPU | 0.063 | 0.000 | 0.063 | 2% | n/a |
| Output projection | 0.702 | 0.666 | 0.036 | 1% | 95% |

| Weight projection | Calls per token | Mean us per call | Minimum us per call | Gap ms per token | Efficiency |
|---|---:|---:|---:|---:|---:|
| q/k/v | 56 | 13.0 | 8.8 | 0.236 | 68% |
| output gate | 56 | 9.6 | 5.9 | 0.210 | 61% |
| attention output | 56 | 10.1 | 5.9 | 0.238 | 58% |
| MLP gate+up | 56 | 25.8 | 20.5 | 0.298 | 79% |
| MLP down | 56 | 14.5 | 10.2 | 0.240 | 71% |
