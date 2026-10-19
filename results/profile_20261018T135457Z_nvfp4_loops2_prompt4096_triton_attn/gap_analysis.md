# Gap to bandwidth minimum: nvfp4, 2 loops, batch size 1, 4096-token prompt (profile_20261018T135457Z_nvfp4_loops2_prompt4096_triton_attn)

- Measured: 6.98 ms per token (143 tokens/s)
- Bandwidth minimum: 3.95 ms per token (253 tokens/s) at 1008 GB/s
- KV cache read per token: 413 MB over 56 attention layer applications

| Component | Measured ms | Minimum ms | Gap ms | Share of total gap | Efficiency |
|---|---:|---:|---:|---:|---:|
| Weight GEMMs | 4.088 | 2.867 | 1.221 | 40% | 70% |
| Normalization and gating | 0.817 | 0.010 | 0.807 | 27% | 1% |
| Attention | 1.212 | 0.410 | 0.803 | 27% | 34% |
| Sampling, bookkeeping, copies | 0.099 | 0.000 | 0.099 | 3% | n/a |
| Idle GPU | 0.063 | 0.000 | 0.063 | 2% | n/a |
| Output projection | 0.702 | 0.666 | 0.036 | 1% | 95% |

| Weight projection | Calls per token | Mean us per call | Minimum us per call | Gap ms per token | Efficiency |
|---|---:|---:|---:|---:|---:|
| q/k/v | 56 | 13.0 | 8.8 | 0.237 | 67% |
| output gate | 56 | 9.6 | 5.9 | 0.210 | 61% |
| attention output | 56 | 10.1 | 5.9 | 0.238 | 58% |
| MLP gate+up | 56 | 25.8 | 20.5 | 0.295 | 80% |
| MLP down | 56 | 14.6 | 10.2 | 0.242 | 70% |
