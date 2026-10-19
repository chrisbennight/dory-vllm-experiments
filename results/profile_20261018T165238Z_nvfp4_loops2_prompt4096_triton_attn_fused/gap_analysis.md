# Gap to bandwidth minimum: nvfp4, 2 loops, batch size 1, 4096-token prompt (profile_20261018T165238Z_nvfp4_loops2_prompt4096_triton_attn_fused)

- Measured: 6.07 ms per token (165 tokens/s)
- Bandwidth minimum: 3.95 ms per token (253 tokens/s) at 1008 GB/s
- KV cache read per token: 413 MB over 56 attention layer applications

| Component | Measured ms | Minimum ms | Gap ms | Share of total gap | Efficiency |
|---|---:|---:|---:|---:|---:|
| Weight GEMMs | 3.692 | 2.867 | 0.825 | 39% | 78% |
| Attention | 1.214 | 0.410 | 0.805 | 38% | 34% |
| Normalization and gating | 0.305 | 0.010 | 0.295 | 14% | 3% |
| Sampling, bookkeeping, copies | 0.101 | 0.000 | 0.101 | 5% | n/a |
| Idle GPU | 0.053 | 0.000 | 0.053 | 3% | n/a |
| Output projection | 0.702 | 0.666 | 0.036 | 2% | 95% |

| Weight projection | Calls per token | Mean us per call | Minimum us per call | Gap ms per token | Efficiency |
|---|---:|---:|---:|---:|---:|
| q/k/v + output gate | 56 | 17.8 | 14.6 | 0.177 | 82% |
| attention output | 56 | 9.7 | 5.9 | 0.213 | 61% |
| MLP gate+up | 56 | 25.3 | 20.5 | 0.272 | 81% |
| MLP down | 56 | 13.1 | 10.2 | 0.163 | 78% |
