# Gap to bandwidth minimum: nvfp4, 2 loops, batch size 1, 7936-token prompt (profile_20261018T165258Z_nvfp4_loops2_prompt7936_triton_attn_fused)

- Measured: 6.71 ms per token (149 tokens/s)
- Bandwidth minimum: 4.17 ms per token (240 tokens/s) at 1008 GB/s
- KV cache read per token: 633 MB over 56 attention layer applications

| Component | Measured ms | Minimum ms | Gap ms | Share of total gap | Efficiency |
|---|---:|---:|---:|---:|---:|
| Attention | 1.859 | 0.628 | 1.231 | 48% | 34% |
| Weight GEMMs | 3.690 | 2.867 | 0.823 | 32% | 78% |
| Normalization and gating | 0.304 | 0.010 | 0.294 | 12% | 3% |
| Sampling, bookkeeping, copies | 0.103 | 0.000 | 0.103 | 4% | n/a |
| Idle GPU | 0.053 | 0.000 | 0.053 | 2% | n/a |
| Output projection | 0.702 | 0.666 | 0.036 | 1% | 95% |

| Weight projection | Calls per token | Mean us per call | Minimum us per call | Gap ms per token | Efficiency |
|---|---:|---:|---:|---:|---:|
| q/k/v + output gate | 56 | 17.8 | 14.6 | 0.178 | 82% |
| attention output | 56 | 9.6 | 5.9 | 0.212 | 61% |
| MLP gate+up | 56 | 25.3 | 20.5 | 0.269 | 81% |
| MLP down | 56 | 13.2 | 10.2 | 0.164 | 78% |
