# Gap to bandwidth minimum: nvfp4, 2 loops, batch size 1, 640-token prompt (profile_20261018T165143Z_nvfp4_loops2_prompt640_triton_attn_fused)

- Measured: 5.41 ms per token (185 tokens/s)
- Bandwidth minimum: 3.70 ms per token (271 tokens/s) at 1008 GB/s
- KV cache read per token: 154 MB over 56 attention layer applications

| Component | Measured ms | Minimum ms | Gap ms | Share of total gap | Efficiency |
|---|---:|---:|---:|---:|---:|
| Weight GEMMs | 3.692 | 2.867 | 0.825 | 48% | 78% |
| Attention | 0.551 | 0.153 | 0.398 | 23% | 28% |
| Normalization and gating | 0.306 | 0.010 | 0.296 | 17% | 3% |
| Sampling, bookkeeping, copies | 0.103 | 0.000 | 0.103 | 6% | n/a |
| Idle GPU | 0.053 | 0.000 | 0.053 | 3% | n/a |
| Output projection | 0.702 | 0.666 | 0.036 | 2% | 95% |

| Weight projection | Calls per token | Mean us per call | Minimum us per call | Gap ms per token | Efficiency |
|---|---:|---:|---:|---:|---:|
| q/k/v + output gate | 56 | 17.8 | 14.6 | 0.177 | 82% |
| attention output | 56 | 9.6 | 5.9 | 0.212 | 61% |
| MLP gate+up | 56 | 25.3 | 20.5 | 0.271 | 81% |
| MLP down | 56 | 13.2 | 10.2 | 0.165 | 78% |
