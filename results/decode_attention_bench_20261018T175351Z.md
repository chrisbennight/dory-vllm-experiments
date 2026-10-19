# Gated decode attention (NVIDIA GeForce RTX 4090, 2 loops, 56 attention layer applications per token, 16 query and 4 KV heads of 256, window 1024)

## Accuracy against FP32 attention

Relative error of the gated attention output; the fused kernel uses 32 splits.

| Cached tokens | Layer | vLLM | Fused | Fused vs vLLM |
|---:|---|---:|---:|---:|
| 640 | full | 2.21e-03 | 1.45e-03 | 2.04e-03 |
| 640 | sliding | 2.25e-03 | 1.45e-03 | 2.13e-03 |
| 2048 | full | 2.30e-03 | 1.49e-03 | 2.23e-03 |
| 2048 | sliding | 2.26e-03 | 1.51e-03 | 2.37e-03 |
| 8192 | full | 2.39e-03 | 1.62e-03 | 2.38e-03 |
| 8192 | sliding | 2.34e-03 | 1.51e-03 | 2.60e-03 |

## Time per token for all attention layer applications (CUDA graph, us)

Each configuration is splits x cache blocks per tile x warps.

| Implementation | 640 tokens | 2048 tokens | 8192 tokens |
|---|---:|---:|---:|
| vLLM (16 segments) + gate | 10.60 | 15.81 | 36.04 |
| Fused 16 x 1 x 4 | 7.27 | 10.80 | 17.85 |
| Fused 16 x 1 x 8 | 8.13 | 12.43 | 20.58 |
| Fused 16 x 2 x 4 | 7.55 | 10.39 | 16.91 |
| Fused 16 x 2 x 8 | 8.14 | 11.35 | 17.86 |
| Fused 16 x 4 x 4 | 7.60 | 11.62 | 18.12 |
| Fused 16 x 4 x 8 | 7.77 | 11.93 | 18.44 |
| Fused 32 x 1 x 4 | 7.15 | 10.01 | 16.56 |
| Fused 32 x 1 x 8 | 7.73 | 10.65 | 17.19 |
| Fused 32 x 2 x 4 | 7.42 | 10.30 | 16.80 |
| Fused 32 x 2 x 8 | 7.69 | 10.90 | 17.38 |
| Fused 32 x 4 x 4 | 8.07 | 11.04 | 17.47 |
| Fused 32 x 4 x 8 | 8.21 | 11.24 | 17.73 |
| Fused 64 x 1 x 4 | 8.34 | 11.12 | 17.74 |
| Fused 64 x 1 x 8 | 8.62 | 11.46 | 18.13 |
| Fused 64 x 2 x 4 | 8.61 | 11.33 | 17.94 |
| Fused 64 x 2 x 8 | 8.90 | 11.61 | 18.35 |
| Fused 64 x 4 x 4 | 9.25 | 12.08 | 18.57 |
| Fused 64 x 4 x 8 | 9.45 | 12.26 | 18.74 |

Microseconds per attention layer application; 56 per token.

## Kernel time per application at 640 cached tokens (profiler, us)

| Implementation | Kernel | us |
|---|---|---:|
| vLLM | `kernel_unified_attention` | 6.85 |
| vLLM | `void at::native::unrolled_elementwise_kernel<at::native::...` | 2.10 |
| vLLM | `reduce_segments` | 1.78 |
| Fused 32 x 1 x 4 | `_decode_attention_kernel` | 5.20 |
| Fused 32 x 1 x 4 | `_combine_gate_kernel` | 2.01 |
