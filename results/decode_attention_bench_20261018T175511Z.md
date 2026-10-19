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

Each configuration is splits x cache blocks per tile x pipeline stages (4 warps).

| Implementation | 640 tokens | 2048 tokens | 8192 tokens |
|---|---:|---:|---:|
| vLLM (16 segments) + gate | 10.59 | 15.81 | 36.11 |
| Fused 32 x 1 x 1 | 7.04 | 10.02 | 16.54 |
| Fused 32 x 1 x 2 | 6.77 | 9.62 | 16.23 |
| Fused 32 x 1 x 3 | 6.84 | 9.73 | 16.33 |
| Fused 32 x 2 x 1 | 7.05 | 10.08 | 16.55 |
| Fused 32 x 2 x 2 | 7.09 | 9.93 | 16.48 |
| Fused 32 x 2 x 3 | 7.09 | 9.97 | 16.54 |
| Fused 64 x 1 x 1 | 7.14 | 9.91 | 16.46 |
| Fused 64 x 1 x 2 | 7.13 | 9.92 | 16.53 |
| Fused 64 x 1 x 3 | 7.24 | 10.01 | 16.66 |
| Fused 64 x 2 x 1 | 7.30 | 10.22 | 16.83 |
| Fused 64 x 2 x 2 | 7.33 | 10.21 | 16.90 |
| Fused 64 x 2 x 3 | 7.34 | 10.21 | 16.95 |
| Fused 128 x 1 x 1 | 7.60 | 10.18 | 16.88 |
| Fused 128 x 1 x 2 | 7.58 | 10.22 | 17.49 |
| Fused 128 x 1 x 3 | 7.66 | 10.31 | 17.61 |
| Fused 128 x 2 x 1 | 7.92 | 10.59 | 17.20 |
| Fused 128 x 2 x 2 | 7.68 | 10.50 | 17.15 |
| Fused 128 x 2 x 3 | 7.62 | 10.43 | 17.04 |

Microseconds per attention layer application; 56 per token.

## Kernel time per application at 640 cached tokens (profiler, us)

| Implementation | Kernel | us |
|---|---|---:|
| vLLM | `kernel_unified_attention` | 6.83 |
| vLLM | `void at::native::unrolled_elementwise_kernel<at::native::...` | 2.12 |
| vLLM | `reduce_segments` | 1.78 |
| Fused 32 x 1 x 2 | `_decode_attention_kernel` | 5.24 |
| Fused 32 x 1 x 2 | `_combine_gate_kernel` | 1.58 |
