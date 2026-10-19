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
| vLLM (16 segments) + gate | 10.57 | 15.82 | 36.02 |
| Fused 32 x 1 x 1 | 7.00 | 9.96 | 16.54 |
| Fused 32 x 1 x 2 | 6.88 | 9.74 | 16.36 |
| Fused 32 x 1 x 3 | 6.88 | 9.77 | 16.38 |
| Fused 32 x 2 x 1 | 6.92 | 10.11 | 16.59 |
| Fused 32 x 2 x 2 | 7.12 | 9.93 | 16.54 |
| Fused 32 x 2 x 3 | 7.13 | 10.04 | 16.58 |
| Fused 64 x 1 x 1 | 7.02 | 9.79 | 16.35 |
| Fused 64 x 1 x 2 | 7.11 | 9.98 | 16.54 |
| Fused 64 x 1 x 3 | 7.22 | 9.95 | 16.62 |
| Fused 64 x 2 x 1 | 7.14 | 10.07 | 16.67 |
| Fused 64 x 2 x 2 | 7.32 | 10.15 | 16.87 |
| Fused 64 x 2 x 3 | 7.36 | 10.26 | 17.05 |
| Fused 128 x 1 x 1 | 7.56 | 10.14 | 16.85 |
| Fused 128 x 1 x 2 | 7.69 | 10.44 | 17.76 |
| Fused 128 x 1 x 3 | 7.72 | 10.41 | 17.72 |
| Fused 128 x 2 x 1 | 7.85 | 10.51 | 17.17 |
| Fused 128 x 2 x 2 | 7.70 | 10.53 | 17.23 |
| Fused 128 x 2 x 3 | 7.75 | 10.57 | 17.25 |

Microseconds per attention layer application; 56 per token.

## Kernel time per application at 640 cached tokens (profiler, us)

| Implementation | Kernel | us |
|---|---|---:|
| vLLM | `kernel_unified_attention` | 6.82 |
| vLLM | `void at::native::unrolled_elementwise_kernel<at::native::...` | 2.11 |
| vLLM | `reduce_segments` | 1.78 |
| Fused 32 x 1 x 2 | `_decode_attention_kernel` | 5.35 |
| Fused 32 x 1 x 2 | `_combine_gate_kernel` | 1.59 |

## Raw-input mode: q/k normalization and cache write inside the kernel

| Cached tokens | Layer | New k and v cached exactly | Relative error |
|---:|---|---|---:|
| 640 | full | True | 1.19e-03 |
| 640 | sliding | True | 1.11e-03 |
| 2048 | full | True | 1.20e-03 |
| 2048 | sliding | True | 1.17e-03 |
| 8192 | full | True | 1.13e-03 |
| 8192 | sliding | True | 1.13e-03 |

Time per attention layer application at 640 cached tokens (CUDA graph): q/k normalization kernel + attention 8.13 us; raw-input mode 7.73 us.
