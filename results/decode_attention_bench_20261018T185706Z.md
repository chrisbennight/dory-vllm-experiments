# Gated decode attention (NVIDIA GeForce RTX 4090, 2 loops, 56 attention layer applications per token, 16 query and 4 KV heads of 256, window 1024)

## Accuracy against FP32 attention

Relative error of the gated attention output; the fused kernel uses 32 splits.

| Cached tokens | Layer | vLLM | Fused | Fused vs vLLM |
|---:|---|---:|---:|---:|
| 640 | full | 2.21e-03 | 1.40e-03 | 2.21e-03 |
| 640 | sliding | 2.25e-03 | 1.40e-03 | 2.37e-03 |
| 2048 | full | 2.30e-03 | 1.53e-03 | 2.54e-03 |
| 2048 | sliding | 2.26e-03 | 1.45e-03 | 2.51e-03 |
| 8192 | full | 2.39e-03 | 1.64e-03 | 2.91e-03 |
| 8192 | sliding | 2.34e-03 | 1.50e-03 | 2.68e-03 |

## Time per token for all attention layer applications (CUDA graph, us)

Each configuration is splits x cache blocks per tile x pipeline stages (4 warps).

| Implementation | 640 tokens | 2048 tokens | 8192 tokens |
|---|---:|---:|---:|
| vLLM (16 segments) + gate | 10.58 | 15.82 | 36.07 |
| Fused 32 x 1 x 1 | 6.67 | 9.83 | 16.37 |
| Fused 32 x 1 x 2 | 6.68 | 9.76 | 16.63 |
| Fused 32 x 2 x 1 | 6.53 | 9.89 | 16.41 |
| Fused 32 x 2 x 2 | 6.64 | 9.88 | 16.51 |
| Fused 64 x 1 x 1 | 6.54 | 9.76 | 16.28 |
| Fused 64 x 1 x 2 | 6.58 | 9.82 | 16.31 |
| Fused 64 x 2 x 1 | 6.69 | 9.68 | 16.26 |
| Fused 64 x 2 x 2 | 6.85 | 9.80 | 16.41 |

Microseconds per attention layer application; 56 per token.

## Kernel time per application at 640 cached tokens (profiler, us)

| Implementation | Kernel | us |
|---|---|---:|
| vLLM | `kernel_unified_attention` | 6.84 |
| vLLM | `void at::native::unrolled_elementwise_kernel<at::native::...` | 2.10 |
| vLLM | `reduce_segments` | 1.78 |
| Fused 32 x 2 x 1 | `_decode_attention_kernel` | 5.11 |
| Fused 32 x 2 x 1 | `_combine_gate_kernel` | 1.47 |

## Raw-input mode: q/k normalization and cache write inside the kernel

| Cached tokens | Layer | New k and v cached exactly | Relative error |
|---:|---|---|---:|
| 640 | full | True | 1.16e-03 |
| 640 | sliding | True | 1.09e-03 |
| 2048 | full | True | 1.13e-03 |
| 2048 | sliding | True | 1.13e-03 |
| 8192 | full | True | 1.15e-03 |
| 8192 | sliding | True | 1.15e-03 |

Time per attention layer application at 640 cached tokens (CUDA graph): q/k normalization kernel + attention 7.95 us; raw-input mode 7.25 us.
