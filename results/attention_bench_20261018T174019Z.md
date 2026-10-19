# Fused Dory attention sublayer (NVIDIA GeForce RTX 4090, 2 loops, 56 attention sublayers per token, 640 cached tokens)

## Chosen launch configurations

- qkvg: LaunchConfig(block_n=4, block_w=32, num_warps=4) (38.34 us per sublayer)
- o: LaunchConfig(block_n=4, block_w=128, num_warps=4) (31.32 us per sublayer)

## Accuracy against the FP32 reference, one decode token per layer

Relative errors. q: normalized query; k, v: entries written to the KV cache; output: the sublayer output after the residual update.

| Layer | Window | q vLLM | q fused | k vLLM | k fused | v vLLM | v fused | output vLLM | output fused | output fused vs vLLM | output fused vs separate residual kernel |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | sliding | 2.64e-03 | 2.12e-03 | 2.73e-03 | 2.11e-03 | 1.62e-03 | 1.62e-03 | 4.16e-03 | 4.17e-03 | 4.72e-03 | 0.00e+00 |
| 2 | sliding | 2.66e-03 | 2.07e-03 | 2.77e-03 | 2.24e-03 | 1.59e-03 | 1.59e-03 | 3.72e-03 | 2.67e-03 | 3.74e-03 | 0.00e+00 |
| 4 | sliding | 2.65e-03 | 2.13e-03 | 2.63e-03 | 2.19e-03 | 1.70e-03 | 1.70e-03 | 3.58e-03 | 3.09e-03 | 3.84e-03 | 0.00e+00 |
| 6 | full | 2.41e-03 | 1.74e-03 | 2.67e-03 | 1.90e-03 | 1.59e-03 | 1.59e-03 | 3.05e-03 | 2.92e-03 | 3.56e-03 | 0.00e+00 |
| 8 | sliding | 2.69e-03 | 2.18e-03 | 2.74e-03 | 2.14e-03 | 1.69e-03 | 1.69e-03 | 3.18e-03 | 3.15e-03 | 4.15e-03 | 0.00e+00 |
| 10 | sliding | 2.69e-03 | 2.11e-03 | 2.76e-03 | 2.28e-03 | 1.64e-03 | 1.64e-03 | 4.11e-03 | 2.48e-03 | 3.56e-03 | 0.00e+00 |
| 12 | sliding | 2.65e-03 | 2.12e-03 | 2.74e-03 | 2.10e-03 | 1.64e-03 | 1.64e-03 | 3.22e-03 | 2.32e-03 | 3.01e-03 | 0.00e+00 |
| 14 | full | 2.30e-03 | 1.78e-03 | 2.63e-03 | 2.01e-03 | 1.63e-03 | 1.63e-03 | 2.94e-03 | 2.25e-03 | 3.10e-03 | 0.00e+00 |
| 16 | sliding | 2.74e-03 | 2.14e-03 | 2.77e-03 | 2.18e-03 | 1.72e-03 | 1.72e-03 | 3.05e-03 | 2.61e-03 | 3.62e-03 | 0.00e+00 |
| 18 | sliding | 2.63e-03 | 2.14e-03 | 2.71e-03 | 2.15e-03 | 1.66e-03 | 1.66e-03 | 2.93e-03 | 2.56e-03 | 3.64e-03 | 0.00e+00 |
| 20 | sliding | 2.67e-03 | 2.16e-03 | 2.64e-03 | 2.10e-03 | 1.61e-03 | 1.61e-03 | 2.27e-03 | 2.15e-03 | 2.38e-03 | 9.31e-10 |
| 22 | full | 2.34e-03 | 1.75e-03 | 2.43e-03 | 1.84e-03 | 1.72e-03 | 1.72e-03 | 3.66e-03 | 2.92e-03 | 4.08e-03 | 0.00e+00 |
| 24 | sliding | 2.63e-03 | 2.09e-03 | 2.74e-03 | 2.22e-03 | 1.63e-03 | 1.63e-03 | 2.36e-03 | 2.00e-03 | 2.24e-03 | 2.98e-08 |
| 26 | sliding | 2.69e-03 | 2.10e-03 | 2.51e-03 | 2.09e-03 | 1.62e-03 | 1.62e-03 | 2.52e-03 | 2.15e-03 | 2.55e-03 | 0.00e+00 |
| 28 | sliding | 2.70e-03 | 2.11e-03 | 2.79e-03 | 2.26e-03 | 1.61e-03 | 1.61e-03 | 2.64e-03 | 2.21e-03 | 2.80e-03 | 0.00e+00 |
| 30 | full | 2.48e-03 | 1.77e-03 | 2.52e-03 | 2.03e-03 | 1.66e-03 | 1.66e-03 | 2.78e-03 | 2.25e-03 | 2.99e-03 | 1.86e-09 |
| 32 | sliding | 2.71e-03 | 2.06e-03 | 2.78e-03 | 2.21e-03 | 1.69e-03 | 1.69e-03 | 2.26e-03 | 1.89e-03 | 2.18e-03 | 0.00e+00 |
| 34 | sliding | 2.68e-03 | 2.04e-03 | 2.75e-03 | 2.18e-03 | 1.66e-03 | 1.66e-03 | 2.77e-03 | 2.06e-03 | 2.70e-03 | 0.00e+00 |
| 36 | sliding | 2.60e-03 | 2.09e-03 | 2.54e-03 | 2.08e-03 | 1.58e-03 | 1.58e-03 | 3.02e-03 | 2.41e-03 | 2.67e-03 | 0.00e+00 |
| 38 | full | 2.40e-03 | 1.76e-03 | 2.56e-03 | 2.01e-03 | 1.65e-03 | 1.65e-03 | 3.98e-03 | 2.86e-03 | 4.20e-03 | 0.00e+00 |
| 40 | sliding | 2.68e-03 | 2.20e-03 | 2.90e-03 | 2.25e-03 | 1.65e-03 | 1.65e-03 | 2.82e-03 | 2.42e-03 | 3.11e-03 | 0.00e+00 |
| 42 | sliding | 2.77e-03 | 2.14e-03 | 2.82e-03 | 2.20e-03 | 1.61e-03 | 1.61e-03 | 2.97e-03 | 2.44e-03 | 3.30e-03 | 0.00e+00 |
| 44 | sliding | 2.76e-03 | 2.12e-03 | 2.76e-03 | 2.15e-03 | 1.62e-03 | 1.62e-03 | 2.26e-03 | 2.08e-03 | 2.54e-03 | 0.00e+00 |
| 46 | full | 2.42e-03 | 1.79e-03 | 2.55e-03 | 1.98e-03 | 1.67e-03 | 1.67e-03 | 3.23e-03 | 2.82e-03 | 3.88e-03 | 0.00e+00 |
| 48 | sliding | 2.68e-03 | 2.16e-03 | 2.71e-03 | 2.13e-03 | 1.66e-03 | 1.66e-03 | 3.02e-03 | 2.30e-03 | 3.50e-03 | 0.00e+00 |
| 50 | sliding | 2.61e-03 | 2.15e-03 | 2.53e-03 | 2.13e-03 | 1.77e-03 | 1.77e-03 | 4.16e-03 | 2.45e-03 | 4.10e-03 | 0.00e+00 |
| 52 | sliding | 2.73e-03 | 2.14e-03 | 2.69e-03 | 2.20e-03 | 1.68e-03 | 1.68e-03 | 2.52e-03 | 2.55e-03 | 3.06e-03 | 0.00e+00 |
| 54 | full | 2.39e-03 | 1.74e-03 | 2.52e-03 | 1.93e-03 | 1.56e-03 | 1.56e-03 | 3.10e-03 | 2.51e-03 | 4.16e-03 | 0.00e+00 |
| 56 | sliding | 2.69e-03 | 2.07e-03 | 2.70e-03 | 2.28e-03 | 1.61e-03 | 1.61e-03 | 3.45e-03 | 2.83e-03 | 3.73e-03 | 0.00e+00 |
| 58 | sliding | 2.67e-03 | 2.11e-03 | 2.66e-03 | 2.16e-03 | 1.67e-03 | 1.67e-03 | 2.61e-03 | 2.09e-03 | 3.15e-03 | 1.86e-09 |
| 60 | sliding | 2.65e-03 | 2.10e-03 | 2.69e-03 | 2.14e-03 | 1.60e-03 | 1.60e-03 | 2.42e-03 | 2.22e-03 | 2.78e-03 | 3.81e-06 |
| 62 | full | 2.39e-03 | 1.74e-03 | 2.40e-03 | 1.78e-03 | 1.63e-03 | 1.63e-03 | 2.41e-03 | 2.13e-03 | 2.63e-03 | 0.00e+00 |
| 64 | sliding | 2.70e-03 | 2.08e-03 | 2.76e-03 | 2.09e-03 | 1.74e-03 | 1.74e-03 | 4.14e-03 | 3.21e-03 | 3.82e-03 | 0.00e+00 |
| 66 | sliding | 2.60e-03 | 2.03e-03 | 2.64e-03 | 2.01e-03 | 1.66e-03 | 1.66e-03 | 2.47e-03 | 2.29e-03 | 2.78e-03 | 0.00e+00 |
| 68 | sliding | 2.68e-03 | 2.18e-03 | 2.73e-03 | 2.25e-03 | 1.66e-03 | 1.66e-03 | 2.97e-03 | 2.49e-03 | 3.55e-03 | 0.00e+00 |
| 70 | full | 2.40e-03 | 1.68e-03 | 2.56e-03 | 1.95e-03 | 1.68e-03 | 1.68e-03 | 5.32e-03 | 2.47e-03 | 6.80e-03 | 0.00e+00 |
| worst | | 2.77e-03 | 2.20e-03 | 2.90e-03 | 2.28e-03 | 1.77e-03 | 1.77e-03 | 5.32e-03 | 4.17e-03 | 6.80e-03 | 3.81e-06 |

## Time per token for all attention sublayers, attention kernel excluded

| Implementation | Kernels per sublayer (without attention) | ms per token | us per sublayer |
|---|---:|---:|---:|
| vLLM path | 7 | 2.247 | 40.1 |
| Fused, separate residual kernel | 4 | 1.713 | 30.6 |
| Fused | 3 | 1.746 | 31.2 |
| Saving | | 0.502 | 9.0 |

## Kernel time per sublayer (profiler, us)

| Implementation | Kernel | us per sublayer |
|---|---|---:|
| vLLM path | `Marlin GEMM` | 32.5 |
| vLLM path | `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rs...` | 2.4 |
| vLLM path | `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_0` | 2.2 |
| vLLM path | `triton_red_fused__to_copy_add_mul_pow_rsqrt_sigmoid_sum_0` | 1.9 |
| vLLM path | `reshape_and_cache_kernel_flash` | 1.1 |
| Fused | `_qkvg_rope_kernel` | 17.9 |
| Fused | `_o_gated_normalized_kernel` | 11.8 |
| Fused | `_qk_norm_cache_kernel` | 1.3 |
