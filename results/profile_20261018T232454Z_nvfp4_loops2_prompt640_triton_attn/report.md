# Decode profile: nvfp4, 2 loops, batch size 1, 640-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 344 (3.1 per sublayer, 112 sublayers)
- Step period: 5.66 ms (176.5 tokens/s)
- Model forward: 4.25 ms wall time, 4.23 ms with a kernel running, 0.02 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 1.42 ms wall time, 0.49 ms busy
- Weight GEMMs inside the forward: 3.78 ms for 2.89 GB = 764 GB/s (76% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Fused weight GEMV | 4.141 | 73% | 37.0 | 225 |
| Attention | 0.433 | 8% | 3.9 | 112 |
| Sampling | 0.100 | 2% | 0.9 | 10 |
| Other | 0.033 | 1% | 0.3 | 20 |
| Fused normalization and residual | 0.008 | 0% | 0.1 | 6 |
| Memory copies and sets | 0.007 | 0% | 0.1 | 15 |
| Idle GPU (gaps between kernels and steps) | 0.941 | 17% | 8.4 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `_gate_up_swiglu_kernel` | Fused weight GEMV | 56 | 24.9 | 1.392 |
| `_qkvg_rope_kernel` | Fused weight GEMV | 56 | 17.8 | 0.998 |
| `_down_normalized_kernel` | Fused weight GEMV | 56 | 14.9 | 0.832 |
| `_o_gated_normalized_kernel` | Fused weight GEMV | 56 | 10.1 | 0.563 |
| `_quantized_logits_kernel` | Fused weight GEMV | 1 | 356.8 | 0.357 |
| `_decode_attention_kernel` | Attention | 56 | 6.2 | 0.345 |
| `_combine_gate_kernel` | Attention | 56 | 1.6 | 0.088 |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at:...` | Sampling | 1 | 44.3 | 0.044 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.5 | 0.037 |
| `_to_residual_kernel` | Fused normalization and residual | 6 | 1.3 | 0.008 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.6 | 0.008 |
| `Kernel` | Other | 4 | 1.7 | 0.007 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 4 | 1.5 | 0.006 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 5.8 | 0.006 |
| `Memcpy HtoD (Pinned -> Device)` | Memory copies and sets | 11 | 0.4 | 0.004 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.3 | 0.003 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | Other | 1 | 2.7 | 0.003 |
| `_topp_sb_stats_kernel` | Sampling | 1 | 2.6 | 0.003 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 2.4 | 0.002 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa...` | Other | 1 | 2.2 | 0.002 |
| `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo...` | Sampling | 1 | 2.2 | 0.002 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsi...` | Other | 1 | 2.1 | 0.002 |
| `Memcpy DtoD (Device -> Device)` | Memory copies and sets | 2 | 1.0 | 0.002 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 1 | 1.4 | 0.001 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<long>, st...` | Other | 1 | 1.2 | 0.001 |

FlashInfer sampler: 0; CUDA_HOME: None; VLLM_USE_V2_MODEL_RUNNER: 0; async scheduling: False
