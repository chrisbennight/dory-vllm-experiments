# Decode profile: nvfp4, 4 loops, batch size 1, 640-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 588 (3.1 per sublayer, 192 sublayers)
- Step period: 8.84 ms (113.2 tokens/s)
- Model forward: 7.31 ms wall time, 7.28 ms with a kernel running, 0.03 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 1.53 ms wall time, 0.49 ms busy
- Weight GEMMs inside the forward: 6.49 ms for 4.95 GB = 763 GB/s (76% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Fused weight GEMV | 6.850 | 78% | 35.7 | 385 |
| Attention | 0.765 | 9% | 4.0 | 192 |
| Sampling | 0.102 | 1% | 0.5 | 10 |
| Other | 0.033 | 0% | 0.2 | 20 |
| Fused normalization and residual | 0.014 | 0% | 0.1 | 10 |
| Memory copies and sets | 0.007 | 0% | 0.0 | 15 |
| Idle GPU (gaps between kernels and steps) | 1.067 | 12% | 5.6 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `_gate_up_swiglu_kernel` | Fused weight GEMV | 96 | 25.0 | 2.399 |
| `_qkvg_rope_kernel` | Fused weight GEMV | 96 | 17.8 | 1.709 |
| `_down_normalized_kernel` | Fused weight GEMV | 96 | 14.8 | 1.424 |
| `_o_gated_normalized_kernel` | Fused weight GEMV | 96 | 10.0 | 0.961 |
| `_decode_attention_kernel` | Attention | 96 | 6.4 | 0.614 |
| `_quantized_logits_kernel` | Fused weight GEMV | 1 | 356.7 | 0.357 |
| `_combine_gate_kernel` | Attention | 96 | 1.6 | 0.152 |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at:...` | Sampling | 1 | 44.2 | 0.044 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.8 | 0.039 |
| `_to_residual_kernel` | Fused normalization and residual | 10 | 1.4 | 0.014 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.8 | 0.008 |
| `Kernel` | Other | 4 | 1.7 | 0.007 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 4 | 1.5 | 0.006 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 5.8 | 0.006 |
| `Memcpy HtoD (Pinned -> Device)` | Memory copies and sets | 11 | 0.4 | 0.004 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.4 | 0.003 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | Other | 1 | 2.7 | 0.003 |
| `_topp_sb_stats_kernel` | Sampling | 1 | 2.6 | 0.003 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 2.4 | 0.002 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa...` | Other | 1 | 2.2 | 0.002 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsi...` | Other | 1 | 2.2 | 0.002 |
| `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo...` | Sampling | 1 | 2.2 | 0.002 |
| `Memcpy DtoD (Device -> Device)` | Memory copies and sets | 2 | 1.0 | 0.002 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 1 | 1.4 | 0.001 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<long>, st...` | Other | 1 | 1.2 | 0.001 |

FlashInfer sampler: 0; CUDA_HOME: None; VLLM_USE_V2_MODEL_RUNNER: 0; async scheduling: False
