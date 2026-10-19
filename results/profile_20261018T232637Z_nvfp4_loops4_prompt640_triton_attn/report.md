# Decode profile: nvfp4, 4 loops, batch size 1, 640-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 588 (3.1 per sublayer, 192 sublayers)
- Step period: 7.77 ms (128.6 tokens/s)
- Model forward: 7.23 ms wall time, 7.20 ms with a kernel running, 0.03 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.54 ms wall time, 0.49 ms busy
- Weight GEMMs inside the forward: 6.45 ms for 4.95 GB = 768 GB/s (76% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Fused weight GEMV | 6.805 | 88% | 35.4 | 385 |
| Attention | 0.737 | 9% | 3.8 | 192 |
| Sampling | 0.098 | 1% | 0.5 | 10 |
| Other | 0.031 | 0% | 0.2 | 19 |
| Fused normalization and residual | 0.014 | 0% | 0.1 | 10 |
| Memory copies and sets | 0.013 | 0% | 0.1 | 15 |
| Idle GPU (gaps between kernels and steps) | 0.076 | 1% | 0.4 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `_gate_up_swiglu_kernel` | Fused weight GEMV | 96 | 24.8 | 2.378 |
| `_qkvg_rope_kernel` | Fused weight GEMV | 96 | 17.7 | 1.704 |
| `_down_normalized_kernel` | Fused weight GEMV | 96 | 14.7 | 1.415 |
| `_o_gated_normalized_kernel` | Fused weight GEMV | 96 | 9.9 | 0.952 |
| `_decode_attention_kernel` | Attention | 96 | 6.1 | 0.588 |
| `_quantized_logits_kernel` | Fused weight GEMV | 1 | 356.8 | 0.357 |
| `_combine_gate_kernel` | Attention | 96 | 1.5 | 0.149 |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at:...` | Sampling | 1 | 42.8 | 0.043 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.5 | 0.038 |
| `_to_residual_kernel` | Fused normalization and residual | 10 | 1.4 | 0.014 |
| `Memcpy HtoD (Pinned -> Device)` | Memory copies and sets | 10 | 0.9 | 0.009 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.6 | 0.008 |
| `Kernel` | Other | 4 | 1.7 | 0.007 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 5.6 | 0.006 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 3 | 1.5 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.2 | 0.003 |
| `Memcpy DtoD (Device -> Device)` | Memory copies and sets | 3 | 0.9 | 0.003 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | Other | 1 | 2.6 | 0.003 |
| `_topp_sb_stats_kernel` | Sampling | 1 | 2.5 | 0.003 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 2.3 | 0.002 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsi...` | Other | 1 | 2.2 | 0.002 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa...` | Other | 1 | 2.2 | 0.002 |
| `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo...` | Sampling | 1 | 2.1 | 0.002 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 1 | 1.3 | 0.001 |
| `Memcpy DtoH (Device -> Pinned)` | Memory copies and sets | 1 | 1.3 | 0.001 |

FlashInfer sampler: 0; CUDA_HOME: None; VLLM_USE_V2_MODEL_RUNNER: 0; async scheduling: True
