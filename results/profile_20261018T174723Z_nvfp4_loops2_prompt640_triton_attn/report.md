# Decode profile: nvfp4, 2 loops, batch size 1, 640-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 401 (3.6 per sublayer, 112 sublayers)
- Step period: 5.37 ms (186.3 tokens/s)
- Model forward: 4.54 ms wall time, 4.52 ms with a kernel running, 0.02 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.82 ms wall time, 0.80 ms busy
- Weight GEMMs inside the forward: 3.90 ms for 2.89 GB = 741 GB/s (73% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Fused weight GEMV | 3.902 | 73% | 34.8 | 224 |
| Dense GEMM | 0.702 | 13% | 6.3 | 1 |
| Attention | 0.356 | 7% | 3.2 | 56 |
| Other | 0.215 | 4% | 1.9 | 76 |
| Fused normalization and residual | 0.083 | 2% | 0.7 | 62 |
| Sampling | 0.061 | 1% | 0.5 | 11 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.046 | 1% | 0.4 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `_gate_up_swiglu_kernel` | Fused weight GEMV | 56 | 25.7 | 1.442 |
| `_qkvg_rope_kernel` | Fused weight GEMV | 56 | 17.8 | 0.998 |
| `_down_normalized_kernel` | Fused weight GEMV | 56 | 14.9 | 0.834 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.5 | 0.702 |
| `_o_gated_normalized_kernel` | Fused weight GEMV | 56 | 11.2 | 0.628 |
| `kernel_unified_attention` | Attention | 56 | 6.3 | 0.356 |
| `reduce_segments` | Other | 56 | 3.2 | 0.178 |
| `_qk_norm_cache_kernel` | Fused normalization and residual | 56 | 1.3 | 0.075 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.5 | 0.038 |
| `_to_residual_kernel` | Fused normalization and residual | 6 | 1.4 | 0.008 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.7 | 0.008 |
| `_gumbel_sample_kernel` | Sampling | 1 | 5.6 | 0.006 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.6 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.4 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.9 | 0.004 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.1 | 0.003 |
| `_gather_block_tables_kernel` | Other | 1 | 3.1 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.8 | 0.003 |
| `_combine_sampled_and_draft_tokens_kernel` | Sampling | 1 | 2.5 | 0.003 |
| `_get_num_sampled_and_rejected_kernel` | Sampling | 1 | 2.5 | 0.002 |
| `_topp_sb_stats_kernel` | Sampling | 1 | 2.3 | 0.002 |
| `_compute_slot_mappings_kernel` | Other | 1 | 2.2 | 0.002 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 2.1 | 0.002 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsi...` | Other | 1 | 2.1 | 0.002 |
| `Memcpy DtoH (Device -> Pinned)` | Memory copies and sets | 2 | 0.9 | 0.002 |

FlashInfer sampler: 0; CUDA_HOME: None
