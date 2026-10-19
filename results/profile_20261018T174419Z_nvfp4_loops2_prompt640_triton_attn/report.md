# Decode profile: nvfp4, 2 loops, batch size 1, 640-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 569 (5.1 per sublayer, 112 sublayers)
- Step period: 5.40 ms (185.3 tokens/s)
- Model forward: 4.57 ms wall time, 4.54 ms with a kernel running, 0.03 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.82 ms wall time, 0.80 ms busy
- Weight GEMMs inside the forward: 3.69 ms for 2.89 GB = 782 GB/s (78% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Fused weight GEMV | 3.694 | 68% | 33.0 | 224 |
| Dense GEMM | 0.702 | 13% | 6.3 | 1 |
| Attention | 0.447 | 8% | 4.0 | 112 |
| Fused normalization and residual | 0.294 | 5% | 2.6 | 174 |
| Other | 0.142 | 3% | 1.3 | 76 |
| Sampling | 0.061 | 1% | 0.5 | 11 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.054 | 1% | 0.5 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `_gate_up_swiglu_kernel` | Fused weight GEMV | 56 | 25.3 | 1.418 |
| `_qkvg_rope_kernel` | Fused weight GEMV | 56 | 17.8 | 0.998 |
| `_down_normalized_kernel` | Fused weight GEMV | 56 | 13.2 | 0.738 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.5 | 0.702 |
| `_o_gated_normalized_kernel` | Fused weight GEMV | 56 | 9.7 | 0.540 |
| `kernel_unified_attention` | Attention | 56 | 6.8 | 0.381 |
| `_residual_update_kernel` | Fused normalization and residual | 112 | 1.9 | 0.212 |
| `reduce_segments` | Other | 56 | 1.9 | 0.104 |
| `_qk_norm_cache_kernel` | Fused normalization and residual | 56 | 1.3 | 0.074 |
| `reshape_and_cache_kernel_flash` | Attention | 56 | 1.2 | 0.066 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.6 | 0.038 |
| `_to_residual_kernel` | Fused normalization and residual | 6 | 1.3 | 0.008 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.8 | 0.008 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.6 | 0.005 |
| `_gumbel_sample_kernel` | Sampling | 1 | 5.0 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.4 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.9 | 0.004 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.1 | 0.003 |
| `_gather_block_tables_kernel` | Other | 1 | 3.1 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.8 | 0.003 |
| `_combine_sampled_and_draft_tokens_kernel` | Sampling | 1 | 2.5 | 0.002 |
| `_get_num_sampled_and_rejected_kernel` | Sampling | 1 | 2.4 | 0.002 |
| `_topp_sb_stats_kernel` | Sampling | 1 | 2.3 | 0.002 |
| `_compute_slot_mappings_kernel` | Other | 1 | 2.2 | 0.002 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 2.2 | 0.002 |

FlashInfer sampler: 0; CUDA_HOME: None
