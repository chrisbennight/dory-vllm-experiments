# Decode profile: nvfp4, 2 loops, batch size 1, 7936-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 569 (5.1 per sublayer, 112 sublayers)
- Step period: 6.71 ms (149.0 tokens/s)
- Model forward: 5.89 ms wall time, 5.86 ms with a kernel running, 0.03 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.82 ms wall time, 0.80 ms busy
- Weight GEMMs inside the forward: 3.69 ms for 2.89 GB = 783 GB/s (78% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Fused weight GEMV | 3.690 | 55% | 32.9 | 224 |
| Attention | 1.760 | 26% | 15.7 | 112 |
| Dense GEMM | 0.702 | 10% | 6.3 | 1 |
| Fused normalization and residual | 0.304 | 5% | 2.7 | 174 |
| Other | 0.137 | 2% | 1.2 | 76 |
| Sampling | 0.061 | 1% | 0.5 | 11 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.054 | 1% | 0.5 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `kernel_unified_attention` | Attention | 56 | 30.3 | 1.695 |
| `_gate_up_swiglu_kernel` | Fused weight GEMV | 56 | 25.3 | 1.416 |
| `_qkvg_rope_kernel` | Fused weight GEMV | 56 | 17.8 | 0.997 |
| `_down_normalized_kernel` | Fused weight GEMV | 56 | 13.2 | 0.737 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.8 | 0.702 |
| `_o_gated_normalized_kernel` | Fused weight GEMV | 56 | 9.6 | 0.540 |
| `_residual_update_kernel` | Fused normalization and residual | 112 | 2.0 | 0.221 |
| `reduce_segments` | Other | 56 | 1.8 | 0.099 |
| `_qk_norm_cache_kernel` | Fused normalization and residual | 56 | 1.3 | 0.073 |
| `reshape_and_cache_kernel_flash` | Attention | 56 | 1.2 | 0.065 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.6 | 0.038 |
| `_to_residual_kernel` | Fused normalization and residual | 6 | 1.6 | 0.009 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.7 | 0.008 |
| `_gumbel_sample_kernel` | Sampling | 1 | 5.2 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.6 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.4 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.9 | 0.004 |
| `_gather_block_tables_kernel` | Other | 1 | 3.2 | 0.003 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.1 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.8 | 0.003 |
| `_combine_sampled_and_draft_tokens_kernel` | Sampling | 1 | 2.5 | 0.003 |
| `_get_num_sampled_and_rejected_kernel` | Sampling | 1 | 2.5 | 0.002 |
| `_topp_sb_stats_kernel` | Sampling | 1 | 2.3 | 0.002 |
| `_compute_slot_mappings_kernel` | Other | 1 | 2.2 | 0.002 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 2.1 | 0.002 |

FlashInfer sampler: 0; CUDA_HOME: None
