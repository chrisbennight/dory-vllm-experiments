# Decode profile: nvfp4, 2 loops, batch size 1, 4096-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 786 (7.0 per sublayer, 112 sublayers)
- Step period: 6.98 ms (143.2 tokens/s)
- Model forward: 6.16 ms wall time, 6.12 ms with a kernel running, 0.04 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.82 ms wall time, 0.80 ms busy
- Weight GEMMs inside the forward: 4.09 ms for 2.89 GB = 707 GB/s (70% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Marlin weight GEMM | 4.088 | 59% | 36.5 | 280 |
| Attention | 1.108 | 16% | 9.9 | 112 |
| Fused elementwise and norms (torch.compile) | 0.817 | 12% | 7.3 | 337 |
| Dense GEMM | 0.702 | 10% | 6.3 | 1 |
| Other | 0.139 | 2% | 1.2 | 74 |
| Sampling | 0.060 | 1% | 0.5 | 11 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.064 | 1% | 0.6 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 112 | 19.4 | 2.171 |
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 168 | 11.4 | 1.918 |
| `kernel_unified_attention` | Attention | 56 | 18.6 | 1.042 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.9 | 0.702 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_1` | Fused elementwise and norms (torch.compile) | 109 | 2.1 | 0.228 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_silu_split_sum_2` | Fused elementwise and norms (torch.compile) | 56 | 3.8 | 0.212 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 55 | 2.3 | 0.124 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 52 | 2.3 | 0.119 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_sigmoid_sum_view_0` | Fused elementwise and norms (torch.compile) | 56 | 2.0 | 0.112 |
| `reduce_segments` | Other | 56 | 1.9 | 0.105 |
| `reshape_and_cache_kernel_flash` | Attention | 56 | 1.2 | 0.066 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.4 | 0.037 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_3` | Fused elementwise and norms (torch.compile) | 3 | 2.5 | 0.008 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.4 | 0.007 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 3 | 2.3 | 0.007 |
| `_gumbel_sample_kernel` | Sampling | 1 | 5.5 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.5 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.4 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.9 | 0.004 |
| `_gather_block_tables_kernel` | Other | 1 | 3.1 | 0.003 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.1 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.9 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.8 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.7 | 0.003 |
| `_combine_sampled_and_draft_tokens_kernel` | Sampling | 1 | 2.5 | 0.002 |

FlashInfer sampler: 0; CUDA_HOME: None
