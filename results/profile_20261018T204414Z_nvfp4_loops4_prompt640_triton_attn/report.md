# Decode profile: nvfp4, 4 loops, batch size 1, 640-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 1346 (7.0 per sublayer, 192 sublayers)
- Step period: 10.22 ms (97.9 tokens/s)
- Model forward: 9.39 ms wall time, 9.33 ms with a kernel running, 0.07 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.82 ms wall time, 0.80 ms busy
- Weight GEMMs inside the forward: 7.00 ms for 4.95 GB = 708 GB/s (70% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Marlin weight GEMM | 6.999 | 68% | 36.5 | 480 |
| Fused elementwise and norms (torch.compile) | 1.383 | 14% | 7.2 | 577 |
| Attention | 0.766 | 7% | 4.0 | 192 |
| Dense GEMM | 0.701 | 7% | 3.7 | 1 |
| Other | 0.212 | 2% | 1.1 | 114 |
| Sampling | 0.062 | 1% | 0.3 | 11 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.091 | 1% | 0.5 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 192 | 19.4 | 3.718 |
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 288 | 11.4 | 3.281 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.5 | 0.701 |
| `kernel_unified_attention` | Attention | 96 | 6.8 | 0.653 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_1` | Fused elementwise and norms (torch.compile) | 187 | 2.1 | 0.393 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_silu_split_sum_2` | Fused elementwise and norms (torch.compile) | 96 | 3.8 | 0.365 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 95 | 2.3 | 0.214 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 90 | 2.3 | 0.204 |
| `reduce_segments` | Other | 96 | 1.9 | 0.178 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_sigmoid_sum_view_0` | Fused elementwise and norms (torch.compile) | 96 | 1.8 | 0.176 |
| `reshape_and_cache_kernel_flash` | Attention | 96 | 1.2 | 0.113 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.6 | 0.038 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_3` | Fused elementwise and norms (torch.compile) | 5 | 2.6 | 0.013 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 5 | 2.3 | 0.011 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.7 | 0.008 |
| `_gumbel_sample_kernel` | Sampling | 1 | 5.8 | 0.006 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.6 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.3 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.9 | 0.004 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.1 | 0.003 |
| `_gather_block_tables_kernel` | Other | 1 | 3.1 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.8 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.8 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.7 | 0.003 |
| `_combine_sampled_and_draft_tokens_kernel` | Sampling | 1 | 2.5 | 0.003 |

FlashInfer sampler: 0; CUDA_HOME: None
