# Decode profile: nvfp4, 2 loops, batch size 1, 640-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 786 (7.0 per sublayer, 112 sublayers)
- Step period: 6.40 ms (156.2 tokens/s)
- Model forward: 5.58 ms wall time, 5.54 ms with a kernel running, 0.04 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.82 ms wall time, 0.80 ms busy
- Weight GEMMs inside the forward: 4.12 ms for 2.89 GB = 702 GB/s (70% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Marlin weight GEMM | 4.118 | 64% | 36.8 | 280 |
| Fused elementwise and norms (torch.compile) | 0.817 | 13% | 7.3 | 337 |
| Dense GEMM | 0.702 | 11% | 6.3 | 1 |
| Attention | 0.426 | 7% | 3.8 | 112 |
| Other | 0.212 | 3% | 1.9 | 74 |
| Sampling | 0.060 | 1% | 0.5 | 11 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.064 | 1% | 0.6 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 112 | 19.6 | 2.194 |
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 168 | 11.5 | 1.925 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.6 | 0.702 |
| `kernel_unified_attention` | Attention | 56 | 6.4 | 0.360 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_1` | Fused elementwise and norms (torch.compile) | 109 | 2.1 | 0.228 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_silu_split_sum_2` | Fused elementwise and norms (torch.compile) | 56 | 3.8 | 0.212 |
| `reduce_segments` | Other | 56 | 3.2 | 0.178 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 55 | 2.3 | 0.125 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 52 | 2.3 | 0.119 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_sigmoid_sum_view_0` | Fused elementwise and norms (torch.compile) | 56 | 2.0 | 0.112 |
| `reshape_and_cache_kernel_flash` | Attention | 56 | 1.2 | 0.066 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.5 | 0.037 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.7 | 0.008 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_3` | Fused elementwise and norms (torch.compile) | 3 | 2.5 | 0.008 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 3 | 2.3 | 0.007 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.5 | 0.005 |
| `_gumbel_sample_kernel` | Sampling | 1 | 4.7 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.4 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.9 | 0.004 |
| `_gather_block_tables_kernel` | Other | 1 | 3.1 | 0.003 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.0 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.9 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.8 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.8 | 0.003 |
| `_combine_sampled_and_draft_tokens_kernel` | Sampling | 1 | 2.5 | 0.002 |

FlashInfer sampler: 0; CUDA_HOME: None
