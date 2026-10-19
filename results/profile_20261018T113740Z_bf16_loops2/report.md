# Decode profile: bf16, 2 loops, batch size 1

- Decode steps analyzed: 63; kernels per model forward: 758 (6.8 per sublayer, 112 sublayers)
- Step period: 14.00 ms (71.4 tokens/s)
- Model forward: 13.18 ms wall time, 13.14 ms with a kernel running, 0.04 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.82 ms wall time, 0.80 ms busy
- Weight GEMMs inside the forward: 11.39 ms for 10.28 GB = 902 GB/s (89% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Dense GEMM | 12.093 | 86% | 108.0 | 281 |
| Attention | 0.945 | 7% | 8.4 | 126 |
| Fused elementwise and norms (torch.compile) | 0.793 | 6% | 7.1 | 337 |
| Sampling | 0.060 | 0% | 0.5 | 11 |
| Other | 0.044 | 0% | 0.4 | 32 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.066 | 0% | 0.6 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 112 | 59.7 | 6.682 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 169 | 32.0 | 5.411 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, fal...` | Attention | 42 | 14.4 | 0.605 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_1` | Fused elementwise and norms (torch.compile) | 109 | 2.0 | 0.223 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_silu_split_sum_2` | Fused elementwise and norms (torch.compile) | 56 | 3.7 | 0.209 |
| `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, fa...` | Attention | 14 | 10.8 | 0.151 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 55 | 2.2 | 0.122 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 52 | 2.3 | 0.118 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, fal...` | Attention | 14 | 7.4 | 0.103 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_sigmoid_sum_view_0` | Fused elementwise and norms (torch.compile) | 56 | 1.8 | 0.101 |
| `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCac...` | Attention | 56 | 1.5 | 0.086 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.4 | 0.037 |
| `memcpy32_post` | Other | 15 | 0.9 | 0.013 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.5 | 0.007 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_3` | Fused elementwise and norms (torch.compile) | 3 | 2.4 | 0.007 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 3 | 2.2 | 0.007 |
| `_gumbel_sample_kernel` | Sampling | 1 | 6.6 | 0.007 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.4 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.1 | 0.004 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.0 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.9 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.8 | 0.003 |
| `_gather_block_tables_kernel` | Other | 1 | 2.8 | 0.003 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 2.7 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.7 | 0.003 |
