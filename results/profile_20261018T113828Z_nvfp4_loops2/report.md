# Decode profile: nvfp4, 2 loops, batch size 1

- Decode steps analyzed: 63; kernels per model forward: 758 (6.8 per sublayer, 112 sublayers)
- Step period: 8.05 ms (124.3 tokens/s)
- Model forward: 7.22 ms wall time, 7.18 ms with a kernel running, 0.04 ms idle between kernels (0.06 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.82 ms wall time, 0.80 ms busy
- Weight GEMMs inside the forward: 4.07 ms for 2.89 GB = 710 GB/s (70% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Marlin weight GEMM | 4.069 | 51% | 36.3 | 280 |
| Attention | 2.291 | 28% | 20.5 | 126 |
| Fused elementwise and norms (torch.compile) | 0.805 | 10% | 7.2 | 337 |
| Dense GEMM | 0.702 | 9% | 6.3 | 1 |
| Sampling | 0.060 | 1% | 0.5 | 11 |
| Other | 0.046 | 1% | 0.4 | 32 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.069 | 1% | 0.6 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 112 | 19.2 | 2.151 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, fal...` | Attention | 42 | 45.9 | 1.928 |
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 168 | 11.4 | 1.918 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.5 | 0.702 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_1` | Fused elementwise and norms (torch.compile) | 109 | 2.1 | 0.227 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_silu_split_sum_2` | Fused elementwise and norms (torch.compile) | 56 | 3.8 | 0.211 |
| `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, fa...` | Attention | 14 | 10.9 | 0.152 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 55 | 2.3 | 0.124 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 52 | 2.3 | 0.119 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, fal...` | Attention | 14 | 8.3 | 0.117 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_sigmoid_sum_view_0` | Fused elementwise and norms (torch.compile) | 56 | 1.8 | 0.102 |
| `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCac...` | Attention | 56 | 1.7 | 0.094 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.4 | 0.037 |
| `memcpy32_post` | Other | 15 | 0.9 | 0.013 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.5 | 0.007 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_3` | Fused elementwise and norms (torch.compile) | 3 | 2.5 | 0.007 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 3 | 2.3 | 0.007 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.6 | 0.005 |
| `_gumbel_sample_kernel` | Sampling | 1 | 4.9 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.3 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.9 | 0.004 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.1 | 0.003 |
| `_gather_block_tables_kernel` | Other | 1 | 3.0 | 0.003 |
| `_post_update_kernel` | Other | 1 | 3.0 | 0.003 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 1 | 2.9 | 0.003 |
