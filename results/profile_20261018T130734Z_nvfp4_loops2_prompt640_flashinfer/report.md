# Decode profile: nvfp4, 2 loops, batch size 1, 640-token prompt, attention backend FLASHINFER

- Decode steps analyzed: 63; kernels per model forward: 786 (7.0 per sublayer, 112 sublayers)
- Step period: 6.46 ms (154.8 tokens/s)
- Model forward: 5.60 ms wall time, 5.57 ms with a kernel running, 0.04 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.85 ms wall time, 0.81 ms busy
- Weight GEMMs inside the forward: 4.05 ms for 2.89 GB = 713 GB/s (71% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Marlin weight GEMM | 4.053 | 63% | 36.2 | 280 |
| Fused elementwise and norms (torch.compile) | 0.809 | 13% | 7.2 | 337 |
| Attention | 0.703 | 11% | 6.3 | 168 |
| Dense GEMM | 0.702 | 11% | 6.3 | 1 |
| Sampling | 0.060 | 1% | 0.5 | 11 |
| Other | 0.039 | 1% | 0.3 | 22 |
| Memory copies and sets | 0.011 | 0% | 0.1 | 17 |
| Idle GPU (gaps between kernels and steps) | 0.083 | 1% | 0.7 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 112 | 19.1 | 2.140 |
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 168 | 11.4 | 1.913 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.5 | 0.702 |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, flashinfer::KernelTraits<(fla...` | Attention | 42 | 8.9 | 0.375 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_1` | Fused elementwise and norms (torch.compile) | 109 | 2.1 | 0.228 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_silu_split_sum_2` | Fused elementwise and norms (torch.compile) | 56 | 3.8 | 0.212 |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, flashinfer::KernelTraits<(fla...` | Attention | 14 | 8.9 | 0.125 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 55 | 2.3 | 0.124 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 52 | 2.3 | 0.120 |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 32u, 4u, 4u, __nv_bfloat...` | Attention | 56 | 1.9 | 0.108 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_sigmoid_sum_view_0` | Fused elementwise and norms (torch.compile) | 56 | 1.8 | 0.103 |
| `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCac...` | Attention | 56 | 1.7 | 0.095 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.5 | 0.037 |
| `Memcpy HtoD (Pinned -> Device)` | Memory copies and sets | 14 | 0.6 | 0.008 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.7 | 0.008 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_3` | Fused elementwise and norms (torch.compile) | 3 | 2.5 | 0.008 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 3 | 2.3 | 0.007 |
| `_copy_page_indices_kernel` | Other | 4 | 1.4 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.5 | 0.005 |
| `_gumbel_sample_kernel` | Sampling | 1 | 4.9 | 0.005 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 2 | 2.4 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.9 | 0.004 |
| `_gather_block_tables_kernel` | Other | 1 | 3.1 | 0.003 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.1 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.9 | 0.003 |

FlashInfer sampler: 0; CUDA_HOME: <repo>/cuda/home
