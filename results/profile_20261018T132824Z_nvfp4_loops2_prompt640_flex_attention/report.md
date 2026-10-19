# Decode profile: nvfp4, 2 loops, batch size 1, 640-token prompt, attention backend FLEX_ATTENTION

- Decode steps analyzed: 63; kernels per model forward: 786 (7.0 per sublayer, 112 sublayers)
- Step period: 13.66 ms (73.2 tokens/s)
- Model forward: 9.46 ms wall time, 9.42 ms with a kernel running, 0.04 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 4.20 ms wall time, 1.20 ms busy
- Weight GEMMs inside the forward: 4.06 ms for 2.89 GB = 712 GB/s (71% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Attention | 4.484 | 33% | 40.0 | 112 |
| Marlin weight GEMM | 4.060 | 30% | 36.3 | 280 |
| Fused elementwise and norms (torch.compile) | 0.822 | 6% | 7.3 | 337 |
| Dense GEMM | 0.702 | 5% | 6.3 | 1 |
| Other | 0.448 | 3% | 4.0 | 324 |
| Sampling | 0.062 | 0% | 0.5 | 11 |
| Memory copies and sets | 0.038 | 0% | 0.3 | 45 |
| Idle GPU (gaps between kernels and steps) | 3.047 | 22% | 27.2 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `triton_tem_fused_flex_attention_0` | Attention | 56 | 78.4 | 4.388 |
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 112 | 19.2 | 2.146 |
| `void marlin::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671...` | Marlin weight GEMM | 168 | 11.4 | 1.914 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, _...` | Dense GEMM | 1 | 701.6 | 0.702 |
| `triton_red_fused__to_copy_abs_add_mul_pow_rsqrt_sub_sum_1` | Fused elementwise and norms (torch.compile) | 109 | 2.1 | 0.231 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_silu_split_sum_2` | Fused elementwise and norms (torch.compile) | 56 | 3.9 | 0.217 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 55 | 2.3 | 0.127 |
| `triton_per_fused__to_copy_add_cat_index_select_mul_pow_rsqrt_split_split_with_sizes_sub...` | Fused elementwise and norms (torch.compile) | 52 | 2.3 | 0.121 |
| `triton_red_fused__to_copy_add_mul_pow_rsqrt_sigmoid_sum_view_0` | Fused elementwise and norms (torch.compile) | 56 | 1.9 | 0.104 |
| `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCac...` | Attention | 56 | 1.7 | 0.096 |
| `memcpy32_post` | Other | 57 | 0.9 | 0.051 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | Other | 32 | 1.6 | 0.050 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.7 | 0.038 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 18 | 1.9 | 0.034 |
| `Memcpy DtoD (Device -> Device)` | Memory copies and sets | 33 | 1.0 | 0.032 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::a...` | Other | 24 | 0.9 | 0.022 |
| `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_g...` | Other | 4 | 5.3 | 0.021 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::(an...` | Other | 8 | 2.4 | 0.019 |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | Other | 20 | 0.9 | 0.018 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, i...` | Other | 12 | 1.3 | 0.016 |
| `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_g...` | Other | 4 | 3.8 | 0.015 |
| `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_g...` | Other | 4 | 3.1 | 0.013 |
| `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_g...` | Other | 5 | 2.5 | 0.012 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<lon...` | Other | 11 | 1.1 | 0.012 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::ma...` | Other | 11 | 1.1 | 0.012 |

FlashInfer sampler: 0; CUDA_HOME: <repo>/cuda/home
