# Decode profile: nvfp4, 4 loops, batch size 1, 640-token prompt, attention backend TRITON_ATTN

- Decode steps analyzed: 63; kernels per model forward: 589 (3.1 per sublayer, 192 sublayers)
- Step period: 7.75 ms (129.0 tokens/s)
- Model forward: 7.28 ms wall time, 7.25 ms with a kernel running, 0.03 ms idle between kernels (0.05 us per kernel boundary)
- After the forward (output projection, sampling, bookkeeping): 0.47 ms wall time, 0.45 ms busy
- Weight GEMMs inside the forward: 6.50 ms for 4.95 GB = 763 GB/s (76% of the 1008 GB/s peak)

## Time per decode step by category

| Category | ms per step | Share of step | us per sublayer | Kernels per step |
|---|---:|---:|---:|---:|
| Fused weight GEMV | 6.853 | 88% | 35.7 | 385 |
| Attention | 0.735 | 9% | 3.8 | 192 |
| Sampling | 0.062 | 1% | 0.3 | 11 |
| Other | 0.031 | 0% | 0.2 | 16 |
| Fused normalization and residual | 0.014 | 0% | 0.1 | 10 |
| Memory copies and sets | 0.005 | 0% | 0.0 | 5 |
| Idle GPU (gaps between kernels and steps) | 0.052 | 1% | 0.3 | 0 |

## Top kernels

| Kernel | Category | Calls per step | Mean us | ms per step |
|---|---|---:|---:|---:|
| `_gate_up_swiglu_kernel` | Fused weight GEMV | 96 | 24.9 | 2.393 |
| `_qkvg_rope_kernel` | Fused weight GEMV | 96 | 18.0 | 1.728 |
| `_down_normalized_kernel` | Fused weight GEMV | 96 | 14.7 | 1.415 |
| `_o_gated_normalized_kernel` | Fused weight GEMV | 96 | 10.0 | 0.960 |
| `_decode_attention_kernel` | Attention | 96 | 6.1 | 0.585 |
| `_quantized_logits_kernel` | Fused weight GEMV | 1 | 356.9 | 0.357 |
| `_combine_gate_kernel` | Attention | 96 | 1.6 | 0.150 |
| `_topp_sb_step_kernel` | Sampling | 5 | 7.6 | 0.038 |
| `_to_residual_kernel` | Fused normalization and residual | 10 | 1.4 | 0.014 |
| `_topp_sb_mask_kernel` | Sampling | 1 | 7.8 | 0.008 |
| `_gumbel_sample_kernel` | Sampling | 1 | 6.0 | 0.006 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 2 | 2.6 | 0.005 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 3.8 | 0.004 |
| `_gather_block_tables_kernel` | Other | 1 | 3.1 | 0.003 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | Sampling | 1 | 3.1 | 0.003 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | Other | 1 | 2.9 | 0.003 |
| `_post_update_kernel` | Other | 1 | 2.9 | 0.003 |
| `_combine_sampled_and_draft_tokens_kernel` | Sampling | 1 | 2.5 | 0.002 |
| `_get_num_sampled_and_rejected_kernel` | Sampling | 1 | 2.5 | 0.002 |
| `_topp_sb_stats_kernel` | Sampling | 1 | 2.3 | 0.002 |
| `_compute_slot_mappings_kernel` | Other | 1 | 2.2 | 0.002 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsi...` | Other | 1 | 2.2 | 0.002 |
| `Memcpy DtoH (Device -> Pinned)` | Memory copies and sets | 2 | 0.9 | 0.002 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | Other | 2 | 0.9 | 0.002 |
| `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_g...` | Other | 1 | 1.8 | 0.002 |

FlashInfer sampler: 0; CUDA_HOME: None; VLLM_USE_V2_MODEL_RUNNER: None; async scheduling: True
