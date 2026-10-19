# Attention backends for Dory1.0-4B on an RTX 4090 (vLLM, NVFP4)

Full serving numbers: `summary_nvfp4_attention_backends.md`. Decode profiles:
`profile_*_prompt640*/report.md`.

## Decode profile (2 loops, batch size 1, 640-token prompt)

| Backend | Tokens/s | Attention ms per token | GPU idle ms per token | Status |
|---|---:|---:|---:|---|
| TRITON_ATTN | 158 | 0.55 | 0.06 | Works |
| FLASHINFER | 155 | 0.70 | 0.08 | Works once a CUDA compiler is available (`bench/setup_cuda_toolkit.sh`) |
| FLASH_ATTN (vLLM default) | 124 | 2.29 | 0.07 | Works |
| FLEX_ATTENTION | 73 | 4.48 | 3.05 | Fails unpatched; works with `patches/vllm-flex-attention-pr57300-port.patch` |

Attention time includes the cache write and, for split kernels, the merge step.

## Serving results

- Short prompts (128 in / 1024 out): Triton is fastest by 1-3% over FlashInfer;
  both beat FlashAttention by 14-38%.
- Long prompts (2560 in / 300 out): FlashInfer is fastest in every
  configuration, 2-7% ahead of Triton, and keeps FlashAttention's
  time to first token, while Triton's is 10-35% slower.
- Greedy outputs of all four backends agree with each other for 30-256 tokens
  per prompt; divergences are floating-point ordering effects.

## FlexAttention

Unpatched, vLLM's FlexAttention backend stops with
`FailOnRecompileLimitHit`: its compiled kernel is specialized on the mask
function, the sliding-window size, and the per-step `doc_ids` shape, and Dory
interleaves sliding-window and full-attention layers, so it exceeds the
backend's limit of 16 recompilations. The same failure is reported in vLLM
issue #27178 (closed as stale) and fixed for Gemma 3 by the unmerged vLLM pull
request #57300. The patch in `patches/` is that pull request ported to the
vLLM commit used here (one hunk needed manual placement). With it,
FlexAttention recompiles 6 times during CUDA graph capture and runs correctly,
but at under half the speed of Triton or FlashInfer: its kernels are slower and
the GPU waits about 3 ms per token on CPU-side block-mask construction.
