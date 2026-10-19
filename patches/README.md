# Patches to vLLM

## What they apply to

These patches target the Dory support pull request
[vllm#60149](https://github.com/vllm-project/vllm/pull/60149) ("[Model] Add
Dory support", branch `shan18/vllm:add-dory`):

| | Commit |
|---|---|
| Pull request head the patches apply to | `55076950b6c155f384d6766d63563c3bc20a0e3f` (reports itself as vLLM `0.1.dev1+g55076950b`) |
| Its base on vLLM's main branch | `153a17166e5d07bc38184dedeea542d2e8885e55` |

`bench/setup_vllm.sh` checks out the pinned commit into `src/vllm` and installs it
into `.venv` with the precompiled kernels vLLM publishes for the base commit.
The patches change Python files only, so no compilation is needed.

## vllm-dory-fused-decode.patch

```shell
bench/setup_vllm.sh
git -C src/vllm apply ../../patches/vllm-dory-fused-decode.patch
```

With `DORY_FUSED_DECODE` unset and `wavefront_decode` disabled in the model
config, the patched vLLM behaves like the pinned version above. The
[vllm-patched directory](../vllm-patched/) contains all six patched files in full.

| File | Change |
|---|---|
| `vllm/model_executor/models/dory.py` | With `DORY_FUSED_DECODE` pointing at this repository's `kernels/`, single-token steps run `kernels/dory_fused_model.py` and single-token logits come from its INT8 LM head; the raw checkpoint tensors are kept while loading to build its weight layouts. With `wavefront_decode`, also builds `kernels/dory_wavefront_engine.py` and implements the hooks below. |
| `vllm/transformers_utils/configs/dory.py` | `wavefront_decode` (default off) and `kv_lookahead_tokens`: `n_recurrent_loops - 1` when it is on. |
| `vllm/config/vllm.py` | Without a speculative config, `num_lookahead_tokens` takes the model config's `kv_lookahead_tokens`. |
| `vllm/v1/core/sched/scheduler.py` | Without a speculative config, lookahead slots are reserved only while at most one request runs. |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | A model can ask that single-request, single-token steps skip FULL CUDA graphs (they then run PIECEWISE). |
| `vllm/v1/worker/gpu/model_runner.py` | Calls the model's optional hooks: `bind_sampler` and `bind_block_tables` once, `begin_step(req_ids, idx_mapping_np)` before every real forward, and reads `eager_single_token_decode`. |

Use it with:

```shell
DORY_FUSED_DECODE=$PWD/kernels VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/vllm serve models/nvfp4 --attention-backend TRITON_ATTN \
  --async-scheduling --hf-overrides \
  '{"n_recurrent_loops": 4, "recurrent_kv_cache_mode": "per_loop", "wavefront_decode": true}'
```

The fused path requires vLLM's V2 model runner (the default), Triton attention
with a BF16 KV cache, and tensor parallelism set to 1. Wavefront decoding also
requires `recurrent_kv_cache_mode` set to `per_loop`, at least 2 loops, and no
speculative config.

## vllm-flex-attention-pr57300-port.patch

This patch is optional and can be applied independently. FlexAttention fails on
Dory with `FailOnRecompileLimitHit`, because it recompiles for each mix of
sliding-window and full-attention layers. This is the unmerged fix
[vllm#57300](https://github.com/vllm-project/vllm/pull/57300) ported to the
commit above; it only touches `vllm/v1/attention/backends/flex_attention.py`.

```shell
git -C src/vllm apply ../../patches/vllm-flex-attention-pr57300-port.patch
# undo: git -C src/vllm checkout -- vllm/v1/attention/backends/flex_attention.py
```

FlexAttention then works but is the slowest backend for this model
(`results/attention_backends.md`).

## Licence

Both patches modify vLLM, which is licensed under Apache-2.0; the patched files
keep vLLM's licence headers.
