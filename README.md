# Fast single-user decoding for Dory1.0-4B in vLLM

This repository speeds up single-user decoding of
[NVIDIA-Nemotron-Labs-Dory1.0-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4)
on one RTX 4090 using [vLLM](https://github.com/vllm-project/vllm). It builds on
the Dory support in [vllm#60149](https://github.com/vllm-project/vllm/pull/60149).

Dory is a looped (latent-recurrent) language model. For each token, a shared
block of 20 attention/MLP layer pairs runs `n_recurrent_loops` times between
an 8-pair input block and an 8-pair output block. More loops can improve output
quality at the cost of more computation. This work reduces decoding time with:

- a **fused single-token decode path**: custom NVFP4 matrix-vector (GEMV),
  attention and sublayer kernels for one-token steps;
- an **INT8 LM head** that computes single-token logits using 8-bit weights;
- **wavefront decoding**: the model's own loops draft its next tokens, which
  keeps several tokens in flight at different loop depths;
- **Dory's shared KV cache**: one key/value cache per recurrent layer shared by
  all loops (the WaveFront paper's cross-recurrence KV sharing), which makes
  drafts match more often and holds about 2.7 times as many tokens at 4 loops;
- the **vLLM changes** that let a model run its own decode step inside the V2
  model runner without changing batched serving.

The table shows output tokens per second for one user with the NVFP4 checkpoint.
Input/output gives the number of prompt and generated tokens. Each column adds
one feature, with the gain over the previous column in parentheses. Triton
attention is an existing vLLM setting (`--attention-backend TRITON_ATTN`);
the later columns measure the changes in this repository:

<!-- table:headline -->
| Loops | Input/output | vLLM as shipped | + Triton attention | + fused decode path | + INT8 LM head | + wavefront | + shared KV cache | Total |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128/1024 | 188.2 | 234.8 (+25%) | 286.2 (+22%) | 318.8 (+11%) | n/a | n/a | +69% |
| 1 | 2560/300 | 152.2 | 199.1 (+31%) | 241.7 (+21%) | 264.9 (+10%) | n/a | n/a | +74% |
| 2 | 128/1024 | 127.7 | 161.9 (+27%) | 200.3 (+24%) | 215.3 (+7%) | 251.0 (+17%) | 264.1 (+5%) | +107% |
| 2 | 2560/300 | 103.3 | 136.3 (+32%) | 167.2 (+23%) | 177.3 (+6%) | 197.9 (+12%) | 208.7 (+5%) | +102% |
| 4 | 128/1024 | 78.0 | 99.8 (+28%) | 125.7 (+26%) | 131.1 (+4%) | 194.2 (+48%) | 243.4 (+25%) | +212% |
| 4 | 2560/300 | 62.5 | 83.4 (+33%) | 104.0 (+25%) | 107.7 (+4%) | 143.9 (+34%) | 168.2 (+17%) | +169% |
<!-- /table -->

The headline measurements disable prefix caching, so each benchmark processes
its own prompts. The full suite covers 1, 2, 4, 8, 16, 32 and 64 concurrent users;
[Benchmarks](docs/BENCHMARKS.md) also labels the older backend, quantization and
development-history tables that used prefix caching.

These measurements use random-token prompts, which are harder to draft than
the chat prompts tested here. Separate in-process tests on chat prompts at 4
loops reached 259 tok/s with sampling and 265 tok/s with greedy decoding on the
shared KV cache (240 and 261 tok/s with per-loop caches).

With 16 and 64 users, vLLM continues to use its batched kernels. The fused path
keeps an extra copy of the weights, leaving less memory for the key/value (KV)
cache. This costs about 2 to 3% throughput in the 4-loop, 64-user, 128/1024
case, where the cache fills up. The shared KV cache more than makes up for
it: at 4 loops it holds about 2.7 times as many tokens at the same weight layout.
In the corrected wavefront runs, shared KV raises 64-user throughput by 20%
at 128/1024 and 14% at 2560/300 over per-loop KV. Earlier long-prompt results
were inflated by reuse of prompt prefixes across benchmarks; the corrected
suite disables that cache. See [Benchmarks](docs/BENCHMARKS.md) for the full
results and caveats.

## What was built

### 1. Fused single-token decode path

A one-token decode step is limited by the time it takes to read the weights.
Unmodified vLLM also launches hundreds of small kernels around those reads.
The fused decoder (`kernels/dory_fused_model.py`, enabled by
`DORY_FUSED_DECODE`) combines these operations into fewer kernels. vLLM still
handles prompt processing (prefill), batching and the KV cache:

- **NVFP4 GEMV kernels** (`kernels/w4a16_gemv.py`): Triton kernels that read
  the checkpoint's packed 4-bit weights and FP8 scales directly and take the
  activations in a "nibble-major" FP32 layout matching the packed weights, so
  each weight byte is used without shuffling. They replace vLLM's Marlin GEMMs
  for one token.
- **Fused sublayers** (`kernels/dory_attention.py`, `kernels/dory_mlp.py`): the
  q/k/v/gate projection with RoPE, the gated output and the SwiGLU MLP in a few
  kernels each, with the model's nGPT residual update (normalize, interpolate,
  normalize) folded into each sublayer's last GEMV.
- **Decode attention** (`kernels/dory_decode_attention.py`): a split-KV kernel
  on vLLM's paged KV cache that normalizes q and k, writes the new key and
  value into the cache, applies the sigmoid output gate and writes its result
  in the next GEMV's input layout. It replaces vLLM's attention, KV cache write
  and q/k normalization kernels for one-token steps.
- **INT8 LM head** (`kernels/dory_lm_head.py`): the 131072 x 2560 output
  projection stored with one byte per weight and a row scale that absorbs
  Dory's logit scale, halving the largest single read of the step. It passed
  the greedy-agreement check, and the GSM8K and MMLU tests found no
  statistically significant accuracy difference.

**Gain for one user:** the fused kernels add 21 to 26% over vLLM with Triton
attention at 1, 2 and 4 loops, and the INT8 head another 3 to 11% (more at
fewer loops, where the head is a larger share of the step). At 2 loops the
decode step goes from 6.32 ms to 4.70 ms, against a bandwidth minimum of
3.70 ms ([experiment log](EXPERIMENTS.md), E0 to E12).

### 2. Wavefront decoding

[WaveFront Decoding](https://arxiv.org/abs/2609.23033)
([code](https://github.com/summerbro-hhj/wavefront-decoding)) uses a looped
model's early predictions to draft upcoming tokens. Several tokens are processed
together at different loop depths: the newest token's loop-1 prediction drafts
the next token, while older tokens finish their loops and verify earlier drafts.
Each step reads the recurrent block's weights once for all tokens in flight
instead of once per loop. No separate draft model is needed.

This implementation (`kernels/dory_wavefront_engine.py`) runs inside vLLM as
the model's own decode step whenever one request decodes alone:

- **Integration with ordinary decode steps.** vLLM's speculative decoding
  verifies K drafts against K + 1 full-depth predictions in one step; the
  wavefront makes one full-depth prediction per step. vLLM therefore keeps
  running ordinary one-token steps and samples every token itself. At 4 loops,
  a steady step advances four in-flight tokens by one loop in about 3.6 ms,
  compared with 7.75 ms for a sequential step. When vLLM samples a token that
  differs from the draft, a catch-up step refills the pipeline.
- **Drafts that follow vLLM's sampling.** Each draft is drawn with the same
  Gumbel noise that vLLM's sampler will use at that position (vLLM's own
  functions, the request's seed), so with temperature 1 and top-p 0.95 vLLM
  picks the drafted token 94.5% of the time on chat prompts. Outputs are
  always vLLM's own samples from the full-depth logits; penalties, logprobs and
  structured output work unchanged.
- **Multi-token kernels** for the in-flight tokens: tensor-core NVFP4 GEMMs for
  up to 16 tokens (`kernels/dory_multi_token.py`), and the decode attention
  kernel extended so each token uses its own loop's cache and its own block
  table, because vLLM places one layer's loop caches in different KV cache
  groups.
- **Accuracy checks.** In the greedy tests, outputs matched those computed
  from scratch with the same kernels. Differences from unmodified vLLM stayed
  within the accuracy check's noise floor.

**Gain for one user** over the fused path with the INT8 head: 12 to 17% at 2
loops and 34 to 48% at 4 loops on the benchmark's random-token prompts; at 4
loops on chat prompts, 77% with temperature 1 and top-p 0.95 and 91% greedy.
It needs at least two loops.

The design, the research behind it (WaveFront, LoopSpec, LoopAhead) and the
options considered are in [LOOP_SPECULATION.md](LOOP_SPECULATION.md).

### 3. Dory's shared KV cache

By default each loop of the recurrent block keeps its own key/value cache, so
at 4 loops the 20 recurrent attention layers need 80 caches. Dory can instead
keep one cache per recurrent layer that every loop overwrites
(`recurrent_kv_cache_mode: "last_loop"`). This is what the
[WaveFront paper](https://arxiv.org/abs/2609.23033) (section 3.3) calls
cross-recurrence KV sharing; it treats it as an optional, approximate
extension that raised draft acceptance on Huginn-3.5B from 0.88 to 0.98.

- **Quality, measured first on unmodified vLLM:** at 4 loops the shared cache
  leaves GSM8K (91.2% against 90.8%) and MMLU (62.7% against 62.3%) unchanged.
  Its greedy outputs differ from per-loop caching more than switching attention
  backends does, as a different cache semantics should.
- **The fused path and the wavefront on one cache:** in-flight tokens rewrite
  their own cache entry at every loop while younger tokens in the same kernel
  launch read it. The multi-token attention kernel therefore caches every
  in-flight token's new key and value first and then attends, so each token
  reads the older in-flight tokens at their latest loop. Sequential decoding
  would read them after their last loop, so on a shared cache the wavefront
  approximates sequential decoding rather than matching it. Against unmodified
  vLLM on the same cache it stays within the accuracy check's noise floor
  (log-probability drift 0.0066 against 0.0072 for switching attention
  backends), and GSM8K (92.0% against 91.2%) and MMLU (62.7%) are unchanged.
- **Gain:** drafts match more often (vLLM samples 96% of them on random-token
  prompts, against 87 to 92% with per-loop caches, and 97% against 94.5% on
  chat prompts with sampling). In the corrected suite, at 4 loops one user
  reaches 243.4 tok/s at 128/1024 and 168.2 at 2560/300, 25% and 17% over
  the wavefront with per-loop caches and 3.1 and 2.7 times vLLM as shipped.
  At 2 loops it reaches 264.1 and 208.7 tok/s, about 5% over per-loop wavefront.
  The cache holds
  151,915 tokens with the fused path (56,966 per-loop), so the 64-user rows gain
  20% (128/1024) and 14% (2560/300) against the per-loop wavefront.

Turn it on with `"recurrent_kv_cache_mode": "last_loop"` in `--hf-overrides`
(see Reproduce). Per-loop caches remain available and keep the wavefront exact
against sequential decoding; which to serve is a deployment choice. Details:
EXPERIMENTS.md E24 and E25, docs/BENCHMARKS.md section 5.

### 4. vLLM changes

[`patches/vllm-dory-fused-decode.patch`](patches/README.md) (6 files, against
the head of vllm#60149) adds:

- the hooks the fused path needs in vLLM's Dory model, off unless
  `DORY_FUSED_DECODE` is set;
- `wavefront_decode` in Dory's config, which asks the scheduler for
  `n_recurrent_loops - 1` lookahead KV slots per request, reserved only while
  at most one request runs, so batches keep their KV cache;
- in the V2 model runner: single-request, single-token steps run outside its
  FULL CUDA graphs (the engine replays its own graphs), and the runner tells
  the model each step's request and gives it the sampler and block tables. The
  engine writes ahead only where the request's blocks are allocated, because a
  block-table row keeps stale entries past its allocation.

The V1 model runner's custom-proposer hook was also tested. It disables
asynchronous scheduling, reducing single-user throughput by 11 to 17%, so this
implementation uses V2 ([experiment log](EXPERIMENTS.md), E21). Those tests
also exposed a warm-up failure when attention metadata was absent; the fused
path now handles those runs.

### 5. Measurement tools

- `bench/run_baseline.py`: the serving matrix (formats, loops, two prompt
  shapes, 1/2/4/8/16/32/64 users) on an idle GPU, with manifests of every setting.
- `bench/profile_decode.py` and `bench/gap_analysis.py`: per-kernel decode
  profiles against the bandwidth minimum.
- `bench/compare_outputs.py`, `bench/common_prefix_delta.py` and
  `bench/eval_tasks.py`: the accuracy gate (greedy agreement and
  log-probability drift against a noise floor) and GSM8K/MMLU checks.
- Studies: attention backends (with a FlexAttention fix,
  `patches/vllm-flex-attention-pr57300-port.patch`), quantizations, the cost of
  kernel launches (E13), vLLM's n-gram speculation (E14), how often Dory's
  shallow and full-depth predictions agree (E15), and the V1 against V2 runner
  (E21).

## Reproduce

You need Linux, an RTX 4090, `uv` and about 30 GB of disk space. The kernels
are tuned for the RTX 4090.

```shell
bench/setup_vllm.sh                      # vLLM at vllm#60149's head, into .venv
bench/download_models.sh                 # BF16, MXFP8 and NVFP4 checkpoints
git -C src/vllm apply ../../patches/vllm-dory-fused-decode.patch

# Serve with the fused path and wavefront decoding at 4 loops. "per_loop"
# keeps one KV cache per loop (exact); "last_loop" shares one (faster, 2.7x the
# KV capacity, approximate within the accuracy gate's noise floor).
DORY_FUSED_DECODE=$PWD/kernels VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/vllm serve models/nvfp4 --attention-backend TRITON_ATTN \
  --async-scheduling --hf-overrides \
  '{"n_recurrent_loops": 4, "recurrent_kv_cache_mode": "last_loop", "wavefront_decode": true}'

# The benchmarks of docs/BENCHMARKS.md.
DORY_FUSED_DECODE=$PWD/kernels .venv/bin/python bench/run_baseline.py \
  --variants nvfp4 --loops 2 4 --attention-backend TRITON_ATTN --wavefront \
  --kv-cache-mode last_loop
.venv/bin/python -I bench/compare_outputs.py --loops 4 nvfp4-triton \
  nvfp4-triton-fused-wavefront nvfp4-triton-fused-int8-head nvfp4-flashinfer
```

- Without `DORY_FUSED_DECODE` the patched vLLM behaves as before; without
  `wavefront_decode` the fused path runs alone. `DORY_LM_HEAD_8BIT=none`
  restores vLLM's BF16 LM head.
- The serving benchmark disables prefix caching by default; `--prefix-caching`
  restores it for workloads intended to measure prompt reuse.
- Drafts match vLLM's samples only with vLLM's own Gumbel sampler
  (`VLLM_USE_FLASHINFER_SAMPLER=0`); with FlashInfer's sampler the outputs are
  still correct, but drafts match less often.
- FlashInfer's attention backend compiles its kernels on first use;
  `bench/setup_cuda_toolkit.sh` installs a CUDA compiler into `cuda/` for it.
- The `transformers` warning about an "incorrect regex pattern" is a false
  positive (the model's `config.json` has no `transformers_version`).

## Repository layout

- `kernels/`: the fused decode path, the wavefront engine and their kernel
  benchmarks (`bench_*.py`); `dory_wavefront.py` is the stand-alone wavefront
  prototype used to develop and validate the schedule outside vLLM.
- `bench/`: setup scripts, the serving, profiling and accuracy harnesses.
- `patches/`: the vLLM patch and what it applies to; `vllm-patched/` holds the
  six patched vLLM files in full.
- `results/`: every run behind the numbers here (serving metrics, manifests,
  profiles, accuracy reports). Raw profiler traces, server logs and per-token
  log-probability dumps are not included.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): backends, quantizations,
  feature-by-feature results, chat against random text, limits, accuracy.
- [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md): what is left and what blocks it.
- [EXPERIMENTS.md](EXPERIMENTS.md): the experiment log, one entry per change
  (goal, change, measurement, decision), including what did not work.
- [LOOP_SPECULATION.md](LOOP_SPECULATION.md): research and design notes for
  speculation with Dory's loops.

## Acknowledgements

NVIDIA provides the Dory checkpoints. This work builds on the Dory support
contributed in [vllm#60149](https://github.com/vllm-project/vllm/pull/60149).
The wavefront schedule follows [WaveFront Decoding](https://arxiv.org/abs/2609.23033);
[LoopSpec](https://arxiv.org/abs/2609.17184) informed the design.
