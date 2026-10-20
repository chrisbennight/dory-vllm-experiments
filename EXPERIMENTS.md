# Experiment log

This log records experiments on the fused single-token decode path and, later,
wavefront decoding. Each entry gives the goal, change, measurement, result and
decision as recorded at the time. Later entries may supersede earlier decisions.
Commit hashes refer to unpublished development history; the current code
includes the retained changes, and the measurements are in `results/`.

## Measurement protocol

- **Primary metric:** single-user decode step time from
  `bench/profile_decode.py` (in-process vLLM, NVFP4, 2 loops, Triton attention,
  640-token prompt, 63 decode steps) with `DORY_FUSED_DECODE` set and the
  patched vLLM checkout, followed by `bench/gap_analysis.py`.
- **Kernel-level checks:** `kernels/bench_*.py` and `kernels/explore_gemv.py`
  for isolated kernels on the real weights, in decode order inside CUDA graphs.
- **Confirmation:** `bench/run_baseline.py` serving benchmark after a batch of
  changes.
- **Priorities:** single-user decode speed is the target. Batched serving must
  not regress: confirmation runs include the 16- and 64-user rows, which must
  stay within 2% of unmodified vLLM, and every confirmation records the KV cache
  size (the fused weight copies take GPU memory that would otherwise hold KV
  cache). Changes that touch multi-token steps (follow-up 4) replace vLLM's
  path only at batch sizes where they are measured to be faster, with added
  2-, 4- and 8-user benchmark points.
- **One GPU job at a time.** Benchmarks and the serving harness refuse to start
  on a busy GPU, but nothing stops a second job starting later, so runs are
  strictly sequential. A scratch probe started during a serving run on
  2026-10-18 invalidated that run, which was discarded and repeated.
  Graphics programs are not listed as compute processes: later that day a
  background load of about 30% with 670 MiB in use passed the idle checks and
  slowed the fused decode step from 4.70 to 6.14 ms. That run was discarded,
  and every idle check (`gpu_busy_reason` in `bench/run_baseline.py`) now also
  requires memory use of at most 1 GiB and an average load of at most 10%
  over 5 seconds. The check runs before each server, so it cannot see load
  that starts during a run.
- **Accuracy gate** for changes that alter numerics:
  `bench/compare_outputs.py` (256 greedy tokens per prompt) against unmodified
  vLLM with Triton attention. The noise floor is FlashInfer versus Triton
  attention on unmodified vLLM. A change passes when no divergent token falls
  outside the reference's top 5 and its mean absolute log-probability
  difference over the positions it shares with the floor configuration
  (`bench/common_prefix_delta.py`) is at or below the floor's.
  - Until E9 the gate used 16 prompts and also required the floor's mean
    shared-token count (111). E2+E3, E5, E7 and E9 have nearly identical
    numerics yet scored 144, 118, 153 and 107 mean shared tokens: one 1-ulp
    BF16 difference eventually flips a near-tie, after which two greedy
    generations share nothing. With 16 prompts that count is noise, so from E9
    on the gate uses 64 prompts and reports shared tokens without gating on
    them.

## Starting point

Results before this log, all NVFP4, 2 loops, single user, RTX 4090:

| Stage | Decode step at 640 tokens | Serving, 128 in / 1024 out | Commit |
|---|---:|---:|---|
| vLLM, FlashAttention (default backend) | 8.05 ms | 128 tok/s | b9f2f38 and earlier |
| vLLM, Triton attention | 6.32 ms | 162 tok/s | |
| First fused decode path | 5.41 ms | 189 tok/s | b9f2f38 |
| Bandwidth minimum | 3.70 ms | | |

Fused path, gap to the minimum at 640 tokens: weight GEMVs 0.825 ms, attention
0.398 ms, normalization and residual 0.296 ms (residual update 112 x 2.0 us =
0.223 ms), sampling and bookkeeping 0.103 ms, idle 0.053 ms, output projection
0.036 ms.

## Summary of the hill-climb (2026-10-18)

| Stage | Decode step at 640 tokens | Single user, 128 in / 1024 out | Single user, 2560 in / 300 out (per user) |
|---|---:|---:|---:|
| vLLM unmodified, Triton attention | 6.32 ms | 161.7 tok/s | 152.4 tok/s |
| First fused decode path | 5.40 ms | 189 tok/s | |
| After E2 + E3 (residual fold, fused KV cache write) | 5.30 ms | | |
| After E5 (own decode attention kernel) | 5.09 ms | 199.2 tok/s | 191.7 tok/s |
| After E9 (q/k normalization inside attention) | 5.05 ms | 200.3 tok/s | 192.6 tok/s |
| E9 + INT8 LM head (E7; the default since E12) | 4.70 ms | 215.7 tok/s | 206.7 tok/s |
| Bandwidth minimum (BF16 LM head) | 3.70 ms | | |

Serving numbers are from
[results/confirmation_final.md](results/confirmation_final.md). With 16 and 64
users, every configuration stays within 0.7% of unmodified vLLM. The fused
weight copy costs 10.7% of the KV cache (12.5% with the INT8 LM head).

Follow-ups planned with the first fused path: 1 (residual fold) gave 0.03 ms instead of the hoped-for
0.22 ms (E2); 2 (attention-output input) was solved by our own attention kernel
(E5) after GEMV-side attempts failed (E1, E1b, E1c); 3 (KV cache write) gave
0.06 ms (E3) and now lives inside the attention kernel (E9); 4 (multi-token
GEMVs) lost to Marlin from two tokens on and is parked (E8).

What remains at 5.05 ms: the weight GEMVs are 0.90 ms above their bandwidth
minimum, mostly a fixed cost per kernel (launch, ramp-up and tail across 224
GEMV launches, plus the residual-fold tails), and attention 0.27 ms, bound by
dependent loads at short contexts. Each further per-kernel tweak tried (E6, E9,
E10, E11) moved the step by 0.03 ms or less. The next large step would remove
kernel boundaries altogether: a persistent kernel that streams the next
weights while waiting on the previous layer's result.

At 4 loops, the core use case, a request decoding alone runs wavefront
decoding inside vLLM (E22): 184.5 to 201.9 tok/s at 128/1024 against 131.1
for the fused path and 99.6 for unmodified vLLM, with batched serving
unchanged.

## Experiments

### E0: baseline noise (2026-10-18)

- **Goal:** know how small a difference the primary metric can resolve.
- **Change:** none; fused path as of b9f2f38, patched vLLM checkout.
- **Measured:** two fresh `profile_decode.py` runs (640-token prompt).
- **Result:** 5.41 ms per step in both (185.0 and 184.9 tokens/s); together
  with the 5.41 ms run behind the step-4 numbers, run-to-run spread is below
  0.1%.
- **Decision:** differences of about 0.02 ms per step or more are real.

### E1: natural-order input via 3D decode (2026-10-18) - rejected

- **Goal:** let GEMVs read their input in natural order, which would remove the
  strided input reads of the attention-output GEMV (follow-up 2, 9.6 us at 61%
  in vLLM) and the nibble-major copies everywhere.
- **Change (exploration only):** `_natural_3d_kernel` in `kernels/explore_gemv.py`
  loads a contiguous [BLOCK_W, 8] input tile and decodes each weight word into
  a [BLOCK_N, BLOCK_W, 8] tile with per-lane shifts, reducing over the last axis.
- **Measured:** `kernels/explore_gemv.py`, real weights, decode order, best
  launch configuration per shape
  ([results/gemv_explore_20261018T173012Z.md](results/gemv_explore_20261018T173012Z.md)).
- **Result:** much slower than the current nibble-major kernel on every shape
  (us per call):

  | Projection | Current (nibble-major) | Natural FP32 | Natural BF16 | Natural BF16 x gate | Current strided gated read |
  |---|---:|---:|---:|---:|---:|
  | output gate 2560x4096 | 8.3 | 12.8 | 12.8 | 13.1 | 10.2 |
  | attention output 4096x2560 | 8.0 | 13.5 | 12.6 | 13.6 | 9.0 |
  | MLP gate+up 2560x14336 | 24.2 | 35.0 | 35.3 | 35.6 | 25.5 |
  | MLP down 7168x2560 | 12.9 | 21.6 | 20.7 | 21.9 | 14.2 |

- **Decision:** rejected. Triton's code for the 3D decode and last-axis
  reduction is poor. Next (E1b): keep the 2D kernel structure and instead
  permute the weights at load time so the input's natural order becomes
  contiguous per nibble stream.

### E1b: natural-order input via segment-major weights (2026-10-18) - rejected

- **Goal:** the same as E1, without the 3D decode. The weights are reordered at
  load time so that nibble j of weight word w multiplies `x[j * K/8 + w]`. Each
  of the eight nibble streams then reads a contiguous stretch of the input in
  natural order, and the 2D kernel structure stays the same.
- **Change (exploration only):** `_segmented_kernel` and `_segment_major` in
  `kernels/explore_gemv.py`. Scale groups of 16 inputs become 16 consecutive
  words of one segment, so each scale still covers whole words.
- **Measured:** `kernels/explore_gemv.py --variants wide_f16_xperm segmented_f32
  segmented_bf16 segmented_bf16_gated`
  ([results/gemv_explore_20261018T173313Z.md](results/gemv_explore_20261018T173313Z.md)).
- **Result (us per call):**

  | Projection | Current (nibble-major) | Segment-major FP32 input | Segment-major BF16 input | Segment-major BF16 x gate |
  |---|---:|---:|---:|---:|
  | output gate 2560x4096 | 8.3 | 8.5 | 9.0 | 9.3 |
  | attention output 4096x2560 | 8.0 | 8.9 | 9.0 | 9.5 |
  | MLP gate+up 2560x14336 | 24.2 | 25.9 | 25.6 | 26.6 |
  | MLP down 7168x2560 | 12.9 | 14.4 | 14.8 | 15.2 |

- **Decision:** rejected. The gated segment-major attention output (9.5 us) is
  no better than today's strided gated read (9.0 us), and the other shapes get
  slower. Eight separate input streams per program cost about as much as the
  strided reads they replace.

### E1c: nibble-major output gate (2026-10-18) - rejected

- **Goal:** halve the strided reads of the attention-output GEMV. The fused
  q/k/v + gate kernel would write the output gate already in nibble-major
  order, so only the attention output is read with a stride.
- **Change (exploration only):** `_gate_nm_kernel` in `kernels/explore_gemv.py`.
- **Measured:** `kernels/explore_gemv.py --variants wide_f16_xperm gated_strided
  gate_nibble_major`
  ([results/gemv_explore_20261018T173441Z.md](results/gemv_explore_20261018T173441Z.md)).
- **Result:** attention output 9.3 us with a nibble-major gate versus 9.0 us
  with today's strided read of both inputs (minimum 5.9 us; 7.9 us with a fully
  nibble-major input).
- **Decision:** rejected. Mixing a strided read with a contiguous read is no
  faster than two strided reads. Follow-up 2 can only gain about 0.1 ms per
  token: it would need the attention kernel itself to write its output in
  nibble-major order, which means patching vLLM's Triton attention kernel. That
  is parked behind the larger items (E2 residual fold, KV cache write,
  multi-token GEMVs).

### E2: residual update folded into the output GEMVs (2026-10-18) - kept

- **Goal:** remove the separate residual-update kernel after each sublayer
  (follow-up 1: 112 launches of about 2.0 us, 0.22 ms per token).
- **Change:** `residual_epilogue` and `residual_inputs` in
  `kernels/dory_common.py`, used by `_down_normalized_kernel` (MLP) and
  `_o_gated_normalized_kernel` (attention) behind a `FUSE_RESIDUAL` constexpr.
  Each program stores its rows of b and five partial sums (|a|^2, |b|^2 and the
  three terms of |z|^2) and increments an arrival counter. The last program
  to arrive combines the partial sums in program order, so results are
  deterministic, and writes h. `DORY_FOLD_RESIDUAL` (default `mlp,attention`)
  selects the sublayers for A/B runs; with the fold, the MLP down GEMV uses 4
  warps instead of 8.
- **Iterations (sublayer benchmarks, us per sublayer, residual separate vs folded):**

  | Version | MLP | Attention (without the attention kernel) |
  |---|---:|---:|
  | Partial sums [programs, 8], loads after the GEMV loop | 43.0 vs 40.8 | 30.5 vs 32.7 |
  | Partial sums [5, programs], a and alpha loaded before the GEMV loop, final-pass loads issued together | 40.1 vs 38.9 | 30.6 vs 31.2 |

  A probe (`build/`, not kept) split the folded cost for the attention-output
  GEMV: 0.3 us for the per-program partial sums and atomic, 1.6 us for the last
  program's final pass, versus 1.8 to 2.0 us for the separate kernel including
  its launch gap. The final pass is a chain of dependent memory round trips on
  one SM, so it costs about as much as a separate one-program kernel. Writing
  the nibble-major copy through a [H/8, 8] tile (coalesced stores) made the
  separate residual kernels 0.2 us faster and is kept there. Inside the fused
  epilogue it slowed the whole GEMV by up to 0.9 us: the transpose's shared
  memory lowers occupancy for every program. The epilogue therefore keeps
  one-dimensional stores.
- **Measured in vLLM:** `bench/profile_decode.py` (640-token prompt), vLLM's KV
  cache write:

  | Fold | Decode step | Residual and normalization kernels | MLP down GEMV | Attention-output GEMV | Profile |
  |---|---:|---:|---:|---:|---|
  | none | 5.40 ms | 0.294 ms | 13.2 us | 9.7 us | `profile_20261018T174419Z_*` |
  | MLP | 5.37 ms | 0.188 ms | 14.8 us | 9.6 us | `profile_20261018T174439Z_*` |
  | attention | 5.38 ms | 0.189 ms | 13.2 us | 11.2 us | `profile_20261018T174546Z_*` |
  | both | 5.37 ms | 0.082 ms | 14.8 us | 11.3 us | `profile_20261018T174527Z_*` |

- **Result:** -0.03 ms per token (0.6%), far below the 0.22 ms the separate
  kernels took: the folded final pass adds about 1.5 us to each output GEMV.
- **Decision:** kept for both sublayers (same step time as MLP only, 112 fewer
  launches). Getting the rest needs the final pass off the critical path,
  which needs the next kernel to start before this one ends. Ada has no
  programmatic dependent launch, so this is parked.

### E3: KV cache write in the fused q/k normalization kernel (2026-10-18) - kept

- **Goal:** drop vLLM's separate cache-write kernel (follow-up 3, 56 launches of
  about 1.1 us).
- **Change:** `FusedDoryDecoder._paged_cache` in `kernels/dory_fused_model.py`.
  With vLLM's Triton attention backend and a BF16 cache, the decoder takes
  the layer's cache and slot mapping from vLLM's forward context. It writes k
  and v with `_qk_norm_cache_kernel` (now skipping padding slots, slot < 0, as
  vLLM does) and calls `unified_attention_with_output` directly. Other backends
  keep vLLM's attention layer. `DORY_FUSED_KV_WRITE=0` disables it for A/B
  runs. Numerics are unchanged: the same BF16 k and v reach the cache.
- **Measured in vLLM:** decode step 5.40 to 5.34 ms without the fold
  (`profile_20261018T174623Z_*`), 5.37 to 5.32 ms with the MLP fold
  (`profile_20261018T174604Z_*`). Attention including the cache write: 0.552 to
  0.486 ms.
- **Result:** -0.06 ms per token (1.1%).
- **Decision:** kept.

### E2 + E3 combined (2026-10-18)

- **Measured:** 5.30 ms per token in two runs (`profile_20261018T174903Z_*`,
  `profile_20261018T174921Z_*`), against 5.40 ms before E2: -0.10 ms (1.9%).
- **Accuracy gate:** `bench/compare_outputs.py nvfp4-triton-fused --reference
  results/output_agreement_20261018T165329Z/outputs_nvfp4-triton.json` (the
  saved unmodified-vLLM generations; `--reference` is new and avoids regenerating
  the baseline). The saved generations are under `results/` but not committed
  (they are large), so this reuse works only on this machine.
  ([results/output_agreement_20261018T174936Z/report.md](results/output_agreement_20261018T174936Z/report.md)):
  5/16 identical, 144 mean shared tokens, no divergent token outside the
  reference's top 5. The report's mean |logprob difference| (0.0063) is taken
  over each configuration's own shared prefix, so a configuration that agrees
  longer accumulates later, noisier tokens. Over the 1541 positions that all
  configurations share with the reference, it is 0.0050, against 0.0060 for
  FlashInfer versus Triton (noise floor) and 0.0053 for the step-4 fused path.
  Passed.

### E4: 64 softmax segments for single-request decode (2026-10-18) - rejected

- **Goal:** attention is the largest gap after the GEMVs (0.33 ms after E3).
  vLLM's Triton backend splits a decode over 16 segments per KV head, which is
  64 programs for Dory's 4 KV heads on 128 SMs. On SM120 GPUs it already uses
  64 segments when the batch underfills the GPU.
- **Change (reverted):** extended that SM120 condition in
  `vllm/v1/attention/backends/triton_attn.py` to SM89; on the 4090 it selects 64
  segments for one request only.
- **Measured:** fused path 5.37 ms (`profile_20261018T174723Z_*`, against
  5.30 ms), unmodified vLLM 6.40 ms (`profile_20261018T174743Z_*`, against
  6.32 ms). The attention kernel itself went from 6.8 to 6.3 us, but the
  segment-reduction kernel from 1.9 to 3.2 us.
- **Decision:** rejected at this context length. The attention kernel reads
  the 2.6 MB of KV cache at about 38% of bandwidth either way, so the next step
  is a decode attention kernel of our own (E5) that can also write its output
  gated and in nibble-major order for the attention-output GEMV (follow-up 2).

### E5: gated decode attention kernel (2026-10-18) - kept

- **Goal:** close the attention gap (0.33 ms after E3) and address follow-up 2 by
  writing the attention output in the GEMV's input layout (the
  attention-output GEMV's strided gated read, 11.2 us with the folded residual).
- **Change:** `kernels/dory_decode_attention.py`. A split-KV ("flash decoding")
  kernel computes the single-token attention over vLLM's paged BF16 cache.
  There is one program per (KV head, split); each runs online softmax for the
  KV head's 4 query heads over a run of 16-token cache blocks. The split count
  is fixed (32), so CUDA graphs hold, and each program derives its block range
  from the sequence length in GPU memory. Splits without blocks exit at once.
  A combine kernel (one program per head and 64-dimension chunk) merges only
  the splits that hold blocks, multiplies by the output gate and writes
  nibble-major FP32. `_o_gated_normalized_kernel` gains a `GATED_NATURAL`
  constexpr: off, it reads that contiguous input like the other GEMVs (launch
  configuration 4 rows x 64 words x 4 warps, best of five tried in vLLM). The
  decoder takes the block table, sequence lengths, window and softmax scale
  from vLLM's attention metadata. It uses the kernel only for a plain causal
  single-token decode (no ALiBi, sinks, soft cap, chunked or prefix attention);
  everything else keeps vLLM's kernel. `DORY_FUSED_ATTENTION=0` disables it for
  A/B runs.
- **Kernel benchmark:** `kernels/bench_decode_attention.py`. Each of the 56
  attention layer applications has its own cache with a shuffled block table;
  the sequence runs in decode order in a CUDA graph. Results in
  [results/decode_attention_bench_20261018T175511Z.md](results/decode_attention_bench_20261018T175511Z.md),
  with the first sweep in
  [results/decode_attention_bench_20261018T175351Z.md](results/decode_attention_bench_20261018T175351Z.md).

  | us per attention layer application | 640 cached tokens | 2048 | 8192 |
  |---|---:|---:|---:|
  | vLLM unified attention + segment reduction (16 segments) | 8.6 | about 13.7 | about 34 |
  | New kernel + combine with gate (32 splits) | 6.8 | 9.6 | 16.2 |

  (The vLLM rows subtract the 2.1 us gate multiplication that the benchmark adds
  to make the outputs comparable; in the decoder that gate is part of the
  attention-output GEMV's input read.) Relative error against FP32 attention
  is 1.5e-3, against vLLM's 2.3e-3, for full and sliding-window layers at all
  three lengths. Sequence lengths 0, 1, 15-17, 1023-1025, 1040 and 3000 (around
  block and window boundaries) match the FP32 reference too. More splits (64,
  128), two or four blocks per tile and other pipelining depths were slower or
  equal. The combine fell from 2.0 to 1.6 us once it read only splits that
  hold blocks. The attention kernel stays at 5.2 us for 2.6 MB at 640 tokens:
  its chain of dependent loads (sequence length, block table, keys and values)
  bounds it there, not parallelism.
- **Measured in vLLM:** 5.09 ms per token in two runs
  (`profile_20261018T175704Z_*`, `profile_20261018T175742Z_*`), against 5.29 ms
  with `DORY_FUSED_ATTENTION=0` in the same session (`profile_20261018T175723Z_*`).
  Attention 0.487 to 0.386 ms; attention-output GEMV 11.2 to 9.9 us.
- **Accuracy gate:** against the saved unmodified-vLLM generations
  ([results/output_agreement_20261018T175926Z/report.md](results/output_agreement_20261018T175926Z/report.md)).
  2/16 identical, 118 mean shared tokens, no divergent token outside the
  reference's top 5, median divergence margin 0.056 (near ties). Over the 1231
  positions shared by every configuration (`bench/common_prefix_delta.py`), the
  mean |logprob difference| is 0.0046. That compares with 0.0055 for FlashInfer
  versus Triton, 0.0047 for the step-4 fused path and 0.0043 for E2 + E3.
  Agreement drops from E2 + E3 because a different attention kernel sums in a
  different order, as FlashInfer does; it stays within the noise floor. Passed.
- **Result:** -0.20 ms per token (3.8%) at 640 tokens; the gain grows with
  context (about -1 ms per token at 8192 cached tokens by the kernel benchmark).
- **Decision:** kept.

### E6: prefetching the attention-output weights into L2 during attention (2026-10-18) - parked

- **Goal:** attention and its combine take about 7 us per layer, using about 40%
  of DRAM bandwidth and fewer than 128 programs. A side-stream kernel could
  load the next GEMV's weights (5.2 MB for the attention output) into the 72 MB
  L2 meanwhile, so that GEMV reads them from L2.
- **Change (probe only, not kept):** 56 x (decode attention at 640 tokens, then
  the nibble-major attention-output GEMV) in a CUDA graph, with a prefetch
  kernel on a forked stream (`evict_last` loads), joined either before the GEMV
  or at the end of the graph.
- **Result (us per layer):** 15.5 without prefetch. The GEMV's second
  back-to-back call, with its weights in L2, costs 5.5 us, so at most about
  2.5 us per layer is available. With the prefetch: 14.6 with 32 programs,
  joined at the end; 15.6 with 16; 19.6 to 19.9 with 64 or 128, which compete
  with attention. Joining before the GEMV is always worse.
- **Decision:** parked. About 1 us per layer (0.05 ms per token) does not
  justify multi-stream capture inside vLLM's CUDA graph for now.

### Confirmation after E5: serving benchmark (2026-10-18)

- **Measured:** `bench/run_baseline.py --variants nvfp4 --loops 2
  --attention-backend TRITON_ATTN`, unmodified vLLM (`results/20261018T180516Z`)
  then the fused path (`results/20261018T181035Z`), in the same session on an
  idle GPU. The fused run's working tree also contained the opt-in FP8 LM head
  code (E7), switched off, which leaves the decode path unchanged.

  | Input/output | Users | Unmodified tok/s | Fused tok/s | Change |
  |---|---:|---:|---:|---:|
  | 128/1024 | 1 | 161.7 | 199.2 | +23.2% |
  | 128/1024 | 16 | 1721.1 | 1719.3 | -0.1% |
  | 128/1024 | 64 | 3145.9 | 3142.8 | -0.1% |
  | 2560/300 | 1 | 138.0 | 169.0 | +22.5% |
  | 2560/300 | 16 | 697.2 | 697.3 | 0.0% |
  | 2560/300 | 64 | 779.3 | 777.6 | -0.2% |

- **Batch guard:** passed; every 16- and 64-user row is within 0.2%.
- **KV cache:** 17.33 GiB (112,666 tokens) unmodified, 15.48 GiB (100,581
  tokens) fused. The fused decoder's own copy of the layer weights costs
  1.85 GiB, 10.7% of the KV cache. The 64-user, 2560-token-prompt run exceeds
  the cache in both configurations; with the smaller cache, fewer requests are
  admitted at once, so median time to first token rises from 3.6 s to 5.1 s
  (P99 21.8 s against 21.4 s), while per-user decode speed rises from 15.8 to
  17.5 tok/s. Removing the duplicate (for example, by having vLLM's batched
  path read the fused layout) would remove this cost; it is not scheduled.

### E7: 8-bit LM head for single-token logits (2026-10-18) - kept, opt-in

- **Goal:** the output projection is a BF16 131072 x 2560 matrix (671 MB) read
  once per token: 0.70 ms, 14% of the step, already at 95% of bandwidth. A
  smaller weight format would reduce that memory traffic.
- **Change:** `kernels/dory_lm_head.py`. `QuantizedLmHead` stores the head with
  one byte per weight and one FP32 scale per row, into which Dory's
  per-vocabulary logit scale (`logit_scale / initializer_range`) is folded.
  `_quantized_logits_kernel` computes FP32 logits for one hidden state.
  `DORY_LM_HEAD_8BIT=int8` (or `fp8`) enables it. The vLLM patch routes
  `compute_logits` for a single row to it, and leaves it to vLLM otherwise. It
  costs 335 MB of GPU memory (the BF16 head stays for multi-row logits).
  **Off by default:** unlike E1 to E5, this changes the precision of the
  model's own weights.
- **Kernel benchmark:** `kernels/bench_lm_head.py`, 64 random unit-norm hidden
  states, errors against FP32 math on the BF16 weights
  ([results/lm_head_bench_20261018T181957Z.md](results/lm_head_bench_20261018T181957Z.md)):

  | Head | Mean max abs logit error | Top-1 agreement | Mean KL divergence | Mean abs top-token logprob error | us per token |
  |---|---:|---:|---:|---:|---:|
  | vLLM BF16 (BF16 logits, then scaled) | 0.033 | 98% | 4.5e-05 | 0.015 | 783 |
  | INT8, per-row scale | 0.148 | 98% | 2.6e-04 | 0.022 | 356 |
  | FP8 E4M3, per-row scale | 0.335 | 91% | 2.3e-03 | 0.059 | 356 |

  FP8's 3 mantissa bits lose about 4% on every weight; INT8's uniform steps
  (amax / 127) are about 4 times finer for these rows. FP8 is not pursued.
- **Measured in vLLM:** 4.73 ms per token with the INT8 head
  (`profile_20261018T182021Z_*`), against 5.08 ms without
  (`profile_20261018T182056Z_*`): -0.35 ms (6.9%). The gap analysis shows the
  output projection at 186% "efficiency" because its minimum assumes the BF16
  head.
- **Accuracy gate:** `bench/compare_outputs.py nvfp4-triton-fused-int8-head
  --reference ...`
  ([results/output_agreement_20261018T182124Z/report.md](results/output_agreement_20261018T182124Z/report.md)):
  4/16 identical, 153 mean shared tokens, no divergent token outside the
  reference's top 5. Over the 1164 positions shared by every configuration, the
  mean |logprob difference| is 0.0051, against 0.0055 for FlashInfer versus
  Triton and 0.0047 for E5 without it. Passed: the head computes logits in FP32
  instead of rounding them to BF16 as vLLM does, which offsets much of the
  weight rounding.
- **Decision:** kept as an option, off by default, pending a decision on
  whether an 8-bit LM head is acceptable for this model.

### E8: multi-token GEMV for 2-8 tokens (2026-10-18) - rejected in this form

- **Goal:** follow-up 4. Replace vLLM's Marlin GEMMs for small decode batches
  only where a multi-token version of the fused GEMV is faster.
- **Change:** `gemv_rows_tokens`, `_w4a16_gemv_tokens_kernel` and
  `w4a16_gemv_tokens` in `kernels/w4a16_gemv.py`. Each weight word is decoded once and
  multiplied with all M tokens' nibble-major inputs in a
  [rows, tokens, words] tile, on CUDA cores. `kernels/bench_gemv_tokens.py`
  times it against Marlin (`MarlinProjection` now accepts [M, K] inputs) in
  decode order for M = 1, 2, 4, 8
  ([results/gemv_tokens_bench_20261018T182550Z.md](results/gemv_tokens_bench_20261018T182550Z.md)).
- **Result (us per call, Triton / Marlin):**

  | Projection | M=1 | M=2 | M=4 | M=8 |
  |---|---:|---:|---:|---:|
  | q/k/v 2560x6144 | 12.3 / 13.0 | 19.9 / 12.9 | 28.0 / 12.8 | 35.5 / 12.9 |
  | output gate 2560x4096 | 9.6 / 9.5 | 14.1 / 9.4 | 20.5 / 9.5 | 25.3 / 9.4 |
  | attention output 4096x2560 | 9.4 / 10.1 | 12.8 / 10.1 | 15.8 / 10.1 | 24.6 / 10.1 |
  | MLP gate+up 2560x14336 | 26.9 / 25.0 | 40.6 / 24.9 | 59.7 / 24.9 | 77.6 / 24.9 |
  | MLP down 7168x2560 | 15.1 / 14.6 | 20.9 / 14.6 | 26.2 / 14.6 | 42.5 / 14.6 |

  Marlin is flat from 1 to 8 tokens (tensor cores, weight-bound); the
  CUDA-core version grows with M and is 1.3 to 3.1 times slower from M = 2.
  Accuracy is the same as single-token (relative error 1.7e-3).
- **Decision:** rejected; vLLM keeps Marlin for batches. Beating Marlin at 2-8
  tokens needs a tensor-core dequantizing GEMM, which is what Marlin is. The
  remaining multi-token gain would come from fusing the non-GEMM work
  (normalization, residuals, rotary embedding, attention), which means a
  multi-token version of every fused kernel and of the vLLM dispatch, including
  batch-padding rows. That is a separate project with no single-user benefit;
  parked.

### E9: q/k normalization and KV cache write inside the decode attention kernel (2026-10-18) - kept

- **Goal:** remove the separate q/k normalization kernel (56 launches of
  1.3 us) now that our own kernel computes attention.
- **Change:** `DecodeAttention.from_raw` and the `RAW_QKV` constexpr in
  `kernels/dory_decode_attention.py`. Each attention program normalizes its
  query heads and applies the query scale itself. The split whose last tile
  holds the new token also normalizes the new key, writes the key and value
  into the cache at vLLM's slot, and substitutes them for that position from
  registers. `FusedAttentionSublayer._qkvg` became the public `project_qkvg`.
  Loading the inputs that do not depend on the sequence length first made no
  difference, and is kept as the clearer order.
- **Kernel benchmark:**
  [results/decode_attention_bench_20261018T183332Z.md](results/decode_attention_bench_20261018T183332Z.md).
  The new k and v are cached bit-exactly for full and sliding-window layers at
  640, 2048 and 8192 tokens; relative error against FP32 attention is 1.1e-3
  to 1.2e-3. Per layer application: 8.13 us for the normalization kernel plus
  attention, 7.74 us combined. The attention kernel itself grows by about
  0.9 us, most likely from the FP32 query tile (padded to 16 rows for the dot
  product) that the normalization needs.
- **Measured in vLLM:** 5.06, 5.05 and 5.05 ms per token
  (`profile_20261018T183408Z_*`, `profile_20261018T183427Z_*`,
  `profile_20261018T183509Z_*`), against 5.08 ms: -0.03 ms (0.6%).
  Normalization kernels 0.083 to 0.008 ms; attention 0.386 to 0.429 ms.
- **Accuracy gate:** on the 16-prompt set
  ([results/output_agreement_20261018T183525Z/report.md](results/output_agreement_20261018T183525Z/report.md)):
  107 mean shared tokens and 0.0064 own-prefix difference, both just outside
  the old floor (111, 0.0059). Over the 1128 common positions it scores 0.0054,
  against 0.0055 for the floor. Prompt by prompt, E9 agreed with the reference
  longer than E5 on five prompts and shorter on four, which is chaotic
  divergence at near-ties, not a bias. The gate was therefore rebuilt with 64
  prompts, regenerating the reference and the floor
  ([results/output_agreement_20261018T184021Z/report.md](results/output_agreement_20261018T184021Z/report.md);
  its `outputs_nvfp4-triton.json` is the reference from here on, kept locally).
  On 64 prompts, E9 scores 128 mean shared tokens (floor 121) and 0.0066
  own-prefix difference (floor 0.0069). Over the 6638 common positions it
  scores 0.0061 (floor 0.0067), with no divergent token outside the top 5.
  Passed.
- **Decision:** kept.

### E10: launch configurations re-tuned inside vLLM (2026-10-18) - no change

- **Goal:** the q/k/v + gate, gate+up and folded down GEMVs were tuned in
  isolated benchmarks before E2 to E9 changed what runs around them.
- **Measured:** 14 decode profiles in vLLM, one configuration changed at a time
  ([results/launch_config_sweep_20261018T1849Z.md](results/launch_config_sweep_20261018T1849Z.md)).
  q/k/v + gate: 5.05 to 5.13 ms against 5.06 ms for the default; gate+up 5.05 to
  5.11 ms; folded down 5.08 to 5.38 ms, with the default (4 rows, 64 words,
  4 warps) the fastest.
- **Decision:** keep the defaults; every alternative is within noise or slower.

### E11: earlier block-table read in decode attention (2026-10-18) - reverted

- **Goal:** shorten the attention kernel's chain of dependent loads (sequence
  length, then block table, then cache) and the combine kernel's (sequence
  length, then partial results).
- **Change (commit 0580293, reverted in the next commit):** splits take tiles
  round-robin, so a split's first tile does not depend on the sequence length
  when attention starts at token 0 (full attention, or a window not yet full).
  Each program loads that tile's block-table entry together with the sequence
  length and reloads it only when the window has moved. Empty splits store a
  maximum of -inf, so the combine kernel loads everything in one round trip
  and discards empty splits with selects.
- **Kernel benchmark:**
  [results/decode_attention_bench_20261018T185706Z.md](results/decode_attention_bench_20261018T185706Z.md).
  Raw-input attention 7.74 to 7.25 us per layer at 640 tokens, combine 1.59 to
  1.47 us. Edge cases (sequence lengths 0 to 9000 around block and window
  boundaries) match the FP32 reference.
- **Measured in vLLM:** 5.06 and 5.05 ms (`profile_20261018T185710Z_*`,
  `profile_20261018T185729Z_*`), against 5.05 to 5.06 ms for E9. Attention
  0.429 to 0.423 ms; the attention kernel stays at 6.1 us.
- **Decision:** reverted; no measurable gain in vLLM. The benchmark gives each
  layer its own block table, so the read misses L2; in vLLM all layers of a KV
  cache group share one block table, which stays in L2 after the first layer.

### Final confirmation: serving benchmark after E9 (2026-10-18)

- **Measured:** `bench/run_baseline.py --variants nvfp4 --loops 2
  --attention-backend TRITON_ATTN` with the fused path (`results/20261018T185825Z`)
  and with the fused path plus `DORY_LM_HEAD_8BIT=int8`
  (`results/20261018T190327Z`), compared with the unmodified run
  `results/20261018T180516Z` from earlier in the session
  ([results/confirmation_final.md](results/confirmation_final.md)). The serving
  harness now records the LM head setting in its manifest, and the summary
  labels it.
- **Single user:** 128/1024: 200.3 tok/s fused, 215.7 tok/s with the INT8 head,
  against 161.7 unmodified (+24%, +33%). 2560/300: 192.6 and 206.7 tok/s per
  user, against 152.4 (+26%, +36%).
- **Batch guard:** passed. Every 16- and 64-user row is within 0.7% of
  unmodified.
- **KV cache:** 100,581 tokens fused and 98,548 with the INT8 head, against
  112,666 unmodified.
- The vLLM checkout under `src/vllm` stays patched with
  `patches/vllm-dory-fused-decode.patch`. Without `DORY_FUSED_DECODE` it
  behaves as unmodified vLLM.

### E12: task-quality check of the INT8 LM head; INT8 becomes the default (2026-10-18) - kept

- **Goal:** E7's INT8 head passed the greedy-agreement gate, which says nothing
  about task accuracy. Check that before turning it on by default.
- **Measured:** `bench/eval_tasks.py nvfp4-triton-fused nvfp4-triton-fused-int8-head`
  ([results/task_eval_20261018T195326Z/report.md](results/task_eval_20261018T195326Z/report.md)).
  Samples come from `bench/fetch_eval_data.py`: 250 GSM8K test problems and
  1000 MMLU test questions over 42 subjects. The fused path, one request at a
  time, thinking disabled through the chat template. GSM8K is a greedy answer
  ending in "Answer: <number>", which exercises the whole fused decode path.
  MMLU is the best of A to D by log-probability of a one-token answer; prefill
  runs vLLM's kernels in both configurations, so only the head differs.

  | Head | GSM8K | MMLU |
  |---|---:|---:|
  | BF16 (vLLM) | 90.4% | 60.7% |
  | INT8 | 89.6% (4 problems right only with BF16, 2 only with INT8; McNemar p = 0.69) | 60.8% (p = 1.0) |

  On MMLU the two heads choose the same letter for 99.5% of questions; the
  log-probability of the BF16 head's letter moves by 0.005 on average (largest
  0.07).
- **64-prompt gate** with E9 and the INT8 head
  ([results/output_agreement_20261018T200401Z/report.md](results/output_agreement_20261018T200401Z/report.md)):
  17/64 identical, 136 mean shared tokens (floor 121), no divergent token
  outside the top 5. Over 6123 common positions: 0.0061, against 0.0065 for the
  floor and 0.0059 for the BF16 head. The FlashInfer floor regenerated
  identically, so the floor is deterministic.
- **Change:** `DORY_LM_HEAD_8BIT` now defaults to `int8`; `none` restores vLLM's
  BF16 head, and other values are rejected. In `bench/compare_outputs.py`,
  `nvfp4-triton-fused` pins the BF16 head, so it means what it meant in earlier
  results. The serving manifest and profile summaries record the head in
  effect.
- **Measured in vLLM with the defaults:** 4.70 ms per token
  (`profile_20261018T200931Z_*`).
- **Decision:** kept as the default. Batch impact: the 335 MB copy costs 2% of
  the KV cache (100,581 to 98,548 tokens), with 16- and 64-user throughput within
  0.7% of unmodified vLLM.

### E13: fixed cost per GEMV launch, the ceiling for a persistent kernel (2026-10-18) - measurement

- **Goal:** decide whether a persistent kernel is worth building, by measuring
  how much of the weight GEMVs' gap is the cost of kernel boundaries.
- **Measured:** `kernels/bench_launch_cost.py`
  ([results/launch_cost_20261018T200956Z.md](results/launch_cost_20261018T200956Z.md)).
  The same weight bytes (56 layers of each projection's shape, random values)
  go through the single-token GEMV in 56, 14, 4 or 1 launches, by stacking the
  layers' weights. A grid-wide barrier built from atomics, the synchronization a
  persistent kernel would use instead of a launch boundary, is timed with every
  program resident.

  | Projection | Minimum us | 56 launches | 14 | 4 | 1 | Cost per launch boundary |
  |---|---:|---:|---:|---:|---:|---:|
  | q/k/v + output gate | 14.6 | 17.9 | 16.2 | 15.8 | 15.6 | 2.3 us |
  | attention output | 5.9 | 9.2 | 6.7 | 6.4 | 6.3 | 3.0 us |
  | MLP gate+up | 20.5 | 24.7 | 22.5 | 22.0 | 21.9 | 2.9 us |
  | MLP down | 10.2 | 14.8 | 11.5 | 11.0 | 10.9 | 4.0 us |

  Grid-wide barrier: 0.51 us with 128 resident programs (one per SM), 0.78 us
  with 256, 1.46 us with 512.
- **Result:** in one long launch the existing GEMV kernels reach 94 to 95% of
  peak bandwidth, against 59 to 83% per layer today. The boundaries between the
  224 GEMV launches of a decode step cost up to 0.68 ms per token. A barrier
  costs 0.5 to 0.8 us where a boundary costs 2.3 to 4.0 us, so a persistent
  kernel could recover roughly 0.5 ms per token (about 11% of 4.70 ms), if each
  program starts loading its next weights before it waits at the barrier.
  The stacked launches have no data dependency between layers, so this is an
  upper bound on the overlap.

### E14: vLLM's prompt-lookup (n-gram) speculation at one user (2026-10-18) - measurement

- **Goal:** a no-new-code control for speculative decoding: how much does
  verifying several copied draft tokens per step gain on top of the fused path?
- **Change:** `nvfp4-triton-fused-ngram` in `bench/compare_outputs.py`: the
  default fused path plus `--speculative-config` with vLLM's GPU n-gram
  proposer (`ngram_gpu`; the CPU `ngram` method rejects async scheduling), up
  to 4 draft tokens, lookup of 2 to 4 tokens. Single-token steps keep the fused
  path; verification steps run vLLM's own kernels and BF16 head.
- **Measured:** `bench/spec_ngram.py nvfp4-triton-fused-int8-head nvfp4-triton-fused-ngram`
  ([results/spec_ngram_20261018T201600Z/report.md](results/spec_ngram_20261018T201600Z/report.md)),
  greedy, one request at a time, up to 512 tokens. "edit" asks for 8 functions
  of this repository back without comments (thinking off). "chat" is 16 chat
  prompts with thinking on.

  | Workload | Fused path | Fused + n-gram | Speed | Draft acceptance | Outputs identical |
  |---|---:|---:|---:|---:|---:|
  | edit | 212.5 tok/s | 662.0 tok/s | 3.12x | 88% of 2924 | 8/8 |
  | chat | 222.0 tok/s | 210.0 tok/s | 0.95x | 7% of 23,536 | 1/16 |

- **Result:** when the output copies its context, n-gram speculation triples
  single-user speed. In chat with thinking, the proposer still finds matches
  but only 7% of drafts are accepted, and the wasted verification costs 5%.
  The differing chat outputs come from numerics: verification steps use vLLM's
  kernels and BF16 head, so near-ties eventually resolve differently.
- **Decision:** not enabled yet. Before it can be, it needs a less eager
  proposer for chat (longer minimum lookup, fewer draft tokens) and a batch-guard
  run at 16 and 64 users. It also shows that a fast fused multi-token
  verification path would pay off directly.

### E15: agreement of Dory's shallow and full-depth predictions (2026-10-18) - measurement

- **Goal:** speculative decoding with Dory's own loops drafts each token from
  a shallow pass and verifies it at full depth. Its payoff depends on how often
  they agree on the same history, which no earlier measurement covered.
- **Method:** `bench/loop_agreement.py run`
  ([results/loop_agreement_20261018T202039Z/report.md](results/loop_agreement_20261018T202039Z/report.md)).
  It makes 2-loop greedy completions (256 tokens, thinking on) of the 64 gate
  prompts, then scores every completion position with 1-, 2- and 4-loop models
  (top-20 prompt log-probabilities). With per-loop KV caches a token's loop-1
  state does not depend on later loops, so the 1-loop model reading a history is
  exactly the draft of a scheme that keeps separate shallow output-block
  caches. The 4-loop comparisons reuse the 2-loop histories. The 4-loop scoring
  process aborted during vLLM's engine teardown after writing its results; the
  driver now accepts that only when the step's output file exists.
- **Result:** sanity check: the 2-loop model's top token matches its own
  greedy history at 99.3% of positions.

  | Draft loops | Verify loops | Top-token agreement | Distribution overlap (sampling acceptance, lower bound) | Mean tokens accepted from a 4-token greedy draft |
  |---:|---:|---:|---:|---:|
  | 1 | 2 | 94.2% | 0.944 | 3.45 |
  | 1 | 4 | 93.5% | 0.936 | 3.38 |
  | 2 | 4 | 97.9% | 0.976 | 3.77 |

- **What it implies:** cost estimates from the measured block times (input
  block 0.60 ms, recurrent block 1.50 ms per loop, output block 0.60 ms, INT8
  head 0.36 ms, other 0.14 ms; 4.70 ms per token sequentially). They assume
  verifying 5 tokens costs 1.3 times a single token, which needs a fast
  multi-token path (vLLM's Marlin GEMMs are flat in token count, E8; the
  surrounding work is not yet fused for several tokens).
  - 2 loops, draft 4 tokens at loop 1 (3.2 ms each, single-token fused path),
    then verify loop 2 onward for 5 tokens reusing the loop-1 work (about
    3.2 ms): 16.0 ms for 4.45 tokens, about 3.6 ms per token (-23%).
  - 4 loops (7.7 ms per token sequentially), same scheme: about 19.9 ms for 4.38
    tokens, about 4.5 ms per token (1.7 times faster).
  - Drafting dominates the cycle and runs single-token steps, so a persistent
    kernel (E13) would speed up the drafts too.
- **Decision:** agreement is well above the 90% threshold set in
  NEXT_DIRECTIONS.md, so speculative decoding with Dory's loops becomes the
  main research direction, ahead of the persistent kernel. The two approaches
  can be combined.

### E16: tensor-core GEMV for several tokens, the base of the wavefront schedule (2026-10-18) - kept

- **Context:** 4 loops is the core use case and the target is the wavefront
  schedule (LOOP_SPECULATION.md). Its recurrent block processes up to 4 tokens
  at different depths in one call, so it needs weight kernels whose cost stays
  flat in the number of tokens. 4-loop baselines at a 640-token prompt: fused
  path 7.71 ms per token (`profile_20261018T204305Z_nvfp4_loops4_*`), vLLM
  10.22 ms (`profile_20261018T204414Z_nvfp4_loops4_*`), bandwidth minimum
  5.85 ms.
- **Change:** `gemm_rows_tc`, `_w4a16_gemm_tc_kernel` and `w4a16_gemm_tc` in
  `kernels/w4a16_gemv.py`. Each weight tile is decoded once, multiplied by its
  block scale in FP16 (exact: an E2M1 value times an E4M3 scale needs at most 5
  significant bits), and multiplied with up to 16 tokens' inputs by `tl.dot`. It
  reads the fused path's existing weight layout, so it needs no extra weight
  memory. Optional split-K: several programs share one row's K and write FP32
  partial sums for the consumer to add, because 2560-row projections otherwise
  give too few programs (16-row tiles are the tensor-core minimum).
- **Measured:** `kernels/bench_gemv_tokens.py`
  ([results/gemv_tokens_bench_20261018T2051Z.md](results/gemv_tokens_bench_20261018T2051Z.md)),
  us per call in decode order, best configuration per shape and token count.
  The first version had no split-K: attention output 15.1 us and down 24.4 us,
  1.9 times the single-token kernel
  ([results/gemv_tokens_bench_20261018T204733Z.md](results/gemv_tokens_bench_20261018T204733Z.md)).

  | Projection | Single-token kernel, 1 token | Tensor core, 1 token | 4 tokens | 8 tokens | Marlin, any count |
  |---|---:|---:|---:|---:|---:|
  | q/k/v + output gate | 17.5 | 21.4 | 21.4 | 21.5 | n/a |
  | attention output | 7.9 | 10.8 | 10.9 | 11.0 | 10.1 |
  | MLP gate+up | 24.2 | 26.5 | 26.6 | 26.7 | 24.9 |
  | MLP down | 12.8 | 16.8 | 16.9 | 16.9 | 14.6 |

  Relative error 1.5e-3 to 1.9e-3, the BF16 output rounding of the other
  kernels.
- **Result:** one attention+MLP layer pair's weights cost 75.5 us for any
  number of tokens up to 8, against 62.4 us for one token on the single-token
  kernels: 1.21 times. That passes the Stage 1 gate of 1.3 times.
- **Decision:** kept as the weight kernel for the multi-token sublayers. With
  it, the wavefront schedule at 4 loops is estimated at about 4.2 ms per
  committed token (about 1.8 times sequential decoding). The step costs: input
  block for 1 token 0.60 ms; recurrent block for 4 tokens about 1.8 ms; output
  block and head for 2 tokens about 1.1 ms; at 93.5% acceptance each rejection
  costs about 3 refill steps.

### E17: multi-token attention and sublayers for the wavefront schedule (2026-10-18) - kept

- **Goal:** the remaining Stage 1 building blocks. The recurrent block must run
  up to 4 tokens at different loop depths in one call, each token with its own
  position, cache slot and loop-specific KV cache.
- **Change:**
  - `kernels/dory_decode_attention.py`: the attention and combine kernels gain
    a token axis. Each token has its own input rows, sequence length, slot and
    cache, given as an element offset from the base cache, so the caches of a
    layer's loops only need equal strides. `DecodeAttention.multi_from_raw` is
    the entry point. A `MULTI_TOKEN` constexpr compiles the axis out for the
    single-token entry points: with the axis compiled in, the fused path
    measured 4.73 ms instead of 4.70 ms (attention 0.459 instead of 0.428 ms),
    and the compile-out restores 4.70 ms in two runs
    (`profile_20261018T210727Z_*`, `profile_20261018T210746Z_*`).
  - `kernels/dory_multi_token.py`: `MultiTokenSublayers` runs one attention or
    MLP sublayer for T tokens on the E16 tensor-core GEMV, with these kernels:
    - a q/k/v + gate epilogue: split-K sum, rotary embedding at each token's
      position, sigmoid gate;
    - gate+up with SwiGLU in the GEMV epilogue;
    - a per-token residual update that also adds the split-K partials and
      applies the input-norm scaling.

    It reads the single-token sublayers' weight layouts.
- **Measured:** `kernels/bench_multi_token.py`
  ([results/multi_token_bench_20261018T205748Z.md](results/multi_token_bench_20261018T205748Z.md)),
  real recurrent-block weights, 640-token context, 4 tokens in their own
  caches.
  - Accuracy against the single-token sublayers run once per token:
    - outputs differ by 4.7e-4 to 9.1e-4 (BF16 rounding of FP16 tensor-core
      inputs);
    - cached keys and values are identical, apart from one key at 1.5e-5.
    - Multi-token attention alone is bit-identical to per-token single calls
      for 1, 2 and 4 tokens, including a single token using a non-zero cache
      offset.
  - Time per attention+MLP layer pair over the 20 recurrent pairs:

    | Path | Tokens | us per pair | Relative |
    |---|---:|---:|---:|
    | single-token sublayers | 1 | 75.3 | 1.00 |
    | multi-token sublayers | 1 | 91.9 | 1.22 |
    | multi-token sublayers | 2 | 94.5 | 1.25 |
    | multi-token sublayers | 4 | 101.5 | 1.35 |

    Attention for 4 tokens costs 18.1 us against 8.0 us for one, because each
    token reads its own loop's cache (4 times the KV bytes, about 10.4 us at
    peak bandwidth).
- **Result:** 4 tokens through a layer pair cost 1.35 times one token on the
  single-token path, a little above the 1.3 weight-kernel gate because of the
  attention traffic and the extra epilogue launches. Estimated wavefront step at
  4 loops:
  - input block for 1 token: 0.60 ms;
  - recurrent block for 4 tokens: 2.03 ms;
  - output block for 2 tokens: 0.76 ms;
  - head and other: about 0.5 ms.

  That is about 3.9 ms per step, and about 4.5 ms per committed token at
  93.5% acceptance (1.7 times the 7.71 ms of sequential decoding).
- **Decision:** kept; Stage 2 (a stand-alone wavefront prototype) builds on it.
  Folding the epilogues into their GEMVs can recover part of the 1-token
  overhead later.

### E18: stand-alone wavefront prototype at 4 loops (2026-10-18) - Stage 2 passed

- **Goal:** Stage 2 of LOOP_SPECULATION.md. Wavefront self-speculative
  decoding at 4 loops, drafting at loop 1, must commit exactly the tokens of
  sequential decoding and be at least 1.5 times faster than the fused
  single-token path.
- **Change:**
  - `kernels/dory_wavefront.py`: `DoryPrototype`, a stand-alone decoder outside
    vLLM. It holds the whole model on the GPU: the fused sublayers' weights,
    the embeddings and the INT8 LM head. It has its own KV caches: one per
    attention layer, and one per loop for recurrent layers, consecutive so
    that a token's loop picks its cache by offset. It decodes three ways:
    - A: sequential, single-token kernels;
    - B: sequential, multi-token kernels for one token;
    - W: wavefront. Each step runs the input block for the newest token and one
      recurrent-block call for every token in flight, each in its own loop.
      Then the output block and head run for the token that finished loop 1
      (its prediction drafts the next token) and the one that finished loop 4
      (its prediction verifies the next token's draft). On a mismatch, the
      full-depth token is committed and the younger tokens are discarded.

    Each block is a CUDA graph; the host keeps the in-flight bookkeeping, with
    one synchronization per step.
  - `QuantizedLmHead.rows` (`kernels/dory_lm_head.py`): the INT8 head for up to
    16 rows on tensor cores, reading the weights once. It takes 360 us for 1
    or 2 rows, against 357 us for the single-row kernel, and each row's result
    does not depend on the row count. A first version broadcast the rows on
    CUDA cores and took 610 us.
  - `kernels/bench_wavefront.py`: runs A, B and W on chat prompts (chat
    template, thinking on). It checks W against B token by token and reports
    decode speed (prefill not timed), acceptance and tokens in flight.
    `--context N` prepends about N tokens of this log to each prompt.
- **Exactness:** W committed exactly B's tokens on all 11 prompts (2816 tokens).
  W reproduces B because each token's computation uses the same per-token
  kernels in both. The draft token can read the output-block cache entry that
  the committing token writes in the same launch, but that only affects draft
  quality. A and B diverge on some prompts after 21 to 251 tokens: near-ties
  that the single-token and multi-token kernels round differently.
- **Results** ([results/wavefront_bench_20261018T211633Z.md](results/wavefront_bench_20261018T211633Z.md),
  [results/wavefront_bench_20261018T211933Z.md](results/wavefront_bench_20261018T211933Z.md)):

  | Workload | A (sequential, fused) | W (wavefront) | Speedup | Drafts accepted | Tokens in flight per step |
  |---|---:|---:|---:|---:|---:|
  | 8 chat prompts (24-42 tokens), 256 tokens each | 137.5 tok/s (7.27 ms) | 242.8 tok/s (4.12 ms) | 1.77x | 95.0% of 2040 | 3.72 |
  | 3 chat prompts after about 1000 tokens of notes, 256 tokens each | 130.1 tok/s (7.68 ms) | 211.3 tok/s (4.73 ms) | 1.62x | 93.6% of 765 | 3.66 |

  B, the multi-token kernels run one token at a time, gives 116.0 and
  110.2 tok/s. Sequential A at about 1050 tokens (7.68 ms) matches vLLM's fused
  path at 4 loops and 640 tokens (7.71 ms). The gain is smaller at the longer
  context because every token in flight reads its own loop's KV cache, so the
  recurrent block's attention traffic grows with context.
- **Decision:** Stage 2 passed (exact, at least 1.5 times). Next steps:
  - lossless sampling: rejection sampling against the shallow distribution,
    since Dory's default is temperature 1 and top-p 0.95;
  - Stage 3: integration into vLLM, with the batch guard;
  - cheaper steps: fold the multi-token epilogues, and move the per-step host
    bookkeeping onto the GPU.

### E19: lossless sampling for the wavefront prototype (2026-10-18) - kept

- **Goal:** Dory's default generation samples at temperature 1 with top-p 0.95,
  so the wavefront must sample without changing the output distribution.
- **Change:**
  - `kernels/dory_sampling.py`:
    - `filtered_probs`: temperature and top-p filtering;
    - `speculative_accept`: the standard speculative-sampling rule. A draft x
      drawn from q is accepted with probability min(1, p(x)/q(x)); otherwise a
      token is drawn from the normalized residual max(0, p - q), so committed
      tokens follow p exactly.
  - In `kernels/dory_wavefront.py`:
    - drafts are sampled from the shallow (loop-1) distribution, which is kept
      with the drafting token;
    - when that token reaches loop 4, its full-depth distribution verifies the
      draft with `speculative_accept`;
    - sequential decoding samples inside its CUDA graph with Gumbel-max.
  - `kernels/bench_wavefront.py` gains `--temperature` and `--top-p`.
- **Statistical check:** `kernels/check_speculative_sampling.py`, 400,000 draws
  per case:
  - random p and q at vocabularies of 50 and 200 with top-p 1.0, 0.9 and 0.95,
    plus an unrelated and an identical draft;
  - committed tokens match p: total variation 0.0008 to 0.0039, chi-square
    within its degrees of freedom, no token outside p's support;
  - the acceptance rate equals the overlap sum(min(p, q)), as theory says.
- **Measured:** 8 chat prompts, 256 tokens, temperature 1.0, top-p 0.95
  ([results/wavefront_bench_20261018T212342Z.md](results/wavefront_bench_20261018T212342Z.md)):
  sequential fused decoding 135.7 tok/s (7.37 ms per token), wavefront
  213.7 tok/s (4.68 ms), 1.58x. 94.5% of drafts accepted (E15's same-history
  overlap of loop 1 against loop 4 was 0.936).
- **Result:** sampling keeps most of the greedy gain (1.77x). Its extra cost,
  about 0.56 ms per token against greedy wavefront, is the filtering, sampling
  and acceptance done eagerly between graphs: a 131,072-entry sort per row and
  a host round trip per step.
- **Decision:** kept. Moving the sampling into the readout graph is part of the
  next item (cheaper steps).

### E20: cheaper wavefront steps: tuned tensor-core GEMVs and sampling inside the graph (2026-10-18) - kept

- **Measured first:** a steady-state step (4 tokens in flight, two readouts)
  measured 3.538 ms of wall time against 3.53 ms of graph GPU time, so the
  host bookkeeping is negligible:
  - input block for 1 token: 0.585 ms;
  - recurrent block for 4 tokens: 1.868 ms (1.752 ms for 1);
  - output block and head for 2 tokens: 1.078 ms.

  The multi-token sublayers' fixed cost (about 92 us per layer pair even for 1
  token, against 75 us on the single-token path) was the largest target.
- **Change:**
  - **Tuning.** `w4a16_gemm_tc` takes a pipelining depth, and
    `kernels/bench_gemv_tokens.py --wide` searches 108 launch configurations
    ([results/gemv_tokens_bench_20261018T212818Z.md](results/gemv_tokens_bench_20261018T212818Z.md)).
    At 4 tokens:
    - q/k/v + gate: 21.4 to 20.2 us (split 2, 2 stages);
    - attention output: 10.9 to 10.2 us (split 4, 2 stages);
    - gate+up: 26.6 to 26.5 us;
    - down: 16.9 to 15.0 us (128 rows, 8 warps, split 4).

    A layer pair's weights take 71.9 us, 1.15 times the single-token kernels.
  - **Sampling in the readout graph.** `kernels/dory_sampling.py` draws with the
    Gumbel-max trick (no host synchronization), and the wavefront's readout
    graph does the filtering, acceptance and the next draft. The draft
    distributions sit in a device ring indexed by position, and the host passes
    four integers per step. The statistical check still passes.
- **Measured:**
  - multi-token layer pair ([results/multi_token_bench_20261018T213530Z.md](results/multi_token_bench_20261018T213530Z.md)):
    97.8 us for 4 tokens (1.31 times one single-token pair, from 1.35), 88.4 us
    for 1;
  - greedy wavefront, 8 chat prompts, 256 tokens
    ([results/wavefront_bench_20261018T213624Z.md](results/wavefront_bench_20261018T213624Z.md)):
    246.6 tok/s (4.05 ms per token) against 137.6 tok/s sequential, 1.79x, still
    exactly equal to B on every prompt, 94.6% of drafts accepted;
  - sampling at temperature 1.0 and top-p 0.95
    ([results/wavefront_bench_20261018T213659Z.md](results/wavefront_bench_20261018T213659Z.md)):
    230.1 tok/s (4.35 ms) against 135.6 tok/s, 1.70x (from 1.58x), 95.1% of
    drafts accepted.
- **Decision:** kept.

### E21: cost of vLLM's V1 model runner and of turning off async scheduling (2026-10-18) - measured

- **Goal:** Stage 3 option 2 (LOOP_SPECULATION.md) would serve the wavefront
  through vLLM's custom speculative-proposer hook. That hook only exists in the
  V1 model runner, and vLLM turns off asynchronous scheduling for it, for all
  traffic. Measure what both cost before choosing.
- **Change:**
  - `bench/runner_comparison.py` runs, one job at a time, the single-user
    profile at 2 and 4 loops and the serving benchmark at 2 loops for three
    configurations: V2 (the default), V1 (`VLLM_USE_V2_MODEL_RUNNER=0`), and V1
    without asynchronous scheduling. The profiler and the serving harness gain
    `--no-async-scheduling` and record both settings.
  - **Fix in the fused path.** The V1 runner's CUDA graph warm-up runs carry
    no attention metadata. vLLM's documented contract allows that, and vLLM's
    attention then skips its work. The fused path assumed metadata was always
    present and crashed. Those runs now take the vLLM attention path
    (`_paged_cache` in `kernels/dory_fused_model.py`). Decode steps always
    carry metadata, so their numerics are unchanged.
- **Measured:**
  [results/runner_comparison_20261018T232337Z/report.md](results/runner_comparison_20261018T232337Z/report.md).
  The V2 rows reproduce earlier numbers (4.71 against 4.70 ms; 215.8 against
  215.7 tok/s for one user).

  | Measurement | V2 | V1 | V1, no async scheduling |
  |---|---:|---:|---:|
  | Decode step, 2 loops | 4.71 ms | 4.78 ms (-1.5%) | 5.66 ms (-16.9%) |
  | Decode step, 4 loops | 7.75 ms | 7.77 ms (-0.3%) | 8.84 ms (-12.3%) |
  | Serving 128/1024, 1 user | 215.8 tok/s | -1.5% | -12.3% |
  | Serving 128/1024, 16 users | 1718.1 tok/s | -0.7% | -8.9% |
  | Serving 128/1024, 64 users | 3135.4 tok/s | -0.7% | -7.8% |
  | Serving 2560/300, 1 user | 180.8 tok/s | -1.3% | -10.8% |
  | Serving 2560/300, 16 users | 694.7 tok/s | -0.1% | -4.4% |
  | Serving 2560/300, 64 users | 782.1 tok/s | -0.4% | -3.7% |

  - Without async scheduling, the GPU sits idle 0.94 to 1.07 ms per decode
    step, against 0.04 to 0.08 ms. The forward itself is unchanged (4.22 to
    4.23 ms at 2 loops).
  - V1, with and without async scheduling, gives exactly V2's greedy reference
    outputs (4 prompts, 256 tokens), and no request failed.
  - KV cache: 98,548 tokens on V2, 94,960 on V1 and 128,383 on V1 without
    async scheduling. The last difference is unexplained and was not
    investigated.
- **Decision:** the V1 runner alone costs at most 1.5%. Turning off
  asynchronous scheduling fails the batch guard (up to 8.9% lost at 16 and 64
  users, against the 2% limit) and costs single users 11 to 17%. The custom
  proposer receives the sampled tokens as Python lists, so it needs the host to
  wait for every step; keeping async scheduling would mean rewriting the hook.
  The roughly 1 ms gap would also land on every wavefront step (about 3.5 ms
  of GPU time at 4 loops, E20). Option 2 as vLLM ships it is ruled out, and
  option 1 (a speculator in the V2 runner) is the recommended route.

### E22: wavefront decoding served by vLLM at 4 loops (2026-10-19) - kept

- **Goal:** Stage 3, option 1 (E21): serve the wavefront (E18 to E20) from
  vLLM's V2 model runner for a request decoding alone, without changing
  batched serving.
- **Change:** a pipelined decode step inside the Dory model instead of a
  speculator (LOOP_SPECULATION.md, "Stage 3, as built"). vLLM keeps running
  plain one-token decode steps and samples every token itself;
  `kernels/dory_wavefront_engine.py` runs a steady step when vLLM's token is
  the draft in flight for its position, else a catch-up step that refills the
  pipeline. Drafts reproduce vLLM's Gumbel noise for their position. In detail:
  - vLLM (patches/vllm-dory-fused-decode.patch, now 6 files): DoryConfig's
    `wavefront_decode` asks for `n_recurrent_loops - 1` lookahead KV slots,
    which the scheduler reserves only while at most one request runs. The V2
    runner runs single-request, single-token steps outside its FULL CUDA
    graphs, tells the model each step's request and binds the sampler and
    block tables. The engine runs only when the request's allocated blocks
    cover every in-flight position: a V2 block-table row keeps stale entries
    past the allocation, which can name another request's block.
  - **Per-token block tables.** vLLM keeps Dory's attention layers in 4 KV
    cache groups, each with its own block table, so one layer's loop caches can
    have different physical blocks. The multi-token attention kernel now takes
    a block-table offset per token and computes each token's write slot from
    its own table (single-token mode unchanged).
  - Tools: `bench/wavefront_smoke.py` (in-process, chat prompts),
    `bench/run_baseline.py --wavefront --shapes --concurrency`, and the
    `nvfp4-triton-fused-wavefront` configuration of `bench/compare_outputs.py`.
- **Checks:**
  - `kernels/bench_multi_token.py`, now with a shuffled block table per token
    ([results/multi_token_bench_20261019T000713Z.md](results/multi_token_bench_20261019T000713Z.md)):
    accuracy as before; with identity tables the timing equals E20's (88.2,
    90.7 and 97.6 us per pair for 1, 2 and 4 tokens; shuffled tables make one
    token 94.0 us from scattered reads).
  - The prototype is still exact (W equals B on 4 prompts;
    [results/wavefront_bench_20261019T000813Z.md](results/wavefront_bench_20261019T000813Z.md)).
  - In vLLM, steady steps equal catch-up-only steps
    (`DORY_WAVEFRONT_FORCE_CATCH_UP=1`, which computes each token from scratch
    with the same kernels) token for token on 2 greedy prompts of 128 tokens.
  - **Accuracy gate passes** at 4 loops
    ([results/output_agreement_20261019T001920Z/report.md](results/output_agreement_20261019T001920Z/report.md),
    64 prompts against unmodified vLLM): no divergent token outside the top 5,
    and a mean |logprob difference| of 0.0068 over the 6291 positions every
    configuration shares, against 0.0070 for the FlashInfer floor and 0.0066
    for the fused path. (Run before the lookahead change below, which leaves
    a lone request's computation unchanged: identical tokens in-process.)
- **Measured:**
  - In-process, chat prompts (results/wavefront_smoke/): greedy 261.2 tok/s
    against 136.7 for the fused path (1.91x), vLLM sampled the draft in 309 of
    315 checks; temperature 1, top-p 0.95: 240.4 against 135.6 tok/s (1.77x),
    1020 of 1079 drafts (94.5%; the prototype's speculative acceptance was
    95.1%).
  - Serving at 4 loops, unmodified (20261019T003224Z), fused
    (20261019T004110Z) and wavefront (20261019T013129Z), random-token prompts
    at the model's default sampling (temperature 1, top-p 0.95):

    | Input/output | Users | Unmodified tok/s | Fused | Wavefront | Wavefront vs unmodified |
    |---|---:|---:|---:|---:|---:|
    | 128/1024 | 1 | 99.6 | 131.1 | 184.5 | +85% |
    | 128/1024 | 16 | 1049.0 | 1048.4 | 1049.7 | +0.1% |
    | 128/1024 | 64 | 1583.8 | 1563.4 | 1566.8 | -1.1% |
    | 2560/300 | 1 | 84.4 | 109.3 | 147.2 | +74% |
    | 2560/300 | 16 | 396.9 | 412.8 | 413.6 | +4.2% |
    | 2560/300 | 64 | 419.2 | 422.7 | 423.0 | +0.9% |

    vLLM sampled 87 to 92% of drafts on these prompts, so more steps are
    catch-ups (about 13.5 ms each, against about 3.6 ms for a steady step).
    128/1024 for one user gives 201.7 to 201.9 tok/s when it is the server's
    first benchmark (four runs) and 184.5 when it follows the 2560-token rows
    (two runs); the cause of that order dependence is not established.
  - **Batch guard:** an earlier version reserved the lookahead slots for every
    request and lost 1.4% to the fused path at 128/1024 and 64 users
    (1532.0/1533.4/1532.4 tok/s, 20261019T004918Z to 20261019T010911Z), where
    the KV cache is full at 4 loops (64 x 1152 tokens against 57k). With
    lone-request lookahead the wavefront equals the fused path there
    (1549.8/1550.8/1566.8 against 1548.4/1550.2/1563.4) and passes every row.
    That row is 1.1 to 2.4% below unmodified vLLM (1583.8/1587.9/1589.6) for
    both, from the fused decoder's weight copy (KV cache 57,010 tokens with the
    wavefront and 56,966 fused, against 65,169 unmodified).
- **Decision:** kept, opt-in through `"wavefront_decode": true` in
  `--hf-overrides`. At 4 loops a lone request decodes 1.4 to 1.9 times faster
  than on the fused path. Open: cheaper catch-up steps (they dominate on
  unpredictable text), the per-step host synchronization, and the fused
  decoder's duplicate weights, which cost the KV-limited 64-user row about 2%.

### E23: wavefront decoding at 2 loops, and bimodal long-prompt batch rows (2026-10-19) - measured

- **Goal:** the 2-loop column of docs/BENCHMARKS.md, which until now had no
  wavefront run, and the missing steps of its feature-by-feature tables (the
  finished fused path with the BF16 LM head at 1 and 4 loops, and with the
  INT8 head at 1 loop).
- **Measured:** `bench/run_baseline.py --loops 2 --wavefront`. With two loops
  the pipeline holds two tokens and drafts at loop 1 for a 2-loop model.

  | Input/output | Users | Unmodified (20261018T180516Z) | Fused, INT8 head (20261018T190327Z) | Wavefront (20261019T022147Z) |
  |---|---:|---:|---:|---:|
  | 128/1024 | 1 | 161.7 | 215.7 | 250.8 |
  | 128/1024 | 16 | 1721.1 | 1721.5 | 1719.5 |
  | 128/1024 | 64 | 3145.9 | 3144.8 | 3145.0 |
  | 2560/300 | 1 | 138.0 | 180.8 | 201.8 |
  | 2560/300 | 16 | 697.2 | 696.5 | 697.8 |
  | 2560/300 | 64 | 779.3 | 784.9 | 784.7 |

  One user gains 16% (128/1024) and 12% (2560/300) over the fused path, less
  than at 4 loops because a sequential 2-loop step has less recurrent work to
  pipeline; vLLM sampled 90.9% of drafts.

  The fused path at 1 loop (20261019T023002Z with the BF16 head,
  20261019T024248Z with the INT8 head) gives one user 286.3 and 318.6 tok/s at
  128/1024, against 234.8 for unmodified vLLM with Triton attention; at 4 loops
  with the BF16 head (20261019T023002Z) 125.7, against 131.1 with the INT8 head.
- **Bimodal long-prompt batch rows.** The 2560/300 rows with 16 and 64 users
  take one of two levels in different runs of the same configuration, with a
  longer time to first token and the same decode speed at the low level:

  | Loops | Low level (16 users) | Runs | High level | Runs |
  |---:|---:|---|---:|---|
  | 2 | 650 to 653 | 20261018T114127Z, 20261018T163313Z, 20261019T020505Z, 20261019T021250Z | 696 to 698 | 20261018T180516Z, 20261018T181035Z, 20261018T185825Z, 20261018T190327Z, 20261019T021556Z, 20261019T021841Z, 20261019T022147Z |
  | 1 | 953 to 959 | 20261018T114127Z, 20261018T163313Z, 20261019T023002Z | 1058 to 1059 | 20261019T024248Z, 20261019T024728Z (BF16 head, the configuration of 20261019T023002Z) |

  Both levels occur with unmodified vLLM, the fused path and the wavefront, so
  the level is not a property of a configuration. The benchmark's prompts are
  fixed (`--seed 0`). Servers that compiled the model got about 0.5 GiB less
  KV cache than servers that loaded it from vLLM's compile cache, but the level
  does not follow that either (20261018T163313Z loaded it and is low,
  20261018T181035Z compiled it and is high). An earlier version of this entry
  attributed the low 2-loop runs to a transient slowdown; the later 1-loop
  pair (low, then high with the same configuration minutes apart) and the
  table above do not support that. The cause remains unknown.
- **Decision:** the published tables keep every run as measured, and
  docs/BENCHMARKS.md marks those rows as not comparable across columns. The
  single-user rows and the 128/1024 rows, this work's targets and guards, are
  not affected. Differences in batched results require a repeat run before they can
  be attributed to a code change.

### E24: Dory's shared "last_loop" KV cache against per-loop caches (2026-10-19) - measured

- **Goal:** WaveFront Decoding reports large gains from cross-recurrence KV
  sharing (one KV cache shared by the loops) as an optional, approximate
  extension, and Dory supports a form of it natively (`recurrent_kv_cache_mode:
  "last_loop"`: one cache per recurrent layer that every loop overwrites). This
  work used per-loop caches throughout. Before any engine work, measure what
  the shared cache does to quality and serving on its own.
- **Change:** `nvfp4-triton-last-loop` in `bench/compare_outputs.py` (also
  used by `bench/eval_tasks.py`), and `bench/run_baseline.py --kv-cache-mode`.
- **Measured** (unmodified vLLM, Triton attention, 4 loops, against per-loop
  caches):
  - Greedy agreement with the per-loop reference
    ([results/output_agreement_20261019T104639Z/report.md](results/output_agreement_20261019T104639Z/report.md)):
    no divergent token outside the top 5; 6 of 64 outputs identical, 104 shared
    tokens on average. Over the 5056 positions every configuration shares, the
    mean |logprob difference| is 0.0078, against 0.0066 for the FlashInfer floor,
    0.0062 for the fused path and 0.0062 for the wavefront.
  - Tasks ([results/task_eval_20261019T105029Z/report.md](results/task_eval_20261019T105029Z/report.md)):
    GSM8K 91.2% against 90.8% (250 problems, McNemar p = 1.00), MMLU 62.7%
    against 62.3% (1000 questions, p = 0.29), same MMLU answer letter for 98.9%
    of questions.
  - Serving (20261019T111134Z against 20261019T003224Z): KV cache 173,814
    tokens against 65,169 (2.67x). One user unchanged (99.9 against 99.6 tok/s
    at 128/1024, 84.6 against 84.4 at 2560/300) and 16 users unchanged; 64
    users 1889.5 against 1583.8 tok/s at 128/1024 (+19%, median time to first
    token 277 ms against 1204 ms) and 637.8 against 419.2 at 2560/300 (+52%,
    3.5 s against 23.9 s).
- **Decision:** the shared cache costs no measurable task quality and lifts
  every KV-limited row, so it goes into docs/FUTURE_WORK.md with the wavefront
  and fused path on a shared cache as next steps. It changes predictions more
  than the accuracy gate's floor allows for kernel changes, as expected from a
  different cache semantics, so it stays a deployment choice rather than a
  default.

### E25: the fused path and the wavefront on Dory's shared KV cache (2026-10-19) - kept

- **Goal:** E24 showed the shared "last_loop" KV cache keeps task quality and
  triples the KV capacity on unmodified vLLM. Run this work's fused path and
  wavefront on it, and measure what the wavefront gains.
- **Change:**
  - On one shared cache an in-flight token rewrites its own slot at every
    loop while the younger tokens of the same recurrent call read it, a race.
    `DecodeAttention.multi_from_raw(..., shared_cache=True)` therefore caches
    every token's new key and value with `_write_new_kv_kernel` first and runs
    attention without writing (`WRITE_KV`), so each token reads the older
    in-flight tokens at their latest loop. `MultiTokenSublayers.attention`
    passes the flag through; the engine sets it for the recurrent layers when
    the model's loops share a cache. The fused single-token path already used
    the shared cache (`mixer.attn[0]`).
  - DoryConfig no longer refuses `wavefront_decode` with `last_loop`.
  - Semantics: sequential decoding on a shared cache reads earlier tokens after
    their last loop; the wavefront's in-flight tokens see older in-flight tokens
    at their deepest loop so far, so it approximates sequential decoding
    (WaveFront Decoding treats cross-recurrence KV sharing the same way). A
    catch-up step still computes vLLM's token after every earlier token is
    complete.
- **Checks:**
  - `kernels/bench_multi_token.py`, one cache shared by 4 tokens, against the
    single-token sublayers run in position order (each token then sees the new
    entries below it): outputs within 4 to 9e-4, cached keys and values about
    exact, as in per-loop mode
    ([results/multi_token_bench_20261019T115318Z.md](results/multi_token_bench_20261019T115318Z.md)).
  - In-process, 4 chat prompts x 256 tokens (results/wavefront_smoke/*_ll_*):
    greedy wavefront against catch-up-only steps (sequential semantics, same
    kernels) shares 256, 256, 158 and 157 tokens; the fused path also diverges
    from catch-up-only at token 158 of the third prompt, so one divergence may
    come from the approximation.
  - **Accuracy gate passes** against unmodified vLLM on the shared cache
    ([results/output_agreement_20261019T115915Z/report.md](results/output_agreement_20261019T115915Z/report.md)):
    no divergent token outside the top 5; over the 6425 positions every
    configuration shares, the mean |logprob difference| is 0.0066 for the
    wavefront and for the fused path, against 0.0072 for the FlashInfer floor
    on the same cache.
  - Tasks against unmodified vLLM on the shared cache
    ([results/task_eval_20261019T120832Z/report.md](results/task_eval_20261019T120832Z/report.md)):
    GSM8K 92.0% against 91.2% (McNemar p = 0.73), MMLU 62.7% against 62.7%
    (p = 1.00).
- **Measured:**
  - In-process, chat prompts: wavefront 265.1 tok/s greedy (97.9% of drafts
    sampled by vLLM) and 259.4 at temperature 1, top-p 0.95 (97.1%; per-loop
    240.4 and 94.5%), against 136.7 and 136.1 for the fused path.
  - Serving at 4 loops, shared cache: unmodified 20261019T111134Z, fused
    20261019T122346Z, wavefront 20261019T123105Z. One user: 248.4 tok/s at
    128/1024 (fused 131.5, unmodified 99.9; per-loop wavefront 184.5 to 201.9)
    and 172.9 at 2560/300 (109.8, 84.6; per-loop 147.2). vLLM sampled 96.2% of
    drafts on these random-token prompts (per-loop 87 to 92%). Batched rows
    equal unmodified vLLM on the same cache within run-to-run variation; the
    64-user rows are 19% and 52% above per-loop serving. KV cache 151,915
    tokens with the fused path's weights (56,966 per-loop).
- **Decision:** kept. The wavefront runs on either cache mode; on the shared
  cache it is the fastest single-user configuration measured (3.2 times vLLM as
  shipped at 4 loops), with quality within the gate's noise floor. Which mode
  to serve is a deployment choice (docs/FUTURE_WORK.md).

### E26: complete serving suite with independent prompt processing (2026-10-20) - measured

- **Goal:** finish the serving comparison across loop counts, cache modes and
  intermediate batch sizes, including the 2-loop shared-cache wavefront.
- **Method correction:** every `vllm bench serve --seed 0` invocation draws
  the same random prompts. With prefix caching enabled on a reused server,
  later benchmarks reuse earlier prompt processing. The first expanded suite
  reached a cumulative prefix-cache hit rate of 73%. This inflated output
  throughput and shortened time to first token for prompt-heavy batches,
  explaining the long-prompt variation recorded in E23. At 2 loops and 16
  users with Triton, 2560/300 measured 849.2 tok/s with prefix caching
  (`results/20261019T132848Z`) and 655.4 with it disabled in the verification
  run (`results/20261019T143123Z`). This does not establish the cause of the
  separate single-user wavefront order effect.
- **Change:** the serving harness disables prefix caching by default,
  `--prefix-caching` restores it explicitly, and the manifest records the
  setting. The suite uses NVFP4, loop counts 1, 2 and 4, shapes 128/1024 and
  2560/300, and 1, 2, 4, 8, 16, 32 and 64 users. Those levels measure 8, 8,
  16, 32, 64, 128 and 128 requests, with two warm-up requests before each row.
  All benchmarks run sequentially on the idle-checked RTX 4090, long prompts
  before short prompts, with the same software, model revisions and sampler.
- **Completed runs:**

  | Configuration | Loops | Run |
  |---|---|---|
  | vLLM as shipped, per-loop KV | 1, 2, 4 | 20261019T152606Z |
  | Triton, per-loop KV | 1, 2, 4 | 20261019T232759Z |
  | Triton, shared KV | 2, 4 | 20261019T160704Z |
  | Fused, BF16 head, per-loop KV | 1, 2, 4 | 20261020T002938Z |
  | Fused, INT8 head, per-loop KV | 1, 2, 4 | 20261020T010402Z |
  | Fused, INT8 head, shared KV | 2, 4 | 20261020T000323Z |
  | Fused and wavefront, per-loop KV | 2, 4 | 20261019T143422Z |
  | Fused and wavefront, shared KV | 2 | 20261020T015236Z |
  | Fused and wavefront, shared KV | 4 | 20261019T150009Z |

- **Validation:** 20 configurations, 280 benchmark rows, 15,360 successful
  measured requests, no failed requests. The table publisher checks the
  complete matrix, request counts, disabled prefix caching, cache and head
  settings, software, model revisions, GPU driver and sampler before writing
  tables. No model or kernel code changed; the quality evaluations in E25
  still apply to the same implementation.
- **Measured:**
  - One user with all applicable features: 318.8 tok/s at 1 loop, 264.1 at
    2 loops and 243.4 at 4 loops for 128/1024, respectively 69%, 107% and
    212% above vLLM as shipped. At 2560/300 the corresponding rates are
    264.9, 208.7 and 168.2 tok/s (74%, 102% and 169% above shipped).
  - Shared KV adds about 5% over per-loop wavefront at 2 loops, and 25% at
    128/1024 or 17% at 2560/300 at 4 loops. With 64 users at 4 loops,
    shared-cache wavefront is 20% and 14% faster than per-loop wavefront;
    earlier larger long-prompt gains included prefix reuse.
  - The wavefront still runs only for a request decoding alone. At 4 loops
    with shared KV and 128/1024, one user gets 243.4 tok/s, but two users
    together get 192.8, about 96.4 each. Small-batch wavefront pipelines remain
    a useful next step.
  - In the per-loop, 4-loop, 64-user, 128/1024 row, the fused configurations
    are 2.1 to 2.7% slower than unmodified Triton. This exceeds the original
    2% batch target: the duplicated weights reduce the cache from 65,180 to
    56,966 tokens. Shared KV holds 173,814 tokens unmodified and 151,915 with
    fused weights and avoids this capacity limit. This benchmark update
    records the limitation; it does not change the weight layout.
  - The operator confirmed a concurrent workload during the earlier 2-loop
    shared-cache wavefront configuration. Its measurements are excluded. The
    entire configuration is replaced by a clean run; the independently measured
    4-loop configuration is retained. In the replacement, all rows with multiple
    users agree with unmodified Triton on the same cache within 0.4%.
- **Publication:** the README headline and the feature, final and shared-cache
  tables now use the corrected suite. New tables show every concurrency
  level. Backend, quantization and early development-history comparisons keep
  their original measurements and are explicitly labeled as using prefix
  caching. Earlier experiment entries retain what was measured at the time.
