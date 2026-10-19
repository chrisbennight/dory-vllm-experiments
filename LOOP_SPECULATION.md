# Speculative decoding with Dory's own loops: design (2026-10-18)

These notes record the design and implementation of decoding with Dory's own
loop predictions. The goal is to produce several tokens per weight read for
one user: draft tokens from the shallow (1-loop) prediction, then verify them
at full depth while preserving the output distribution.

The early sections describe the options considered on 2026-10-18, based on
published work and local measurements. The final section records the vLLM
integration built on 2026-10-19. For current setup instructions, see the
[README](README.md).

## What we know

**Measured here** (EXPERIMENTS.md):

- E15: on the same token histories, the 1-loop prediction matches the 2-loop
  one on 94.2% of tokens (distribution overlap 0.944) and the 4-loop one on
  93.5%. A 4-token greedy draft gets 3.45 tokens accepted on average.
- Block costs on the fused single-token path (INT8 head, 640-token context):
  input block (8 attention+MLP pairs) 0.60 ms, recurrent block (20 pairs)
  1.50 ms per loop, output block (8 pairs) 0.60 ms, LM head 0.36 ms, sampling
  and bookkeeping 0.14 ms. Sequential decoding: 4.70 ms per token at 2 loops,
  about 7.7 ms at 4 loops.
- E8: vLLM's Marlin GEMM costs the same for 1 to 8 tokens; our CUDA-core
  multi-token GEMV does not (1.3 to 3.1 times slower than Marlin from 2
  tokens).
- E14: vLLM's speculative-decoding machinery (verification, rejection, KV
  slots) already works with Dory and the fused path.

**Published work** (all training-free and lossless unless noted):

| Work | Idea | Reported result | Code |
|---|---|---|---|
| WaveFront Decoding ([arXiv 2609.23033](https://arxiv.org/abs/2609.23033)) | Keep tokens at different loop depths in flight and advance them in one batched recurrent call: the newest token is drafted at a shallow depth while older ones are verified at full depth. Its baseline, draft-then-verify (DtV), drafts several tokens at the shallow depth and then resumes their kept states together to full depth. | 2.42x (Ouro-2.6B, 4 loops), 3.54x (Huginn-3.5B, 32 loops) over normal decoding; 1.27x over DtV. Shallow drafts accepted 92-94% of the time. With a separate KV cache per loop, the token sequence matches normal decoding exactly in FP32. | [summerbro-hhj/wavefront-decoding](https://github.com/summerbro-hhj/wavefront-decoding) |
| LoopSpec ([arXiv 2609.17184](https://arxiv.org/abs/2609.17184)) | The same pipeline as branches, plus a second, deeper proposal used only when the first is rejected; rejection sampling makes sampling lossless; closed-form best draft depths. | Up to 3.36x on Ouro (4 loops), up to 6.83x on Raven (32 loops); 2.83x with sampling. Notes that models with heavy input and output blocks gain less, because drafting runs them more often. | [kaist-flexml-lab/loopspec](https://github.com/kaist-flexml-lab/loopspec), an SGLang extension with per-branch KV management and CUDA graphs |
| LoopAhead ([arXiv 2610.21594](https://arxiv.org/abs/2610.21594)) | Adds a small trained look-ahead drafter on early-loop states. | Up to 7.82x | Needs training; later option |

**Engines:**

- vLLM: the Dory pull request
  ([vllm#60149](https://github.com/vllm-project/vllm/pull/60149), the code in
  `src/vllm`) has no
  speculation code. Its author wrote on 2026-10-14 that they have an initial
  native self-speculative implementation (1-loop draft, 4-loop verify, reuse of
  early-loop KV caches and activations); it is not public. vLLM offers a
  `custom_class` speculation method, but its proposer only receives token ids,
  so it cannot reuse the model's shallow states on its own. Draft models only
  inherit the target's `hf_overrides` when those are callables, so a 1-loop
  draft of the same checkpoint needs its own model directory.
- llama.cpp: the Dory pull request
  ([llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306), draft)
  supports per-loop KV and shared last-loop KV, without speculation; the Ouro
  pull request ([llama.cpp#29823](https://github.com/ggml-org/llama.cpp/pull/29823)) runs
  with n-gram speculation only.
- SGLang: nothing upstream; LoopSpec ships its own extension.

## Why Dory gains less than the papers at 2 loops

Dory's input and output blocks are large (16 of 36 attention+MLP pairs), and
every committed token still runs them once, plus the LM head. Only the
recurrent block gets cheaper. With the measured costs and verifying 5 tokens at
1.3 times the cost of 1 (2 tokens at 1.15 times):

| Schedule | 2 loops (now 4.70 ms) | 4 loops (now about 7.7 ms) |
|---|---:|---:|
| DtV: draft 4 tokens at loop 1 (3.2 ms each), resume their loop-1 states together for the remaining loops | about 3.6 ms per token (1.31x) | about 4.55 ms (1.69x) |
| Wavefront: one batched recurrent call advances every token in flight (2 tokens at 2 loops, 4 at 4 loops) | about 3.7 ms per token at 94% acceptance (1.27x) | about 4.3 ms at 93.5% (1.79x) |
| Upper bound, every draft accepted | 3.5 ms (1.34x) | about 3.7 ms (2.1x) |

The second column uses the model's default of 2 loops. The third uses 4 loops,
the setting recommended in the model card for agentic tasks. These estimates
suggest that speculation could make 4-loop decoding about as fast as the
measured 2-loop baseline; they are not measured speedups.

## Design

Decision (2026-10-18): 4 loops is the core use case, so the target is the
wavefront schedule at 4 loops, drafting at loop 1, with up to 4 tokens in
flight. Each step runs the input block for the newest token, one batched
recurrent call over every token in flight (each at its own depth, reading its
own loop's KV cache), and the output block and head for the token that just
reached loop 1 (a draft) and the token that just reached loop 4 (a commit).
The DtV description below remains the simpler reference for the KV rules.

**State kept per draft position:** the input-block output (Dory feeds it back
into every loop) and the state after loop 1. Two hidden vectors per position.

**KV caches, keeping vLLM's per-loop layout:**

- input block and loop 1: the draft computes exactly what verification
  would, so drafts write these entries directly, and verification skips them;
- loops 2 and later: written only by verification;
- output block: one cache, as today. Drafts write shallow entries, and
  verification overwrites every position it processes with full-depth
  entries. Verification only reads committed positions (full depth) and its
  own batch, so it is exact. Drafts see full-depth entries for the history,
  which may draft better than the separate shallow caches E15 measured;
- after a rejection, the stale entries of rejected positions are overwritten
  when those positions are next processed, as in vLLM's existing speculative
  decoding.

**Cycle (DtV, k drafts):**

1. Draft: run the last committed token through the input block, loop 1, the
   output block and the head on the single-token fused path. Keep its states,
   and take the top token (greedy) or a sample with its shallow probabilities.
   Repeat for k draft tokens.
2. Verify: resume all k+1 positions from their loop-1 states through loops
   2..T, the output block and the head in one batched pass.
3. Accept: the longest prefix of drafts that matches the full-depth
   prediction (greedy), or standard rejection sampling against the shallow
   probabilities (sampling). Commit the accepted tokens plus the full-depth
   token at the first mismatch (or the bonus token if all were accepted).

**Required building blocks:**

1. Multi-token versions of the fused sublayers for 2 to 8 tokens:
   - the GEMVs (Marlin, or a tensor-core Triton GEMV over our layout, since the
     CUDA-core version of E8 is too slow);
   - normalization and residual updates, and rotary embedding with per-token
     positions;
   - decode attention with several query positions over the paged cache,
     causal within the batch.
2. A resume entry point in the fused decoder: start at loop d from kept states.
3. The scheduler loop above, with greedy and sampling acceptance.

## Plan, with a measurable gate at each stage

1. **Multi-token building blocks, measured alone.** Gate: verifying 5 tokens
   through the recurrent and output blocks costs at most 1.3 times a single
   token on our fused path. If this fails, speculation at 2 loops gains under
   10% and the plan stops there.
2. **Stand-alone DtV prototype** (outside vLLM's scheduler, with our own
   paged cache, on the 64 gate prompts):
   - measure acceptance with the shared output-block cache and real tokens
     per second against sequential fused decoding;
   - check that greedy outputs match sequential decoding, apart from
     floating-point near-ties;
   - gates: at least 1.2x at 2 loops and at least 1.5x at 4 loops.
3. **Wavefront schedule** in the prototype, if 4-loop serving matters.
4. **vLLM integration.** Use vLLM's speculative-decoding scheduling,
   verification and rejection with a proposer that calls the fused decoder,
   and a model forward that resumes verified positions from kept states.
   Measure with the serving benchmark, including the 16- and 64-user batch
   guard: speculation must switch off when a batch already fills the GPU, as
   WaveFront's batch-size results suggest.

**Options after that:**

- **Last-loop KV caching.** The model card supports it, and it removes the
  per-loop KV memory; WaveFront found that shared KV raises acceptance. But it
  changes the model's predictions, and the wavefront schedule is then only
  approximately lossless.
- **n-gram speculation (E14)** for copy-heavy requests, combined with loop
  drafts.
- **A persistent kernel (E13).** It also speeds up the single-token draft
  steps, which dominate DtV's cycle.

## Stage 3 options: serving the wavefront (2026-10-18)

The stand-alone prototype (EXPERIMENTS.md E18, E19) is exact, and 1.6 to 1.8
times faster than sequential fused decoding at 4 loops. Serving it needs prompt
prefill, an API, stop conditions and streaming. Facts from the local vLLM
checkout:

- Our servers use vLLM's V2 model runner (`vllm/v1/worker/gpu/`), including
  with the `ngram_gpu` speculator (E14). Its speculators live in
  `vllm/v1/worker/gpu/spec_decode/`; V2 does not support the `custom_class`
  method, and vLLM falls back to the V1 runner for it.
- V1's `custom_class` proposer only receives token ids. V1 takes draft
  probabilities (for sampling) only from drafters that implement
  `take_last_draft_probs`, in the EAGLE and draft-model branch.
- In speculative decoding, the scheduler allocates KV slots for
  `1 + num_speculative_tokens + num_lookahead_tokens` tokens per request and
  step.

Options:

1. **Inside vLLM, V2 runner (a new speculator).** The Dory model keeps a
   wavefront engine that runs on vLLM's own per-loop KV caches (block table
   and slots from the attention metadata). A Dory speculator proposes the
   tokens the engine has already committed at full depth, and the
   verification forward for a single-request step returns the engine's
   full-depth logits for them:
   - with greedy decoding the rejection sampler accepts them;
   - with sampling, the draft probabilities equal the target's, so acceptance
     is certain and vLLM samples the bonus token as usual.

   vLLM keeps prefill, scheduling, the API and streaming. With other requests
   in the batch, the speculator proposes nothing and vLLM's normal path runs,
   which preserves batch serving. The engine must keep its in-flight positions
   within the KV slots vLLM allocated (through `num_lookahead_tokens`). Effort:
   the largest; it changes vLLM's speculative-decoding interfaces.
2. **Inside vLLM, V1 runner (`custom_class` proposer).** The same engine, with
   vLLM's official custom-proposer hook. The proposer reaches the engine
   through the model, and sampling needs a small patch for draft
   probabilities. It needs fewer changes to vLLM but runs on the older V1 runner,
   whose speed with the fused path has not been measured.
3. **A dedicated single-user server** around the prototype. It needs a fast
   prompt prefill (a causal multi-token attention for prefill chunks, or
   vLLM's prefill kernel on our caches), the chat template, an
   OpenAI-compatible streaming API and stop handling. It gives full control and
   is the quickest to a working demo, but duplicates serving and has no
   batching (other users would need a separate vLLM).

Measured (EXPERIMENTS.md E21): running on the V1 runner costs at most 1.5%,
but the custom proposer also turns off asynchronous scheduling for all
traffic. That costs single users 11 to 17% and batches up to 8.9% (the batch
guard allows 2%), and would add about 1 ms of idle GPU time to every
wavefront step. Option 2 is ruled out as vLLM ships it; option 1 is the
recommended route.

## Stage 3, as built: a pipelined decode step inside the model (2026-10-19)

Option 1 was chosen, but not as a speculator. vLLM verifies K drafts against
K + 1 full-depth distributions in one step, and the wavefront produces one
full-depth distribution per step, so it cannot fill vLLM's verification
rows. Instead, the wavefront pipeline completes about one token per step at
about half a sequential step's cost. vLLM keeps running ordinary one-token
decode steps without a speculative config. Dory's forward pass
(`kernels/dory_wavefront_engine.py`) runs one wavefront step when the batch
contains a single decode token:

- vLLM's token for position q is compared with the draft in flight there. If
  they match, a steady step advances the pipeline by one loop (input block for
  the newest draft, one recurrent call for all four in-flight tokens, output
  block and head for vLLM's token at full depth and the newest token after
  loop 1). Otherwise a catch-up step pushes vLLM's token through every loop
  while new drafts enter behind it, which refills the pipeline.
- vLLM samples every token from the returned full-depth logits with its own
  sampler, so penalties, logprobs, structured output and stop handling are
  vLLM's, and outputs follow the full-depth distribution whatever the drafts
  are.
- Drafts reproduce vLLM's sampling noise: the engine takes the argmax of the
  filtered loop-1 logits plus the Gumbel noise vLLM's sampler will use at that
  position (seed, position, token id), using vLLM's own functions. Close
  distributions then draw the same token.

vLLM changes (patches/vllm-dory-fused-decode.patch): DoryConfig's
`wavefront_decode` asks for loops - 1 lookahead KV slots, which the scheduler
reserves only while at most one request runs; the V2 runner runs
single-request, single-token steps outside its FULL CUDA graphs (the engine
replays its own) and tells the model each step's request, its sampler and its
block tables. The engine runs only when the request's allocated blocks cover
every in-flight position, because a block-table entry past the allocation can
name another request's block. Results: EXPERIMENTS.md E22.
