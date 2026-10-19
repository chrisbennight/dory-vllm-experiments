# Next directions after the hill-climb (2026-10-18)

Starting point: the fused single-token decode path takes 5.05 ms per token at a
640-token prompt (NVFP4, 2 loops, RTX 4090), against 6.32 ms for unmodified
vLLM with Triton attention and a 3.70 ms bandwidth minimum. The opt-in INT8 LM
head brings it to 4.70 ms. Details are in [EXPERIMENTS.md](EXPERIMENTS.md).

This note records the options considered after the initial kernel optimization
work on 2026-10-18. Part 1 develops two proposals; Part 2 compares them with a
separate research note. The final section records the measurements used to
choose the next step. For current priorities, see
[Future work](docs/FUTURE_WORK.md).

Component times used below, from the E9 profile
(`results/profile_20261018T183509Z_*`): weight GEMVs 3.77 ms, attention 0.43 ms,
LM head 0.70 ms, sampling and bookkeeping 0.10 ms, idle 0.04 ms. Per block:
input block (8 attention+MLP pairs) 0.60 ms, recurrent block (20 pairs) 1.50 ms
per loop, output block (8 pairs) 0.60 ms.

## Part 1: two proposals

### A. A persistent kernel that removes kernel boundaries

**What:** run the decoder (or, first, one layer pair) as one long-running
kernel whose programs stay resident on the GPU. Instead of ending, each program
waits for the previous step's result and meanwhile starts loading the next
weights, which do not depend on that result.

**Why:** the remaining 0.90 ms gap of the weight GEMVs is mostly a fixed cost
per kernel, not slow streaming:

- a kernel that does nothing but stream the same weight bytes, launched once
  per projection, already reaches only 79 to 90% of peak bandwidth. It takes
  1.5 to 2.3 us longer than the bandwidth minimum per launch (attention output
  and output gate 7.4 against 5.9 us, gate+up 22.8 against 20.5 us, down 12.0
  against 10.2 us; `results/gemv_explore_20261018T143131Z.md`). That is the
  cost of starting up and draining a kernel, and over 224 GEMV launches per
  token it adds up to roughly 0.4 ms that no single-kernel change can recover;
- the residual updates folded into the output GEMVs (E2) add about 1.5 us of
  single-program tail to each of 112 kernels, about 0.17 ms;
- every per-kernel improvement tried since (E6, E9, E10, E11) moved the step by
  0.03 ms or less.

Ada has no programmatic dependent launch (it needs compute capability 9.0), so
overlapping one kernel's start with the previous kernel's end requires them to
be the same kernel.

**Expected gain:** 0.3 to 0.5 ms per token (6 to 10%), at any context length and
loop count, without changing numerics. It composes with the INT8 head (about
4.2 to 4.4 ms together) and with speculation. This is an estimate; it is not
measured.

**Effort and risks:** weeks. Triton has no grid-wide barrier, so one must be
built from atomics, which deadlocks unless every program is resident at once
(grid no larger than what the GPU holds), or the kernel moves to CUDA C++.
Register pressure grows when several operations share one kernel; CUDA graph
capture and debugging get harder.

**First bounded step:** a persistent MLP layer pair (gate+up, down, residual
update) with a resident grid, compared with today's 38.9 us per MLP sublayer.
Continue only if it saves at least about 1 us per kernel boundary.

### B. Speculative decoding that drafts with fewer of Dory's loops

**What:** use Dory's own early loops as the draft model. A token's prediction
after 1 recurrent loop is a cheap guess at its prediction after 2 loops. Draft
with the shallow prediction, then verify several positions at once with the full
depth, reusing the draft's early-loop work.

**Why:** a single-user decode step reads every weight once to produce one token.
Verifying several tokens per weight read is the only way to beat the 3.70 ms
floor without changing weight precision. E8 measured that vLLM's Marlin GEMM
costs the same for 1 to 8 tokens (gate+up 25.0 us at 1 token, 24.9 us at 8), so
verifying several positions is nearly free in weight traffic.

**Expected gain:** depends on how often the shallow prediction matches the
full-depth one, which has not been measured. At 2 loops there is only one
shallow depth, so at most one draft token is in flight. Pipelining tokens at
different depths (one token in loop 2 while the next is in loop 1) would cost
about 3.9 ms per step with the component times above, if two-token kernels cost
about 15% more than one-token kernels. Each wrong draft adds a step of about
3.5 ms. Per committed token that gives about 4.2 ms at 90% agreement (-17%),
4.6 ms at 80% (-9%) and 4.9 ms at 70% (-3%). With 4 loops, where more tokens can be
in flight, the ceiling is much higher.

**Effort and risks:** very high. vLLM scheduling changes, separate KV caches per
loop and per branch, rollback of rejected positions, multi-token versions of
the fused kernels (E8 showed a straightforward CUDA-core version loses to Marlin
from 2 tokens), and sampling-aware acceptance: Dory's default generation samples
with temperature 1.0 and top-p 0.95, so acceptance is about matching
distributions, not just the top token.

**First bounded step:** measure, on the same token histories, how often the
loop-1 prediction agrees with the loop-2 prediction, and the distribution of
accepted lengths.

## Part 2: ranking with the research note

The [research note](results/research_20261018/research.md) has supporting files
`analyze.py`, `limits.json` and `head_screening.json` in the same directory.
It reports CPU analysis and a literature search, with no new GPU benchmarks
or measured speedups. Its main proposals are:

1. speculative decoding with Dory's loops as the main research branch (same as
   idea B), with WaveFront and LoopSpec as designs to start from, n-gram
   (prompt-lookup) speculation as a cheap control, and same-history acceptance
   measurement first;
2. using the INT8 head only to screen candidates, then recomputing those
   candidates with the BF16 head, so greedy decoding keeps the exact BF16 winner
   (a CPU pilot on random hidden states kept a median of 37 of 131,072 rows);
3. FP8 KV cache for the full-attention layers at long contexts;
4. targeted kernel work: Nsight Compute counters on the gate+up and q/k/v+gate
   GEMVs first, a non-tensor-core attention variant for the 4 query heads per
   KV head, rebuilding the residual input inside the next projection, and a
   persistent layer-pair prototype (idea A) before a whole decoder;
5. recovering the 1.85 GiB fused weight copy (10.7% of the KV cache) with a
   layout shared by the batched and single-token paths, and judging speed by
   time to a correct completed answer.

I checked its most consequential external claim. The vLLM Dory pull request
author did write, on 2026-10-14, that they "already have an initial native
self-speculative decoding implementation, using 1 loop for drafting and 4 for
verification, with shared weights and reuse of early-loop KV caches and
activations" ([vllm#60149 comment](https://github.com/vllm-project/vllm/pull/60149#issuecomment-6168429418)).
The comment does not say that code is public. I did not verify the papers the
note cites.

### Where I agree and disagree with the note

- **Speculation:** I agree it has the highest ceiling. Its "ideal 1.41x at 2 loops" is a
  weight-byte bound that assumes every draft is accepted. With measured
  component times and the cost of wrong drafts, I estimate 9 to 17% at 80 to 90%
  agreement, about 3% at 70%, and nothing below that. At 4 loops the case is much stronger,
  which matches the upstream author's choice of 4 verification loops.
- **Head screening:** it is a sound way to keep exact BF16 greedy decisions, but
  Dory's default generation samples with top-p 0.95, which needs the full
  distribution. It therefore does not help the default path. E7's INT8 head
  already passes the accuracy gate and helps sampling as well.
- **Persistent kernel:** the note points out that idle time between kernels is
  only 0.04 ms. That is true, but it is not the cost a persistent kernel
  removes: the kernel start-up and drain time shows up as busy time inside each
  GEMV. The streaming-read measurements above put it at roughly 0.4 ms.
- **Rebuilding the residual input in the next projection:** I expect this to
  lose. Every program of the next GEMV reads the whole input vector; rebuilding
  it from the input, output and alpha vectors doubles that read (gate+up's 3,584
  programs would each read about 20 KB instead of 10 KB from L2). E2 showed the
  fold's final pass is latency-bound, not the part this would save.
- **Nsight Compute counters before more kernel work:** I agree. They are cheap,
  and they tell us whether the GEMVs are limited by DRAM, by decoding work or
  by latency.
- **Serving capacity and long-context KV:** both are valid and now easier, because
  the decode attention kernel is ours. They improve batch capacity and long
  contexts, not single-user speed at the contexts we benchmark.

### Combined ranking

Ranked by expected single-user gain per unit of effort, with batch impact
considered.

| Rank | Idea | Source | Expected single-user gain | Confidence | Effort | Changes numerics | Batch impact |
|---:|---|---|---|---|---|---|---|
| 1 | Enable the INT8 LM head after a task-quality check | E7 (mine); the note's variant 2 | 0.35 ms, measured: 200.3 to 215.7 tok/s | High | About half a day (quality check) | Yes, passes the accuracy gate | -2% KV cache |
| 2 | Decision measurements: same-history loop-1/loop-2 agreement (at 2 and 4 loops), Nsight Compute counters on the GEMVs, vLLM's built-in n-gram speculation on a repetitive and a chat workload | Both | None directly; picks between ranks 3 and 4 | High | 2-3 days | No | None |
| 3 | Persistent kernel, starting with one MLP layer pair | Mine (A); the note's item 4 | 0.3-0.5 ms (6-10%), estimated | Medium | Weeks | No | None |
| 4 | Speculative decoding with Dory's loops | Both (B; the note's item 1) | 9-17% at 2 loops if agreement is 80-90%; much more at 4 loops | Low until rank 2 measures agreement | Many weeks; upstream may deliver part of it | Sampling-equivalent if done right | Must switch off under load |
| 5 | Shared weight layout to recover the 10.7% KV cache | The note's item 5 | None | Medium | Weeks | No | Recovers capacity |
| 6 | FP8 KV cache for full-attention layers | The note's item 3 | 0.02 ms at 640 tokens, 0.23 ms at 8K, 0.93 ms at 32K (byte estimates) | Medium | 1-2 weeks plus quality work | Yes | More capacity |
| 7 | Exact-greedy head screening | The note's item 2 | Up to about 0.3 ms, greedy only | Medium | 1-2 weeks | No | -2% KV cache |
| 8 | Small kernel probes (attention for 4 query heads without padding, residual rebuild) | The note's item 4 | 0.05 ms or less each, based on E2, E9 and E11 | Low | Days each | No | None |

## Recommendation

1. **Now:** run a short task-quality check of the INT8 head, for example a few
   hundred math and multiple-choice questions, fused with and without it. If
   there is no measurable drop, turn it on by default. It is the only remaining
   gain that is already measured (+7.7% single-user tokens/s) and costs half a
   day.
2. **Next 2-3 days:** run the three decision measurements in rank 2. They help
   determine which approach warrants several weeks of implementation work.
3. **Then choose:**
   - If Dory will mostly be served at its default 2 loops, or loop-1/loop-2
     agreement is below about 90%, build the persistent-kernel prototype (rank 3).
     Its gain is smaller in principle but far more certain, and it helps every
     context length.
   - If 4-loop serving matters, or agreement is at least about 90%, pursue
     speculation. First ask the vLLM pull request author about their
     implementation instead of building one in parallel. Our part would then be
     what upstream will need from us: fast fused kernels for verifying 2 to 5
     tokens at once.
4. **Only if a goal requires them:** batch KV capacity (rank 5) and long-context
   KV compression (rank 6).

## Results of the decision measurements (2026-10-18)

Recorded as E13 to E15 in [EXPERIMENTS.md](EXPERIMENTS.md); the INT8 head is
now the default (E12, 4.70 ms per token).

- Loop-1 drafts agree with loop-2 verification on 94.2% of tokens (93.5% for
  4 loops); a 4-token draft gets 3.45 tokens accepted on average. That is well
  above the 90% threshold above, so speculation with Dory's loops goes first.
  Estimated gain: about 23% at 2 loops and 1.7 times at 4 loops, assuming a
  multi-token verification path that costs 1.3 times a single token for 5
  tokens.
- Kernel launch boundaries cost 2.3 to 4.0 us each; one long launch reaches 94
  to 95% of peak bandwidth with the existing GEMVs, and a grid-wide barrier
  costs 0.5 to 0.8 us. A persistent kernel could recover about 0.5 ms per token
  (11%), and it would also speed up the single-token draft steps.
- vLLM's n-gram speculation already works with Dory and the fused path: 3.12
  times faster on a code-copying workload, 5% slower on chat with thinking.
