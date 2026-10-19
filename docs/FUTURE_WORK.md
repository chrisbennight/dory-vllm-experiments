# Future work

These are possible improvements to single-user Dory decoding and the work
needed to evaluate them. Numbers refer to 4 loops on the RTX 4090 unless
stated otherwise. Experiment references such as E13 point to entries in
[EXPERIMENTS.md](../EXPERIMENTS.md).

## Current blockers

- **The vLLM base is an open pull request.** Dory support is
  [vllm#60149](https://github.com/vllm-project/vllm/pull/60149), not yet merged
  and behind vLLM's main branch. The patch here applies to its head
  (`55076950`) only; moving to a newer vLLM means rebasing the patch and
  repeating the accuracy gate and the serving benchmarks.
- **The patch changes vLLM's scheduler and model runner.** The scheduler's lookahead, the V2
  runner's CUDA graph dispatch and its per-step hooks are written for one
  model. Contributing this upstream would require a general interface for
  models that run their own decode steps for a single request. That interface
  must specify which requests, KV slots and sampler state the model may use.
  The stale block-table entries that require the engine's allocation check
  also need review.
- **The fused path keeps a second copy of the layer weights** in its own
  layout. This reduces available KV cache capacity by 11 to 13%. Batched
  throughput falls where the cache is full (128/1024 with 64 users at 4 loops: 1 to 2%
  slower than unmodified vLLM, for the fused path with or without the
  wavefront). Removing it needs vLLM's batched GEMMs to read the fused layout,
  or the fused kernels to read vLLM's.
- **Narrow support:** an RTX 4090 (kernel launch configurations tuned for it),
  vLLM's Triton attention backend with a BF16 KV cache, tensor parallelism 1,
  the per-loop KV cache mode, and the V2 model runner. Other configurations
  fall back to vLLM's own path where the code detects them, but are not tuned.
- **The wavefront runs only for a request decoding alone.** With two or more
  users decoding at once, every step runs vLLM's batched path.

## Wavefront decoding

1. **Cheaper catch-up steps.** When vLLM samples a different token than the
   draft, a catch-up step rebuilds the pipeline in about 13.5 ms, 3.7 steady
   steps. On random-token prompts (87 to 92% of drafts sampled) catch-ups take
   a quarter to a third of decode time; on chat prompts (94.5 to 98%), 7 to
   18%.
   Options: overlap each refill's input block and draft readout with the next
   recurrent call on a second stream; fall back to a plain sequential step and
   refill lazily while the recent hit rate is low.
2. **A steady step closer to its bandwidth minimum.** A steady step takes about
   3.6 ms against 2.2 ms for its weight bytes at full bandwidth. Known costs: the
   multi-token kernels' epilogues (a layer pair costs 88 us for one token
   against 75 us on the single-token path), per-row residual kernels that could
   run as one, and kernel boundaries in general (up to 0.68 ms per token in the
   sequential path, E13), which a persistent kernel would remove.
3. **Reduce host synchronization.** The engine reads two numbers back from the
   GPU each step to choose between the steady and catch-up graphs. CUDA graph
   conditional nodes could make that choice on the GPU. Measure the cost first;
   it is likely tens of microseconds.
4. **Better drafts.** Drafting from loop 2 agrees with 4 loops 97.9% of the
   time, against 93.5% from loop 1 (E15), at the cost of a deeper pipeline;
   LoopSpec's second, deeper proposal on rejection is another option. Copy-heavy
   text could combine the wavefront with n-gram drafts, which triple speed when
   the output copies its context (E14).
5. **Small batches.** The multi-token kernels take up to 16 tokens, so one
   recurrent call could advance separate pipelines for two to four users.
   This needs per-request pipeline state and measurements to determine when
   it is faster than vLLM's batched path.
6. **Understand the order effect.** The single-user 128/1024 row measured 201.9
   tok/s as a server's first benchmark and 184.5 after the 2560-token rows,
   reproducibly; the cause is not established.

## Fused decode path

- **Persistent kernel**: one long-running kernel that streams the next layer's
  weights while waiting for the previous layer's result. In one long launch
  the GEMVs reach 94 to 95% of bandwidth against 59 to 83% per layer today
  (E13). It would speed up sequential steps and the wavefront's steady steps
  alike.
- **FP8 KV cache** for the full-attention layers, if contexts beyond 8K tokens
  become a goal.
- **Exact greedy decoding with the INT8 head**: use it to screen candidates and
  recompute the top few rows in BF16, if bit-exact greedy output against the
  BF16 head is required.
- **Other GPUs**: Hopper and Blackwell offer programmatic dependent launch and
  FP4 tensor cores, which change which of these kernels are worth keeping.

## Measurement

- Chat serving benchmarks: the serving harness uses random-token prompts,
  where wavefront gains were smaller than in the in-process chat tests. A
  dataset of chat prompts in `vllm bench serve` would measure that gain under
  serving conditions.
- Long contexts (8K) and the time to first token, which this work does not
  change.
