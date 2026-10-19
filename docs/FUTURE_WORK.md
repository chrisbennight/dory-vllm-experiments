# Future work

These are possible improvements to single-user Dory decoding and the work
needed to evaluate them. Numbers refer to 4 loops on the RTX 4090 unless
stated otherwise. Experiment references such as E13 point to entries in
[EXPERIMENTS.md](../EXPERIMENTS.md).

## Done

Items from earlier plans that are finished, so they are not repeated below:

- **Fused single-token decode path**: NVFP4 GEMVs, fused attention and MLP
  sublayers, the residual update folded into the GEMVs (E2), the KV cache write
  fused (E3), our own decode attention kernel (E5) with the q/k normalization
  and cache write inside it (E9). Variants that did not pay off are recorded
  as rejected (E1, E1b, E1c, E4, E6, E8, E11), and the launch configurations
  were re-checked inside vLLM (E10).
- **INT8 LM head** (E7), checked on GSM8K and MMLU before becoming the default
  (E12).
- **The decision measurements** for what to build next: the cost of kernel
  launches (E13), vLLM's n-gram speculation (E14) and how often Dory's shallow
  and full-depth predictions agree (E15).
- **Speculation with Dory's own loops**, the main direction chosen from those
  measurements: multi-token GEMVs and attention (E16, E17), the stand-alone
  wavefront prototype (E18), sampling that keeps the output distribution
  (E19), cheaper wavefront steps (E20), and serving inside vLLM's V2 model
  runner (E22, E23) after measuring and rejecting the V1 runner's
  custom-proposer hook (E21). The serving work includes per-token block tables
  in the attention kernel, lookahead KV slots reserved only for a request
  decoding alone, and an allocation check before writing ahead.
- **Measurement**: the serving matrix, decode profiles, the accuracy gate
  against a noise floor, task checks, idle-GPU checks that also see graphics
  load, the attention-backend and quantization surveys, and a port of vLLM's
  FlexAttention fix.

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

## Shared KV cache across loops (cross-recurrence KV sharing)

WaveFront Decoding's cross-recurrence KV sharing
([arXiv 2609.23033](https://arxiv.org/abs/2609.23033), section 3.3) lets the
loops read and write one KV cache instead of one cache per loop. The paper
treats it as an optional, approximate extension: on Huginn-3.5B it raised
draft acceptance from 0.88 to 0.98 and the speedup over normal decoding from
2.54x to 4.81x, with accuracy within 0.7 points. Dory supports a form of it
natively, `recurrent_kv_cache_mode: "last_loop"` (one cache per recurrent
layer that every loop overwrites). This work uses per-loop caches throughout,
which keeps the wavefront exact against sequential decoding.

The first step is measured (E24): the shared cache on unmodified vLLM at 4
loops, against per-loop caches.

| 4 loops, unmodified vLLM | Per-loop KV | Shared KV (`last_loop`) |
|---|---:|---:|
| KV cache | 65,169 tokens | 173,814 tokens (2.67x) |
| One user, 128/1024 | 99.6 tok/s | 99.9 tok/s |
| 64 users, 128/1024 | 1583.8 tok/s | 1889.5 tok/s (+19%) |
| 64 users, 2560/300 | 419.2 tok/s | 637.8 tok/s (+52%) |
| GSM8K, 250 problems | 90.8% | 91.2% (McNemar p = 1.00) |
| MMLU, 1000 questions | 62.3% | 62.7% (p = 0.29) |
| Greedy agreement, 64 prompts | reference | no divergent token outside the top 5 |

The shared cache does not change one user's speed on its own (the weights
dominate a step at these context lengths), triples the KV capacity, and turns
the KV-limited 64-user rows into large gains. It changes predictions more than
switching attention backends does (log-probability drift 0.0078 against the
noise floor's 0.0066), as a different cache semantics should, but shows no
task-quality loss on GSM8K or MMLU.

Next steps:

1. **The wavefront on a shared cache.** The kernels only need every loop's
   cache offset and block table to point at the one cache; the engine refuses
   `last_loop` today. The wavefront then becomes approximate even against
   sequential `last_loop` decoding, because an in-flight token reads entries
   that a sequential step would already have overwritten with deeper loops.
   Measure the draft match rate (higher, if Dory behaves like Huginn), one
   user's speed, and accuracy and tasks against sequential `last_loop`
   decoding.
2. **The fused path on a shared cache.** It already reads the shared cache
   (`mixer.attn[0]`); confirm its speed and accuracy, which would also recover
   far more KV capacity than its duplicate weights cost.
3. **A serving choice.** For batch-heavy deployments at several loops, the
   shared cache is worth offering on its own; the choice between it and exact
   per-loop caches belongs to the deployment.

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
- **Bimodal long-prompt batch rows**: the 2560/300 rows with 16 and 64 users
  land on one of two levels, up to 11% apart, in runs of the same
  configuration (E23). Finding the cause would make those rows comparable
  across configurations; until then they need repeated runs.
