# Dory cache, precision, and wavefront validation

The author's prefill objection is supported by a controlled comparison: with
shared KV, changing only prefill chunking changes the reference computation.
Large full-vocabulary log-probability differences also occur with matched
chunks, so chunking alone does not explain every numerical discrepancy. The
direct attention audit found no cache mismatch in the tested FP32 execution.
Our four-loop shared-cache wavefront adds a measurable approximation; per-loop
wavefront execution matches its forced catch-up control exactly in this suite.

This investigates [the upstream report](https://github.com/vllm-project/vllm/pull/60149#issuecomment-6270937659)
and [the author's reply](https://github.com/vllm-project/vllm/pull/60149#issuecomment-6276896577).
It does not reproduce their H100 hardware or the reporter's full-model FP64
reference, and it does not rerun the model-card accuracy benchmarks.

The main suite completed 44 jobs successfully on one RTX 4090. Five prompt
cases cover a short natural prompt with 256 generated tokens, random and
repeated natural text with 1,000 prompt tokens and 128 generated tokens, and
the same text types with 1,100 prompt tokens and 64 generated tokens. The
random prompt uses the report's seed 7 and token range [1000, 131072). The
1,000-token cases cross the 1,024-token window during generation. Natural
prompts are synthetic repeated prose, not a representative task benchmark.

Each comparison uses identical prompt and completion tokens. Engine controls
generate greedy continuations; references replay those continuations, and our
fused, catch-up, and wavefront paths are teacher-forced to the matching NVFP4
control. Prefill chunks are recorded from actual engine forwards. Prefix
caching is disabled, the outer model executes eagerly, and one request runs
at a time. Nonfinal prefill chunks produce no recorded next-token prediction.
The first, final-prefill prediction is excluded from decode aggregates.

The control imports the unchanged Dory model file at vLLM PR commit
`55076950b6c155f384d6766d63563c3bc20a0e3f` into the installed vLLM runtime.
The reference retains those projection and nGPT operations while replacing
paged attention with contiguous PyTorch math attention. BF16 engines use
Triton attention. FP32 controls use FlexAttention because the unchanged
FP32/head-256 Triton prefill kernel exceeds the 4090's shared-memory limit.
The one-loop control produces bitwise-identical logits in both cache modes.
All 212 raw logit arrays contain only finite values.

Changing only prefill chunking in an FP32 reference, on the 1,100-token random
prompt and its fixed BF16-control continuation, gives:

| Cache mode, 2 loops | RMS difference in log probability | Same top-token prediction | Mean total variation |
|---|---:|---:|---:|
| Per-loop: token prefill versus matched chunk | 0.0000207 | 63/63 | 0.000000306 |
| Shared: token prefill versus matched chunk | 0.66807 | 63/63 | 0.00625 |

This supports the author's explanation. Within a shared-cache chunk, earlier
tokens supply the current loop's KV; across chunks, they supply final-loop KV.
A token-by-token reference and a whole-prompt engine therefore compute
different functions even with the same weights, token history, and precision.
The 512-token prefill controls also have matched reference comparisons in
[the complete table](comparisons.md).

Matching precision and chunks reduces the discrepancy substantially in most
FP32 controls, but a late exception remains:

| FP32 engine versus matched FP32 reference, 2 loops | Per-loop RMS | Shared RMS | Shared top-token agreement |
|---|---:|---:|---:|
| Short natural prompt | 0.000133 | 0.0000186 | 255/255 |
| 1,100-token random prompt | 0.0000324 | 0.000113 | 63/63 |
| 1,000-token random prompt, decode positions before 1024 | 0.0000112 | 0.0000938 | 24/24 |
| Same prompt, positions from 1024 | 0.00000845 | 0.35630 | 103/103 |

The shared-cache error does not jump at position 1024: its row RMS there is
0.0000747, with larger excursions appearing later. Mean total variation in
the post-boundary rows is 0.000427 despite the much larger unweighted
log-probability RMS. The latter gives extremely unlikely vocabulary entries
the same weight as likely tokens, so it is not a task-accuracy measure.

A direct audit checks the engine's own query/key/value inputs, rather than
letting two complete models evolve separately. It mirrors every layer's KV
in a contiguous cache, compares every retained paged entry exactly, and
compares the last query of each attention call with FP64 PyTorch attention.
Across the two FP32 shared-cache random cases, all 10,752 attention calls have
zero cache-entry error. The largest attention-output absolute error is
0.000004776; the largest relative L2 error is 0.00001472. The audit leaves
the saved original-engine logits bitwise unchanged. This is evidence against
a cache read/write or window-mask error in these executions. The late
full-model drift is consistent with amplification of small numerical
differences; it is not proof that every backend or workload is correct.

The same audit also passes on BF16/Triton, the attention path used in our
benchmarks. All 10,752 calls again have zero cache-entry error, with maximum
attention-output absolute error 0.001192 against FP64 math. Its engine logits
also remain bitwise unchanged. See [audit summaries](attention-summary.json)
and [all per-call errors](attention-audit.json.gz).

Our implementations use the NVFP4 checkpoint and are compared with the
unchanged upstream Dory model on that same checkpoint. Forced catch-up runs
each committed token sequentially using the wavefront's own kernels. Each
configuration contributes 635 decode positions across the five cases:

| Loops | Cache mode | Wavefront versus catch-up RMS | Same top-token prediction | Mean total variation | Steady steps exercised |
|---:|---|---:|---:|---:|---:|
| 2 | Per-loop | 0 | 635/635 | 0 | 611 |
| 4 | Per-loop | 0 | 635/635 | 0 | 607 |
| 2 | Shared | 0 | 635/635 | 0 | 615 |
| 4 | Shared | 0.44511 | 633/635 | 0.00301 | 615 |

All configurations exercised the intended wavefront path with zero allocation
fallbacks. Zero difference here means bitwise-identical saved logits on these
histories, not a universal guarantee. The four-loop shared-cache result is
the expected approximation from reading older in-flight tokens before their
final loop. Against upstream NVFP4, the fused path has 99.37% to 99.84%
top-token agreement across these configurations, and wavefront has 99.53% to
99.69%. These numerical differences are separate from the historical GSM8K
and MMLU scores, and should not be described as exact distributional agreement.

The original 64-prompt quality gate never crossed the sliding window. This
suite extends numerical coverage but establishes no long-context task score
or model-card claim. The measured prefill confound is real, the reported
numerical concern deserves careful qualification, and the available evidence
does not establish a general paged-cache bug.

Reproduce with `.venv/bin/python -I bench/window_agreement.py run --out-dir
logs/window-agreement-NEW`, then run `bench/window_attention_audit.py` with
that directory. Use `--dtype bfloat16` for the Triton audit. Successful jobs
are reused by `run --resume`; their original harness revisions are retained
in provenance. Raw logits and job logs stay in the ignored local run
directory. Published evidence includes [all aggregate comparisons](summary.json),
[per-position metrics](metrics.json.gz), [inputs and forced histories](inputs.json),
and [versions and successful job statuses](provenance.json).
