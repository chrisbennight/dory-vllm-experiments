# Benchmarks

All measurements come from one machine using the benchmark tools in `bench/`.
`bench/publish_tables.py` generates the tables from the run directories in
`results/`.

## Setup

- **Hardware:** one NVIDIA GeForce RTX 4090 (24 GB, 1008 GB/s, 72 MB L2,
  compute capability 8.9).
- **Software:** vLLM at the head of the Dory support pull request
  ([vllm#60149](https://github.com/vllm-project/vllm/pull/60149), commit
  `55076950`), torch 2.14.0+cu130, V2 model runner, asynchronous scheduling,
  per-loop or shared KV cache as labeled, `max_model_len` 8192,
  `gpu_memory_utilization` 0.90.
- **Model:** NVIDIA-Nemotron-Labs-Dory1.0-4B in its three official
  quantizations (BF16, MXFP8, NVFP4) at pinned revisions
  (`bench/download_models.sh`). "Loops" is `n_recurrent_loops`, how many times
  the shared recurrent block runs per token. More loops can improve output
  quality at the cost of more computation.
- **Workload:** `vllm bench serve` with random-token prompts, in the two shapes
  of NVIDIA's DGX Spark table: 128 input / 1024 output tokens and 2560 / 300.
  1, 2, 4, 8, 16, 32 and 64 concurrent users (8, 8, 16, 32, 64, 128 and
  128 requests), sampled with the model's defaults (temperature 1, top-p 0.95).
- **Prefix caching:** disabled in the corrected serving suite. Every benchmark
  uses seed 0, so keeping a server's prefix cache across benchmarks lets later
  rows reuse earlier prompts. The headline, feature, final, shared-cache and
  complete-suite tables use the corrected runs. The older attention-backend,
  quantization and development-history tables retain their original runs,
  which had prefix caching enabled.
- **Values:** total output tokens per second, so with one user it is that
  user's speed. Runs are strictly one at a time on an idle GPU (the harness
  refuses to start otherwise). The complete serving suite does not estimate
  uncertainty for every row. Cache-limited rows are discussed below. An earlier
  2-loop shared-cache wavefront run had a concurrent workload, confirmed by
  the operator; it is excluded and replaced by a complete clean run.

## 1. vLLM's attention backends

Unmodified vLLM with the NVFP4 checkpoint. vLLM picks FlashAttention by
default; Triton and FlashInfer are selected with `--attention-backend`.
These historical runs had prefix caching enabled; their prompt-heavy batch
results should not be compared directly with the corrected suite below.

### 1 loop

<!-- table:backends-1 -->
| Input/output | Users | FLASH_ATTN (default) | TRITON_ATTN | FLASHINFER |
|---|---:|---:|---:|---:|
| 128/1024 | 1 | 188.8 | 234.8 | 229.5 |
| 128/1024 | 16 | 2204.2 | 2523.5 | 2536.3 |
| 128/1024 | 64 | 3453.5 | 4702.5 | 4613.8 |
| 2560/300 | 1 | 154.7 | 201.9 | 204.4 |
| 2560/300 | 16 | 911.5 | 952.8 | 1002.6 |
| 2560/300 | 64 | 1312.1 | 1576.2 | 1611.8 |

Runs: 20261018T045319Z, 20261018T114127Z, 20261018T130843Z.
<!-- /table -->

### 2 loops

<!-- table:backends-2 -->
| Input/output | Users | FLASH_ATTN (default) | TRITON_ATTN | FLASHINFER |
|---|---:|---:|---:|---:|
| 128/1024 | 1 | 128.1 | 161.7 | 158.2 |
| 128/1024 | 16 | 1488.2 | 1720.4 | 1726.6 |
| 128/1024 | 64 | 2281.0 | 3144.3 | 3090.8 |
| 2560/300 | 1 | 104.3 | 137.8 | 140.6 |
| 2560/300 | 16 | 661.3 | 650.4 | 687.7 |
| 2560/300 | 64 | 659.3 | 750.0 | 780.3 |

Runs: 20261018T045319Z, 20261018T114127Z, 20261018T130843Z.
<!-- /table -->

### 4 loops

<!-- table:backends-4 -->
| Input/output | Users | FLASH_ATTN (default) | TRITON_ATTN | FLASHINFER |
|---|---:|---:|---:|---:|
| 128/1024 | 1 | 77.9 | 99.8 | 97.5 |
| 128/1024 | 16 | 903.2 | 1048.4 | 1055.4 |
| 128/1024 | 64 | 1164.7 | 1578.5 | 1563.5 |
| 2560/300 | 1 | 63.2 | 84.6 | 86.4 |
| 2560/300 | 16 | 377.9 | 397.5 | 422.5 |
| 2560/300 | 64 | 391.7 | 420.4 | 446.8 |

Runs: 20261018T045319Z, 20261018T114127Z, 20261018T130843Z.
<!-- /table -->

- For a single request, FlashAttention spends 2.3 ms per token in attention at
  2 loops and a 640-token context, against 0.55 ms for Triton
  (`results/attention_backends.md`). Triton and FlashInfer are 22 to 37%
  faster than FlashAttention for one user and 34 to 38% faster for 64 users
  with short prompts.
- Triton leads with short prompts; FlashInfer leads with 2560-token prompts.
  The rest of this work uses Triton, whose KV cache layout the fused kernels
  read directly.
- vLLM's FlexAttention backend fails on this model unless
  `patches/vllm-flex-attention-pr57300-port.patch` is applied. With the patch,
  it remains the slowest backend tested (73 tok/s in the 2-loop decode profile).

## 2. Quantization: 16, 8 and 4 bits

The three official checkpoints on unmodified vLLM with Triton attention;
"Change" is NVFP4 against BF16.
These historical runs had prefix caching enabled, as did the default-backend
comparison below.

### 1 loop

<!-- table:quants-1 -->
| Input/output | Users | BF16 | MXFP8 | NVFP4 | Change |
|---|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 111.8 | 169.8 | 234.8 | +110% |
| 128/1024 | 16 | 1418.6 | 2011.4 | 2523.5 | +78% |
| 128/1024 | 64 | 3768.7 | 4390.9 | 4702.5 | +25% |
| 2560/300 | 1 | 103.8 | 150.0 | 201.9 | +94% |
| 2560/300 | 16 | 749.8 | 833.8 | 952.8 | +27% |
| 2560/300 | 64 | 1058.3 | 1458.5 | 1576.2 | +49% |

Runs: 20261018T114127Z, 20261018T121113Z.
<!-- /table -->

### 2 loops

<!-- table:quants-2 -->
| Input/output | Users | BF16 | MXFP8 | NVFP4 | Change |
|---|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 74.2 | 114.8 | 161.7 | +118% |
| 128/1024 | 16 | 940.9 | 1352.2 | 1720.4 | +83% |
| 128/1024 | 64 | 2151.6 | 2918.6 | 3144.3 | +46% |
| 2560/300 | 1 | 69.0 | 101.0 | 137.8 | +100% |
| 2560/300 | 16 | 502.4 | 563.9 | 650.4 | +29% |
| 2560/300 | 64 | 638.9 | 679.0 | 750.0 | +17% |

Runs: 20261018T114127Z, 20261018T121113Z.
<!-- /table -->

### 4 loops

<!-- table:quants-4 -->
| Input/output | Users | BF16 | MXFP8 | NVFP4 | Change |
|---|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 44.4 | 69.7 | 99.8 | +125% |
| 128/1024 | 16 | 562.6 | 817.5 | 1048.4 | +86% |
| 128/1024 | 64 | 1081.5 | 1376.1 | 1578.5 | +46% |
| 2560/300 | 1 | 41.2 | 61.3 | 84.6 | +105% |
| 2560/300 | 16 | 303.1 | 342.6 | 397.5 | +31% |
| 2560/300 | 64 | 330.6 | 363.9 | 420.4 | +27% |

Runs: 20261018T114127Z, 20261018T121113Z.
<!-- /table -->

<details>
<summary>The same with vLLM's default attention backend</summary>

#### 1 loop

<!-- table:quants-default-1 -->
| Input/output | Users | BF16 | MXFP8 | NVFP4 | Change |
|---|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 100.1 | 143.6 | 188.8 | +89% |
| 128/1024 | 16 | 1302.8 | 1791.1 | 2204.2 | +69% |
| 128/1024 | 64 | 2902.8 | 3267.1 | 3453.5 | +19% |
| 2560/300 | 1 | 89.7 | 121.8 | 154.7 | +72% |
| 2560/300 | 16 | 716.9 | 802.6 | 911.5 | +27% |
| 2560/300 | 64 | 903.9 | 1230.2 | 1312.1 | +45% |

Runs: 20261018T035249Z, 20261018T041933Z, 20261018T045319Z.
<!-- /table -->

#### 2 loops

<!-- table:quants-default-2 -->
| Input/output | Users | BF16 | MXFP8 | NVFP4 | Change |
|---|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 66.3 | 96.2 | 128.1 | +93% |
| 128/1024 | 16 | 860.7 | 1200.2 | 1488.2 | +73% |
| 128/1024 | 64 | 1662.9 | 2157.4 | 2281.0 | +37% |
| 2560/300 | 1 | 59.4 | 81.5 | 104.3 | +76% |
| 2560/300 | 16 | 482.1 | 540.3 | 661.3 | +37% |
| 2560/300 | 64 | 567.4 | 594.1 | 659.3 | +16% |

Runs: 20261018T035249Z, 20261018T041933Z, 20261018T045319Z.
<!-- /table -->

#### 4 loops

<!-- table:quants-default-4 -->
| Input/output | Users | BF16 | MXFP8 | NVFP4 | Change |
|---|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 39.5 | 57.9 | 77.9 | +97% |
| 128/1024 | 16 | 512.8 | 721.8 | 903.2 | +76% |
| 128/1024 | 64 | 883.2 | 1053.8 | 1164.7 | +32% |
| 2560/300 | 1 | 35.4 | 49.0 | 63.2 | +79% |
| 2560/300 | 16 | 290.3 | 327.2 | 377.9 | +30% |
| 2560/300 | 64 | 309.3 | 336.6 | 391.7 | +27% |

Runs: 20261018T035249Z, 20261018T041933Z, 20261018T045319Z.
<!-- /table -->

</details>

- Decoding one token reads every weight once per loop, so a lone request is
  bound by memory bandwidth: NVFP4 (about 0.56 bytes per weight with its
  scales) is about twice as fast as BF16 for one user. With many users the
  weight read is shared across the batch and the gap narrows.
- NVFP4 is the fastest in every configuration tested here. The remaining
  benchmarks use it.

## 3. NVFP4, feature by feature

Each column adds one feature in the order described in the
[README](../README.md). Cells show throughput, with the change from the
previous column in parentheses. The fused decode column uses vLLM's BF16 LM
head, allowing the next column to show the INT8 head's effect separately.
Wavefront decoding requires at least two loops. The shared KV cache column runs
the wavefront on Dory's shared KV cache (section 5); with one loop there is
nothing to share. These tables use the corrected suite with prefix caching off.

After the Triton column, batched rows continue to use vLLM's own kernels.
Differences in those rows need care: the 4-loop, 64-user, 128/1024 case fills
the per-loop KV cache, while shared caching has room for the workload
(section 5). Section 7 includes the intermediate concurrency levels.

### 1 loop

<!-- table:steps-1 -->
| Input/output | Users | vLLM as shipped | + Triton attention | + fused decode path | + INT8 LM head | + wavefront | + shared KV cache | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 188.2 | 234.8 (+25%) | 286.2 (+22%) | 318.8 (+11%) | n/a | n/a | +69% |
| 128/1024 | 16 | 2201.1 | 2523.0 (+15%) | 2524.2 (+0%) | 2527.5 (+0%) | n/a | n/a | +15% |
| 128/1024 | 64 | 3421.4 | 4648.9 (+36%) | 4628.7 (-0%) | 4637.5 (+0%) | n/a | n/a | +36% |
| 2560/300 | 1 | 152.2 | 199.1 (+31%) | 241.7 (+21%) | 264.9 (+10%) | n/a | n/a | +74% |
| 2560/300 | 16 | 956.3 | 996.5 (+4%) | 997.2 (+0%) | 996.9 (-0%) | n/a | n/a | +4% |
| 2560/300 | 64 | 1055.5 | 1256.2 (+19%) | 1256.9 (+0%) | 1256.1 (-0%) | n/a | n/a | +19% |

Runs: 20261019T152606Z, 20261019T232759Z, 20261020T002938Z, 20261020T010402Z.
<!-- /table -->

### 2 loops

<!-- table:steps-2 -->
| Input/output | Users | vLLM as shipped | + Triton attention | + fused decode path | + INT8 LM head | + wavefront | + shared KV cache | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 127.7 | 161.9 (+27%) | 200.3 (+24%) | 215.3 (+7%) | 251.0 (+17%) | 264.1 (+5%) | +107% |
| 128/1024 | 16 | 1485.4 | 1718.5 (+16%) | 1718.9 (+0%) | 1713.5 (-0%) | 1717.0 (+0%) | 1716.3 (-0%) | +16% |
| 128/1024 | 64 | 2263.3 | 3108.3 (+37%) | 3104.2 (-0%) | 3100.2 (-0%) | 3111.1 (+0%) | 3103.2 (-0%) | +37% |
| 2560/300 | 1 | 103.3 | 136.3 (+32%) | 167.2 (+23%) | 177.3 (+6%) | 197.9 (+12%) | 208.7 (+5%) | +102% |
| 2560/300 | 16 | 627.8 | 655.7 (+4%) | 655.8 (+0%) | 654.4 (-0%) | 656.3 (+0%) | 654.4 (-0%) | +4% |
| 2560/300 | 64 | 657.4 | 776.9 (+18%) | 774.0 (-0%) | 780.0 (+1%) | 782.1 (+0%) | 814.8 (+4%) | +24% |

Runs: 20261019T143422Z, 20261019T152606Z, 20261019T232759Z, 20261020T002938Z, 20261020T010402Z, 20261020T015236Z.
<!-- /table -->

### 4 loops

<!-- table:steps-4 -->
| Input/output | Users | vLLM as shipped | + Triton attention | + fused decode path | + INT8 LM head | + wavefront | + shared KV cache | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 78.0 | 99.8 (+28%) | 125.7 (+26%) | 131.1 (+4%) | 194.2 (+48%) | 243.4 (+25%) | +212% |
| 128/1024 | 16 | 900.7 | 1048.4 (+16%) | 1049.2 (+0%) | 1047.8 (-0%) | 1049.5 (+0%) | 1050.1 (+0%) | +17% |
| 128/1024 | 64 | 1171.3 | 1591.2 (+36%) | 1557.9 (-2%) | 1548.3 (-1%) | 1550.8 (+0%) | 1864.7 (+20%) | +59% |
| 2560/300 | 1 | 62.5 | 83.4 (+33%) | 104.0 (+25%) | 107.7 (+4%) | 143.9 (+34%) | 168.2 (+17%) | +169% |
| 2560/300 | 16 | 372.2 | 389.1 (+5%) | 389.3 (+0%) | 388.5 (-0%) | 389.1 (+0%) | 389.6 (+0%) | +5% |
| 2560/300 | 64 | 392.9 | 422.9 (+8%) | 424.2 (+0%) | 421.4 (-1%) | 422.4 (+0%) | 480.6 (+14%) | +22% |

Runs: 20261019T143422Z, 20261019T150009Z, 20261019T152606Z, 20261019T232759Z, 20261020T002938Z, 20261020T010402Z.
<!-- /table -->

At 4 loops the decode step is 10.22 ms on unmodified vLLM with Triton
attention, 7.75 ms on the fused path, and the wavefront commits a token about
every 3.8 to 5.4 ms depending on the text (section 4).

### How the fused path got there (2 loops)

Each column records a benchmark run made when that stage was completed,
using the code available at the time. These historical runs had prefix caching
enabled, so prompt-heavy batch differences do not isolate each feature's effect.

<!-- table:features-2 -->
| Input/output | Users | vLLM default | Triton attention | + first fused decode path | + own attention kernel | + q/k norm in attention | + INT8 LM head | + wavefront |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 128.1 | 161.7 | 188.6 | 199.2 | 200.3 | 215.7 | 250.8 |
| 128/1024 | 16 | 1488.2 | 1721.1 | 1720.2 | 1719.3 | 1721.7 | 1721.5 | 1719.5 |
| 128/1024 | 64 | 2281.0 | 3145.9 | 3143.9 | 3142.8 | 3142.1 | 3144.8 | 3145.0 |
| 2560/300 | 1 | 104.3 | 138.0 | 156.6 | 169.0 | 169.2 | 180.8 | 201.8 |
| 2560/300 | 16 | 661.3 | 697.2 | 652.8 | 697.3 | 697.0 | 696.5 | 697.8 |
| 2560/300 | 64 | 659.3 | 779.3 | 748.2 | 777.6 | 776.8 | 784.9 | 784.7 |

Runs: 20261018T045319Z, 20261018T163313Z, 20261018T180516Z, 20261018T181035Z, 20261018T185825Z, 20261018T190327Z, 20261019T022147Z.
<!-- /table -->

The single-user decode step behind these stages, from the in-process profiler
(`bench/profile_decode.py`, 640-token prompt, 2 loops):

| Stage | Decode step | Change |
|---|---:|---:|
| Unmodified vLLM, Triton attention | 6.32 ms | |
| First fused decode path (fused NVFP4 GEMVs and sublayers) | 5.41 ms | -14% |
| Residual update folded into the GEMVs, KV cache write fused | 5.30 ms | -2% |
| Own decode attention kernel, output in the GEMV's input layout | 5.09 ms | -4% |
| q/k normalization and KV write inside the attention kernel | 5.05 ms | -1% |
| INT8 LM head | 4.70 ms | -7% |
| Bandwidth minimum (weights and BF16 LM head at 1008 GB/s) | 3.70 ms | |

## 4. Chat text, random text, and what the wavefront can reach

The wavefront keeps a request's next tokens in flight at different loop depths
and drafts each one from the shallow (loop-1) prediction; vLLM's sampled token
is checked against the draft at every step. At 4 loops:

- a **steady step** (the draft was right) runs the input block once, the
  recurrent block once for four tokens and the output block for two: about
  3.6 ms;
- a **catch-up step** (the draft was wrong) refills the pipeline from vLLM's
  token: about 13.5 ms.

So the time per token is about `a x 3.6 + (1 - a) x 13.5` ms, where `a` is how
often vLLM samples the drafted token:

| Text | Drafts sampled by vLLM | Predicted ms per token | Measured |
|---|---:|---:|---:|
| Chat prompts, greedy | 98% | 3.8 | 261 tok/s (3.83 ms) |
| Chat prompts, temperature 1, top-p 0.95 | 94.5% | 4.1 | 240 tok/s (4.16 ms) |
| Random-token prompts (this benchmark), temperature 1 | 87 to 92% | 4.4 to 4.9 | median 5.10 to 5.28 ms |

The chat measurements are in-process decode times from `bench/wavefront_smoke.py`,
with results in `results/wavefront_smoke/`. The random-prompt measurements are
median times per output token from the serving benchmark. Random-token prompts
produce less predictable continuations, so loop-1 drafts agree less often with
the full-depth model. Each miss costs about 3.7 steady steps. These prompts
were harder to draft than the chat examples tested here; performance on other
chat workloads may differ.

The following comparison measures agreement between shallow and full-depth
predictions on the same greedy histories from 64 chat prompts
(`results/loop_agreement_20261018T202039Z/report.md`):

| Draft loops | Verify loops | Top-token agreement | Distribution overlap |
|---:|---:|---:|---:|
| 1 | 2 | 94.2% | 0.944 |
| 1 | 4 | 93.5% | 0.936 |
| 2 | 4 | 97.9% | 0.976 |

Drafts follow vLLM's sampling: the engine adds the Gumbel noise vLLM's
sampler will draw for that position, so a sampled draft matches about as often
as the overlap of the two distributions allows, which is the most any single
draft can match. Measured: 94.5% with top-p 0.95 on chat prompts, against an
overlap of 0.936 measured without top-p on greedy histories.

**Limits.** A steady step reads each weight once, like a 1-loop decode: 1.86
GB of NVFP4 layer weights plus 0.34 GB of INT8 LM head, 2.2 ms at full
bandwidth, against 5.3 GB (5.2 ms) for a sequential 4-loop step. So:

| Bound at 4 loops | ms per token | tok/s | Over the fused path (7.75 ms) |
|---|---:|---:|---:|
| Steady steps only, today's kernels (every draft right) | 3.6 | 278 | 2.2x |
| Steady steps at full memory bandwidth | 2.2 | 460 | 3.5x |
| Sequential decoding at full memory bandwidth | 5.2 | 191 | 1.5x |

The wavefront's ceiling is set by the draft hit rate and by the steady step's
efficiency (60% of bandwidth today; launch boundaries alone cost up to 0.7 ms,
`results/launch_cost_20261018T200956Z.md`). See [Future work](FUTURE_WORK.md).

## 5. Baseline against this work

The table compares vLLM's default FlashAttention backend, unmodified vLLM with
Triton attention, and the changes in this repository. The final configuration
uses the fused decode path and INT8 LM head, plus wavefront decoding and the
shared KV cache at two or more loops. All three use NVFP4 and disable prefix
caching.

<!-- table:final -->
| Loops | Input/output | Users | vLLM as shipped | Unmodified vLLM, Triton attention | This work | Over vLLM as shipped | Over Triton attention |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 128/1024 | 1 | 188.2 | 234.8 | 318.8 | +69% | +36% |
| 1 | 128/1024 | 16 | 2201.1 | 2523.0 | 2527.5 | +15% | +0% |
| 1 | 128/1024 | 64 | 3421.4 | 4648.9 | 4637.5 | +36% | -0% |
| 1 | 2560/300 | 1 | 152.2 | 199.1 | 264.9 | +74% | +33% |
| 1 | 2560/300 | 16 | 956.3 | 996.5 | 996.9 | +4% | +0% |
| 1 | 2560/300 | 64 | 1055.5 | 1256.2 | 1256.1 | +19% | -0% |
| 2 | 128/1024 | 1 | 127.7 | 161.9 | 264.1 | +107% | +63% |
| 2 | 128/1024 | 16 | 1485.4 | 1718.5 | 1716.3 | +16% | -0% |
| 2 | 128/1024 | 64 | 2263.3 | 3108.3 | 3103.2 | +37% | -0% |
| 2 | 2560/300 | 1 | 103.3 | 136.3 | 208.7 | +102% | +53% |
| 2 | 2560/300 | 16 | 627.8 | 655.7 | 654.4 | +4% | -0% |
| 2 | 2560/300 | 64 | 657.4 | 776.9 | 814.8 | +24% | +5% |
| 4 | 128/1024 | 1 | 78.0 | 99.8 | 243.4 | +212% | +144% |
| 4 | 128/1024 | 16 | 900.7 | 1048.4 | 1050.1 | +17% | +0% |
| 4 | 128/1024 | 64 | 1171.3 | 1591.2 | 1864.7 | +59% | +17% |
| 4 | 2560/300 | 1 | 62.5 | 83.4 | 168.2 | +169% | +102% |
| 4 | 2560/300 | 16 | 372.2 | 389.1 | 389.6 | +5% | +0% |
| 4 | 2560/300 | 64 | 392.9 | 422.9 | 480.6 | +22% | +14% |

Runs: 20261019T150009Z, 20261019T152606Z, 20261019T232759Z, 20261020T010402Z, 20261020T015236Z.
<!-- /table -->

- The single-user rows are this work's target. Batched rows run vLLM's own
  kernels; the fused path's second copy of the layer weights takes 11 to 13%
  of the KV cache, which only matters where 64 users fill the cache (128/1024
  at 4 loops: 64 x 1152 tokens against about 57,000 tokens of cache).
- Against the default backend, batched rows also gain from Triton attention.

### With Dory's shared KV cache (2 and 4 loops)

Dory can also keep one KV cache per recurrent layer that every loop overwrites
(`recurrent_kv_cache_mode: "last_loop"`, the WaveFront paper's cross-recurrence
KV sharing), instead of one cache per loop. The fused path and the wavefront
both run on it; the wavefront then approximates sequential decoding (in-flight
tokens see older in-flight tokens at their latest loop, not their last), within
the accuracy gate's noise floor (section 6). "Change" is the last column against
the first.

#### 2 loops

<!-- table:shared-kv-2 -->
| Input/output | Users | Unmodified, per-loop KV | This work, per-loop KV | Unmodified, shared KV | Fused path, shared KV | This work, shared KV | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 161.9 | 251.0 | 161.6 | 216.1 | 264.1 | +63% |
| 128/1024 | 16 | 1718.5 | 1717.0 | 1718.2 | 1718.0 | 1716.3 | -0% |
| 128/1024 | 64 | 3108.3 | 3111.1 | 3107.7 | 3105.1 | 3103.2 | -0% |
| 2560/300 | 1 | 136.3 | 197.9 | 136.0 | 178.5 | 208.7 | +53% |
| 2560/300 | 16 | 655.7 | 656.3 | 655.6 | 655.7 | 654.4 | -0% |
| 2560/300 | 64 | 776.9 | 782.1 | 815.2 | 816.4 | 814.8 | +5% |

Runs: 20261019T143422Z, 20261019T160704Z, 20261019T232759Z, 20261020T000323Z, 20261020T015236Z.
<!-- /table -->

#### 4 loops

<!-- table:shared-kv-4 -->
| Input/output | Users | Unmodified, per-loop KV | This work, per-loop KV | Unmodified, shared KV | Fused path, shared KV | This work, shared KV | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 99.8 | 194.2 | 99.7 | 131.5 | 243.4 | +144% |
| 128/1024 | 16 | 1048.4 | 1049.5 | 1049.6 | 1049.7 | 1050.1 | +0% |
| 128/1024 | 64 | 1591.2 | 1550.8 | 1869.7 | 1869.8 | 1864.7 | +17% |
| 2560/300 | 1 | 83.4 | 143.9 | 83.5 | 107.9 | 168.2 | +102% |
| 2560/300 | 16 | 389.1 | 389.1 | 389.4 | 389.2 | 389.6 | +0% |
| 2560/300 | 64 | 422.9 | 422.4 | 480.8 | 480.3 | 480.6 | +14% |

Runs: 20261019T143422Z, 20261019T150009Z, 20261019T160704Z, 20261019T232759Z, 20261020T000323Z.
<!-- /table -->

- At 4 loops the shared cache holds 173,814 tokens on unmodified vLLM and
  151,915 with the fused path's weight copy, against 65,180 and 56,966
  per-loop. Comparing the wavefront configurations, shared KV raises 64-user
  throughput by 20% (128/1024) and 14% (2560/300).
- Drafts match more often on the shared cache: vLLM sampled 96.2% of them in
  these random-token runs (87 to 92% per-loop) and 97.1% on chat prompts with
  temperature 1 and top-p 0.95 (94.5% per-loop). In the corrected 4-loop suite,
  one user gains 85% over the fused path at 128/1024 (243.4 against 131.5
  tok/s) on shared KV, compared with 48% (194.2 against 131.1) per-loop.

## 6. Accuracy

The fused kernels compute in a different order than vLLM's, and the INT8 head
and multi-token kernels round differently, so outputs are not bit-identical.
The accuracy check compares each configuration with unmodified vLLM on 64
prompts of 256 greedy tokens (`bench/compare_outputs.py`). To pass, every
divergent token must be in the reference's top 5. The mean log-probability
difference must also be no larger than the difference between FlashInfer and
Triton attention, which serves as the noise floor. This comparison uses only
positions shared by every configuration (`bench/common_prefix_delta.py`).

| 4 loops, against unmodified vLLM with Triton | Divergent token outside top 5 | Mean abs logprob difference |
|---|---:|---:|
| FlashInfer attention (the noise floor) | 0 | 0.0070 |
| Fused path, INT8 LM head | 0 | 0.0066 |
| Fused path, INT8 LM head, wavefront | 0 | 0.0068 |

(`results/output_agreement_20261019T001920Z/report.md`.)

On the shared KV cache, against unmodified vLLM on the same cache (over the
6425 positions every configuration shares,
`results/output_agreement_20261019T115915Z/report.md`):

| 4 loops, shared KV, against unmodified vLLM with Triton | Divergent token outside top 5 | Mean abs logprob difference |
|---|---:|---:|
| FlashInfer attention (the noise floor) | 0 | 0.0072 |
| Fused path, INT8 LM head | 0 | 0.0066 |
| Fused path, INT8 LM head, wavefront | 0 | 0.0066 |

The shared cache itself, against per-loop caches on unmodified vLLM, drifts
more (0.0078 against a 0.0066 floor), as a different cache semantics should,
and leaves tasks unchanged: GSM8K 91.2% against 90.8%, MMLU 62.7% against
62.3% (`results/task_eval_20261019T105029Z/report.md`). With the wavefront on
it, GSM8K is 92.0% against 91.2% and MMLU 62.7% against 62.7%
(`results/task_eval_20261019T120832Z/report.md`). The INT8 head was
also checked on tasks at 2 loops (`results/task_eval_20261018T195326Z/report.md`):
GSM8K 89.6% against 90.4% with the BF16 head (250 problems, McNemar p = 0.69)
and MMLU 60.8% against 60.7% (1000 questions, p = 1.00).

With per-loop caches, every token is drawn by vLLM's sampler from the full-depth
logits, so drafting preserves the sequential decoder's sampling distribution
apart from numerical differences. On the shared cache, the full-depth logits
also reflect the approximation described in section 5.

### Agreement across the sliding window

The 64-prompt agreement gate above has prompt lengths of 23 to 42 tokens and
never reaches the 1,024-token sliding window. The separate
[controlled validation](../results/window_agreement_20261020/report.md)
completed 44 jobs with matched token histories, recorded prefill chunks,
contiguous PyTorch attention references, and single-request generation across
the boundary. These are numerical measurements, separate from task accuracy.

With 2 loops and the same FP32 reference weights, changing only prefill from
one chunk to individual tokens gives full-vocabulary log-probability RMS
differences of 0.66807 on shared KV and 0.00002 on per-loop KV, on the tested
1,100-token random prompt and fixed continuations. Shared-cache chunking
therefore changes the reference computation itself.

The NVFP4 wavefront comparison against forced catch-up uses 635 decode
positions per configuration, including natural and random text on both sides
of the window:

| Loops | KV cache | RMS difference in log probability | Same top-token prediction |
|---:|---|---:|---:|
| 2 | per-loop | 0 | 635/635 |
| 4 | per-loop | 0 | 635/635 |
| 2 | shared | 0 | 635/635 |
| 4 | shared | 0.44511 | 633/635 |

The four-loop shared-cache wavefront is an approximation. Neither high
top-token agreement nor low task-score differences establish identical
sampling distributions. The report includes matched FP32 controls, direct
cache checks, total variation, and the limits of this comparison with the
upstream H100/FP64 report.

## 7. Complete serving suite, 1 to 64 users

All configurations below use NVFP4 and disable prefix caching. "Fused path"
uses the INT8 LM head; section 3 separately measures the BF16 head. Shared
caching and the wavefront do not apply at one loop. "Best over shipped" compares
the fastest measured configuration of this work in each row with vLLM's default
backend; it is not a single serving configuration.

### 1 loop

<!-- table:suite-1 -->
| Input/output | Users | vLLM as shipped | Triton attention | Fused path | Fused + wavefront | Triton, shared KV | Fused, shared KV | Fused + wavefront, shared KV | Best over shipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 188.2 | 234.8 | 318.8 | n/a | n/a | n/a | n/a | +69% |
| 128/1024 | 2 | 372.3 | 455.3 | 454.8 | n/a | n/a | n/a | n/a | +22% |
| 128/1024 | 4 | 732.4 | 857.7 | 857.6 | n/a | n/a | n/a | n/a | +17% |
| 128/1024 | 8 | 1401.8 | 1538.1 | 1539.0 | n/a | n/a | n/a | n/a | +10% |
| 128/1024 | 16 | 2201.1 | 2523.0 | 2527.5 | n/a | n/a | n/a | n/a | +15% |
| 128/1024 | 32 | 3037.7 | 3620.8 | 3620.8 | n/a | n/a | n/a | n/a | +19% |
| 128/1024 | 64 | 3421.4 | 4648.9 | 4637.5 | n/a | n/a | n/a | n/a | +36% |
| 2560/300 | 1 | 152.2 | 199.1 | 264.9 | n/a | n/a | n/a | n/a | +74% |
| 2560/300 | 2 | 282.6 | 353.1 | 352.4 | n/a | n/a | n/a | n/a | +25% |
| 2560/300 | 4 | 488.5 | 560.8 | 561.1 | n/a | n/a | n/a | n/a | +15% |
| 2560/300 | 8 | 767.7 | 793.0 | 793.2 | n/a | n/a | n/a | n/a | +3% |
| 2560/300 | 16 | 956.3 | 996.5 | 996.9 | n/a | n/a | n/a | n/a | +4% |
| 2560/300 | 32 | 1072.5 | 1140.1 | 1142.1 | n/a | n/a | n/a | n/a | +6% |
| 2560/300 | 64 | 1055.5 | 1256.2 | 1256.1 | n/a | n/a | n/a | n/a | +19% |

Runs: 20261019T152606Z, 20261019T232759Z, 20261020T010402Z.
<!-- /table -->

### 2 loops

<!-- table:suite-2 -->
| Input/output | Users | vLLM as shipped | Triton attention | Fused path | Fused + wavefront | Triton, shared KV | Fused, shared KV | Fused + wavefront, shared KV | Best over shipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 127.7 | 161.9 | 215.3 | 251.0 | 161.6 | 216.1 | 264.1 | +107% |
| 128/1024 | 2 | 253.2 | 313.5 | 312.4 | 313.0 | 313.1 | 313.5 | 312.8 | +24% |
| 128/1024 | 4 | 498.9 | 589.1 | 587.1 | 588.5 | 588.2 | 588.6 | 587.5 | +18% |
| 128/1024 | 8 | 954.5 | 1054.0 | 1050.6 | 1050.7 | 1051.8 | 1052.9 | 1051.9 | +10% |
| 128/1024 | 16 | 1485.4 | 1718.5 | 1713.5 | 1717.0 | 1718.2 | 1718.0 | 1716.3 | +16% |
| 128/1024 | 32 | 2030.5 | 2440.0 | 2439.5 | 2441.7 | 2444.0 | 2445.2 | 2438.1 | +20% |
| 128/1024 | 64 | 2263.3 | 3108.3 | 3100.2 | 3111.1 | 3107.7 | 3105.1 | 3103.2 | +37% |
| 2560/300 | 1 | 103.3 | 136.3 | 177.3 | 197.9 | 136.0 | 178.5 | 208.7 | +102% |
| 2560/300 | 2 | 189.5 | 239.1 | 238.4 | 239.0 | 238.7 | 238.7 | 238.8 | +26% |
| 2560/300 | 4 | 327.1 | 377.3 | 375.9 | 376.3 | 376.4 | 376.9 | 375.5 | +15% |
| 2560/300 | 8 | 507.9 | 526.8 | 524.3 | 526.4 | 526.1 | 526.4 | 524.5 | +4% |
| 2560/300 | 16 | 627.8 | 655.7 | 654.4 | 656.3 | 655.6 | 655.7 | 654.4 | +5% |
| 2560/300 | 32 | 699.1 | 744.9 | 743.4 | 745.3 | 745.1 | 745.1 | 743.8 | +7% |
| 2560/300 | 64 | 657.4 | 776.9 | 780.0 | 782.1 | 815.2 | 816.4 | 814.8 | +24% |

Runs: 20261019T143422Z, 20261019T152606Z, 20261019T160704Z, 20261019T232759Z, 20261020T000323Z, 20261020T010402Z, 20261020T015236Z.
<!-- /table -->

### 4 loops

<!-- table:suite-4 -->
| Input/output | Users | vLLM as shipped | Triton attention | Fused path | Fused + wavefront | Triton, shared KV | Fused, shared KV | Fused + wavefront, shared KV | Best over shipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 78.0 | 99.8 | 131.1 | 194.2 | 99.7 | 131.5 | 243.4 | +212% |
| 128/1024 | 2 | 154.5 | 192.8 | 192.8 | 193.3 | 192.9 | 193.1 | 192.8 | +25% |
| 128/1024 | 4 | 304.3 | 362.4 | 361.8 | 362.2 | 362.4 | 362.7 | 362.1 | +19% |
| 128/1024 | 8 | 583.9 | 645.5 | 644.0 | 645.6 | 645.5 | 646.0 | 645.2 | +11% |
| 128/1024 | 16 | 900.7 | 1048.4 | 1047.8 | 1049.5 | 1049.6 | 1049.7 | 1050.1 | +17% |
| 128/1024 | 32 | 1220.6 | 1480.3 | 1477.3 | 1479.7 | 1479.7 | 1480.8 | 1480.0 | +21% |
| 128/1024 | 64 | 1171.3 | 1591.2 | 1548.3 | 1550.8 | 1869.7 | 1869.8 | 1864.7 | +60% |
| 2560/300 | 1 | 62.5 | 83.4 | 107.7 | 143.9 | 83.5 | 107.9 | 168.2 | +169% |
| 2560/300 | 2 | 114.6 | 145.7 | 145.5 | 145.7 | 145.5 | 145.8 | 145.5 | +27% |
| 2560/300 | 4 | 196.7 | 227.8 | 227.5 | 227.6 | 227.7 | 227.9 | 227.9 | +16% |
| 2560/300 | 8 | 303.9 | 315.0 | 314.1 | 315.1 | 314.8 | 315.1 | 315.2 | +4% |
| 2560/300 | 16 | 372.2 | 389.1 | 388.5 | 389.1 | 389.4 | 389.2 | 389.6 | +5% |
| 2560/300 | 32 | 392.9 | 422.8 | 421.5 | 422.5 | 439.9 | 439.6 | 440.3 | +12% |
| 2560/300 | 64 | 392.9 | 422.9 | 421.4 | 422.4 | 480.8 | 480.3 | 480.6 | +22% |

Runs: 20261019T143422Z, 20261019T150009Z, 20261019T152606Z, 20261019T160704Z, 20261019T232759Z, 20261020T000323Z, 20261020T010402Z.
<!-- /table -->

Wavefront decoding currently runs only when one request is decoding. At 4 loops
and 128/1024 with shared KV, one user reaches 243.4 tok/s, while two users reach
192.8 tok/s in total, about 96.4 each. Supporting separate wavefront pipelines
for small batches is therefore a useful next step; see [Future work](FUTURE_WORK.md).

## Caveats

- All results come from one machine. The kernels' launch configurations are
  tuned for its RTX 4090.
- The 64-user rows that fill the KV cache (128/1024 at 4 loops) preempt
  requests and vary by about 1% between runs; their time to first token varies
  much more. With per-loop KV, the fused configurations are 2.1 to 2.7% slower
  than unmodified Triton in the corrected suite. This exceeds the original 2%
  batch target in EXPERIMENTS.md; the extra weight copy still reduces cache
  capacity. Shared KV avoids that capacity limit in this workload.
- The 128/1024 single-user row with the wavefront measured 201.7 to 201.9
  tok/s when it was a server's first benchmark and 184.5 when it followed the
  2560-token rows; the cause is not established.
- **Older serving runs reused prompt prefixes.** Repeated random-token
  benchmarks used the same seed on a server with prefix caching enabled.
  Later rows therefore reused some earlier prompt processing. In the first
  expanded suite, the cumulative prefix-cache hit rate reached 73%. This
  inflated throughput and shortened time to first token for prompt-heavy rows,
  explaining the earlier 2560/300 batch variation recorded in EXPERIMENTS.md,
  E23. With the cache disabled, Triton at 2 loops and 16 users measured 655.4
  tok/s, against 849.2 in that expanded suite with caching enabled. The
  corrected tables above disable prefix caching; the historical tables are
  labeled explicitly. This finding does not establish the cause of the
  separate wavefront order effect described above.
