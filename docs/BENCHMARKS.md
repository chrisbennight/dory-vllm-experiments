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
  per-loop KV cache, `max_model_len` 8192, `gpu_memory_utilization` 0.90.
- **Model:** NVIDIA-Nemotron-Labs-Dory1.0-4B in its three official
  quantizations (BF16, MXFP8, NVFP4) at pinned revisions
  (`bench/download_models.sh`). "Loops" is `n_recurrent_loops`, how many times
  the shared recurrent block runs per token. More loops can improve output
  quality at the cost of more computation.
- **Workload:** `vllm bench serve` with random-token prompts, in the two shapes
  of NVIDIA's DGX Spark table: 128 input / 1024 output tokens and 2560 / 300.
  1, 16 and 64 concurrent users (8, 64 and 128 requests), sampled with the
  model's defaults (temperature 1, top-p 0.95).
- **Values:** total output tokens per second, so with one user it is that
  user's speed. Runs are strictly one at a time on an idle GPU (the harness
  refuses to start otherwise). Repeated runs generally agree to about 0.2%.
  Exceptions include rows limited by KV cache capacity, long-prompt batches
  and a wavefront result that depends on benchmark order; see Caveats below.

## 1. vLLM's attention backends

Unmodified vLLM with the NVFP4 checkpoint. vLLM picks FlashAttention by
default; Triton and FlashInfer are selected with `--attention-backend`.

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
nothing to share, and "pending" marks a run of the benchmark suite still in
progress.

After the Triton column, batched rows continue to use vLLM's own kernels.
Differences in those rows need care: the 2560/300 results at 16 and 64 users
vary by up to 11% between runs of the same configuration (see Caveats), and
the 4-loop, 64-user, 128/1024 case fills the KV cache (section 5).

### 1 loop

<!-- table:steps-1 -->
| Input/output | Users | vLLM as shipped | + Triton attention | + fused decode path | + INT8 LM head | + wavefront | + shared KV cache | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 188.8 | 234.8 (+24%) | 286.3 (+22%) | 318.6 (+11%) | n/a | n/a | +69% |
| 128/1024 | 16 | 2204.2 | 2523.5 (+14%) | 2524.3 (+0%) | 2531.8 (+0%) | n/a | n/a | +15% |
| 128/1024 | 64 | 3453.5 | 4702.5 (+36%) | 4703.7 (+0%) | 4677.7 (-1%) | n/a | n/a | +35% |
| 2560/300 | 1 | 154.7 | 201.9 (+31%) | 244.9 (+21%) | 268.7 (+10%) | n/a | n/a | +74% |
| 2560/300 | 16 | 911.5 | 952.8 (+5%) | 954.2 (+0%) | 1058.5 (+11%) | n/a | n/a | +16% |
| 2560/300 | 64 | 1312.1 | 1576.2 (+20%) | 1575.9 (-0%) | 1707.4 (+8%) | n/a | n/a | +30% |

Runs: 20261018T045319Z, 20261018T114127Z, 20261019T023002Z, 20261019T024248Z.
<!-- /table -->

### 2 loops

<!-- table:steps-2 -->
| Input/output | Users | vLLM as shipped | + Triton attention | + fused decode path | + INT8 LM head | + wavefront | + shared KV cache | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 128.1 | 161.7 (+26%) | 200.3 (+24%) | 215.7 (+8%) | 250.8 (+16%) | pending | +96% |
| 128/1024 | 16 | 1488.2 | 1721.1 (+16%) | 1721.7 (+0%) | 1721.5 (-0%) | 1719.5 (-0%) | pending | +16% |
| 128/1024 | 64 | 2281.0 | 3145.9 (+38%) | 3142.1 (-0%) | 3144.8 (+0%) | 3145.0 (+0%) | pending | +38% |
| 2560/300 | 1 | 104.3 | 138.0 (+32%) | 169.2 (+23%) | 180.8 (+7%) | 201.8 (+12%) | pending | +94% |
| 2560/300 | 16 | 661.3 | 697.2 (+5%) | 697.0 (-0%) | 696.5 (-0%) | 697.8 (+0%) | pending | +6% |
| 2560/300 | 64 | 659.3 | 779.3 (+18%) | 776.8 (-0%) | 784.9 (+1%) | 784.7 (-0%) | pending | +19% |

Runs: 20261018T045319Z, 20261018T180516Z, 20261018T185825Z, 20261018T190327Z, 20261019T022147Z.
<!-- /table -->

### 4 loops

<!-- table:steps-4 -->
| Input/output | Users | vLLM as shipped | + Triton attention | + fused decode path | + INT8 LM head | + wavefront | + shared KV cache | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 77.9 | 99.6 (+28%) | 125.7 (+26%) | 131.1 (+4%) | 184.5 (+41%) | 248.4 (+35%) | +219% |
| 128/1024 | 16 | 903.2 | 1049.0 (+16%) | 1050.9 (+0%) | 1048.4 (-0%) | 1049.7 (+0%) | 1050.9 (+0%) | +16% |
| 128/1024 | 64 | 1164.7 | 1583.8 (+36%) | 1573.4 (-1%) | 1563.4 (-1%) | 1566.8 (+0%) | 1890.7 (+21%) | +62% |
| 2560/300 | 1 | 63.2 | 84.4 (+34%) | 105.8 (+25%) | 109.3 (+3%) | 147.2 (+35%) | 172.9 (+17%) | +174% |
| 2560/300 | 16 | 377.9 | 396.9 (+5%) | 414.1 (+4%) | 412.8 (-0%) | 413.6 (+0%) | 398.3 (-4%) | +5% |
| 2560/300 | 64 | 391.7 | 419.2 (+7%) | 426.2 (+2%) | 422.7 (-1%) | 423.0 (+0%) | 638.9 (+51%) | +63% |

Runs: 20261018T045319Z, 20261019T003224Z, 20261019T004110Z, 20261019T013129Z, 20261019T023002Z, 20261019T123105Z.
<!-- /table -->

At 4 loops the decode step is 10.22 ms on unmodified vLLM with Triton
attention, 7.75 ms on the fused path, and the wavefront commits a token about
every 3.8 to 5.4 ms depending on the text (section 4).

### How the fused path got there (2 loops)

Each column records a benchmark run made when that stage was completed,
using the code available at the time.

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
uses the fused decode path and INT8 LM head, plus wavefront decoding at two or
more loops. All three use NVFP4.

<!-- table:final -->
| Loops | Input/output | Users | vLLM as shipped | Unmodified vLLM, Triton attention | This work | Over vLLM as shipped | Over Triton attention |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 128/1024 | 1 | 188.8 | 234.8 | 318.6 | +69% | +36% |
| 1 | 128/1024 | 16 | 2204.2 | 2523.5 | 2531.8 | +15% | +0% |
| 1 | 128/1024 | 64 | 3453.5 | 4702.5 | 4677.7 | +35% | -1% |
| 1 | 2560/300 | 1 | 154.7 | 201.9 | 268.7 | +74% | +33% |
| 1 | 2560/300 | 16 | 911.5 | 952.8 | 1058.5 | +16% | +11% |
| 1 | 2560/300 | 64 | 1312.1 | 1576.2 | 1707.4 | +30% | +8% |
| 2 | 128/1024 | 1 | 128.1 | 161.7 | 250.8 | +96% | +55% |
| 2 | 128/1024 | 16 | 1488.2 | 1721.1 | 1719.5 | +16% | -0% |
| 2 | 128/1024 | 64 | 2281.0 | 3145.9 | 3145.0 | +38% | -0% |
| 2 | 2560/300 | 1 | 104.3 | 138.0 | 201.8 | +94% | +46% |
| 2 | 2560/300 | 16 | 661.3 | 697.2 | 697.8 | +6% | +0% |
| 2 | 2560/300 | 64 | 659.3 | 779.3 | 784.7 | +19% | +1% |
| 4 | 128/1024 | 1 | 77.9 | 99.6 | 248.4 | +219% | +149% |
| 4 | 128/1024 | 16 | 903.2 | 1049.0 | 1050.9 | +16% | +0% |
| 4 | 128/1024 | 64 | 1164.7 | 1583.8 | 1890.7 | +62% | +19% |
| 4 | 2560/300 | 1 | 63.2 | 84.4 | 172.9 | +174% | +105% |
| 4 | 2560/300 | 16 | 377.9 | 396.9 | 398.3 | +5% | +0% |
| 4 | 2560/300 | 64 | 391.7 | 419.2 | 638.9 | +63% | +52% |

Runs: 20261018T045319Z, 20261018T114127Z, 20261018T180516Z, 20261019T003224Z, 20261019T022147Z, 20261019T024248Z, 20261019T123105Z.
<!-- /table -->

- The single-user rows are this work's target. Batched rows run vLLM's own
  kernels; the fused path's second copy of the layer weights takes 11 to 13%
  of the KV cache, which only matters where 64 users fill the cache (128/1024
  at 4 loops: 64 x 1152 tokens against about 57,000 tokens of cache).
- Against the default backend, batched rows also gain from Triton attention.

### With Dory's shared KV cache (4 loops)

Dory can also keep one KV cache per recurrent layer that every loop overwrites
(`recurrent_kv_cache_mode: "last_loop"`, the WaveFront paper's cross-recurrence
KV sharing), instead of one cache per loop. The fused path and the wavefront
both run on it; the wavefront then approximates sequential decoding (in-flight
tokens see older in-flight tokens at their latest loop, not their last), within
the accuracy gate's noise floor (section 6). "Change" is the last column against
the first.

<!-- table:shared-kv-4 -->
| Input/output | Users | Unmodified, per-loop KV | This work, per-loop KV | Unmodified, shared KV | Fused path, shared KV | This work, shared KV | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 99.6 | 184.5 | 99.9 | 131.5 | 248.4 | +149% |
| 128/1024 | 16 | 1049.0 | 1049.7 | 1051.0 | 1051.7 | 1050.9 | +0% |
| 128/1024 | 64 | 1583.8 | 1566.8 | 1889.5 | 1889.2 | 1890.7 | +19% |
| 2560/300 | 1 | 84.4 | 147.2 | 84.6 | 109.8 | 172.9 | +105% |
| 2560/300 | 16 | 396.9 | 413.6 | 398.2 | 415.3 | 398.3 | +0% |
| 2560/300 | 64 | 419.2 | 423.0 | 637.8 | 659.6 | 638.9 | +52% |

Runs: 20261019T003224Z, 20261019T013129Z, 20261019T111134Z, 20261019T122346Z, 20261019T123105Z.
<!-- /table -->

- The shared cache holds 173,814 tokens on unmodified vLLM and 151,915 with the
  fused path's weight copy, against 65,169 and 56,966 per-loop, so the
  64-user rows that filled the per-loop cache gain 19% (128/1024) and 52%
  (2560/300).
- Drafts match more often on the shared cache: vLLM sampled 96.2% of them in
  these random-token runs (87 to 92% per-loop) and 97.1% on chat prompts with
  temperature 1 and top-p 0.95 (94.5% per-loop), so one user gains 89% over the
  fused path at 128/1024 (248.4 against 131.5 tok/s) instead of 41%.

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

With sampling, every token is drawn by vLLM's sampler from the full-depth
logits, so the wavefront changes which tokens get computed ahead of time, not
the distribution of the output.

## Caveats

- All results come from one machine. The kernels' launch configurations are
  tuned for its RTX 4090.
- The 64-user rows that fill the KV cache (128/1024 at 4 loops) preempt
  requests and vary by about 1% between runs; their time to first token varies
  much more.
- The 128/1024 single-user row with the wavefront measured 201.7 to 201.9
  tok/s when it was a server's first benchmark and 184.5 when it followed the
  2560-token rows; the cause is not established.
- **The 2560/300 rows with 16 and 64 users are bimodal.** Runs of one
  configuration land on one of two levels, for example 650 or 697 tok/s at 16
  users and 2 loops, and 955 or 1058 at 1 loop. The low level has a longer
  time to first token and the same decode speed, and both levels occur with
  and without this work's code, so differences of up to 11% between columns
  in those rows are not effects of the features. The cause is not established:
  it is not the prompts (the benchmark's seed is fixed) and not whether the
  server compiled the model or loaded it from vLLM's compile cache (both
  levels occur either way). Single-user rows and the 128/1024 rows are not
  affected. Details: EXPERIMENTS.md, E23.
