# Single-user decode profile findings (RTX 4090, vLLM, NVFP4, 2 loops)

Source data: `profile_*/report.md` in this directory, produced by
`bench/profile_decode.py`, and the serving runs listed in
`summary_nvfp4_attention_backends.md`.

## Where one decode step goes (640-token prompt)

| Part of the step | FlashAttention backend, ms | Triton attention backend, ms | Achievable estimate, ms |
|---|---:|---:|---:|
| 4-bit weight GEMMs (Marlin) | 4.07 | 4.09 | about 3.0 |
| Attention | 2.29 | 0.45 | about 0.25 |
| Normalization and gating kernels (torch.compile) | 0.80 | 0.82 | about 0.2 |
| Output projection (BF16, 131k vocabulary) | 0.70 | 0.70 | 0.70 |
| Sampling, bookkeeping, idle GPU | 0.18 | 0.27 | about 0.2 |
| Total step | 8.05 (124 tok/s) | 6.32 (158 tok/s) | about 4.4 (about 225 tok/s) |

The achievable column is an estimate, not a measurement: weight GEMMs at
about 95% of the 1008 GB/s peak, attention at memory bandwidth plus small
fixed costs, and normalization merged into neighbouring kernels.

## Findings

- The GPU is busy for the whole step. CUDA graphs leave about 0.04 ms of gaps,
  so launch overhead is not a cost.
- Each Marlin GEMM call carries about 4 us of fixed cost whatever its size.
  The small projections (q/k/v 68%, output gate 61%, attention output 58% of
  peak bandwidth) suffer most; the MLP up projection reaches 81%. BF16 GEMMs
  through cuBLAS reach 89%.
- The default FlashAttention backend runs single-token attention far below
  memory bandwidth, and its cost grows with context: 1.0 ms per step at about
  160 tokens of context, 2.3 ms at about 670.
- The Triton attention backend cuts attention to 0.45 ms per step at about 670
  tokens of context. In serving it raises output throughput by 24-38% in most
  configurations, but prompt processing is slower, so long prompts with 16
  users gain nothing (-2% to +5%) and time to first token rises by 10-35%.
- Greedy outputs with Triton attention stay identical to FlashAttention for
  30-256 tokens per prompt, longer than any quantized checkpoint stays
  identical to BF16; the differences are floating-point ordering effects.
- Normalization costs are the same for BF16 and NVFP4 (0.8 ms per step from
  337 small kernels), confirming they do not depend on the weight format.

## Remaining opportunities, largest first

1. A single-token 4-bit GEMV kernel with lower fixed cost, and merging the
   q/k/v and output-gate projections into one GEMM: about 1.1 ms per step.
2. Merging normalization, gating and residual updates into the GEMM kernels:
   about 0.6 ms per step.
3. A decode attention kernel that groups the four query heads sharing each
   KV head: about 0.2 ms per step at this context length, more at longer ones.
4. The BF16 output projection is already at 95% of bandwidth; only quantizing
   it would help (about 0.35 ms at 8 bits), at some accuracy risk.
