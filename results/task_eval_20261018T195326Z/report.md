# Task accuracy (2 loops, thinking disabled, one request at a time)

GSM8K: 250 problems, greedy, up to 768 tokens. MMLU: 1000 questions over 42 subjects, best answer letter by log-probability.

| Configuration | Task | Accuracy | Right only here | Right only in nvfp4-triton-fused | McNemar p | Seconds |
|---|---|---:|---:|---:|---:|---:|
| nvfp4-triton-fused | gsm8k | 90.4% | 0 | 0 |  | 287 |
| nvfp4-triton-fused | mmlu | 60.7% | 0 | 0 |  | 20 |
| nvfp4-triton-fused-int8-head | gsm8k | 89.6% | 2 | 4 | 0.69 | 268 |
| nvfp4-triton-fused-int8-head | mmlu | 60.8% | 2 | 1 | 1.00 | 20 |

## MMLU answer letters compared with nvfp4-triton-fused

| Configuration | Same letter | Mean abs logprob change of the reference's letter | Largest change |
|---|---:|---:|---:|
| nvfp4-triton-fused-int8-head | 99.5% | 0.0050 | 0.0692 |
