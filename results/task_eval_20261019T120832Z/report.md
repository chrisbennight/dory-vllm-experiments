# Task accuracy (4 loops, thinking disabled, one request at a time)

GSM8K: 250 problems, greedy, up to 768 tokens. MMLU: 1000 questions over 42 subjects, best answer letter by log-probability.

| Configuration | Task | Accuracy | Right only here | Right only in nvfp4-triton-last-loop | McNemar p | Seconds |
|---|---|---:|---:|---:|---:|---:|
| nvfp4-triton-last-loop | gsm8k | 91.2% | 0 | 0 |  | 567 |
| nvfp4-triton-last-loop | mmlu | 62.7% | 0 | 0 |  | 32 |
| nvfp4-triton-fused-wavefront-last-loop | gsm8k | 92.0% | 5 | 3 | 0.73 | 215 |
| nvfp4-triton-fused-wavefront-last-loop | mmlu | 62.7% | 2 | 2 | 1.00 | 31 |

## MMLU answer letters compared with nvfp4-triton-last-loop

| Configuration | Same letter | Mean abs logprob change of the reference's letter | Largest change |
|---|---:|---:|---:|
| nvfp4-triton-fused-wavefront-last-loop | 99.6% | 0.0052 | 0.0750 |
