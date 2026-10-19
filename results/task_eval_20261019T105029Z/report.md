# Task accuracy (4 loops, thinking disabled, one request at a time)

GSM8K: 250 problems, greedy, up to 768 tokens. MMLU: 1000 questions over 42 subjects, best answer letter by log-probability.

| Configuration | Task | Accuracy | Right only here | Right only in nvfp4-triton | McNemar p | Seconds |
|---|---|---:|---:|---:|---:|---:|
| nvfp4-triton | gsm8k | 90.8% | 0 | 0 |  | 557 |
| nvfp4-triton | mmlu | 62.3% | 0 | 0 |  | 32 |
| nvfp4-triton-last-loop | gsm8k | 91.2% | 5 | 4 | 1.00 | 567 |
| nvfp4-triton-last-loop | mmlu | 62.7% | 6 | 2 | 0.29 | 32 |

## MMLU answer letters compared with nvfp4-triton

| Configuration | Same letter | Mean abs logprob change of the reference's letter | Largest change |
|---|---:|---:|---:|
| nvfp4-triton-last-loop | 98.9% | 0.0155 | 0.3001 |
