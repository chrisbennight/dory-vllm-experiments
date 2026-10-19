# Prompt-lookup speculation at one user (2 loops, greedy, up to 512 tokens)

edit: 8 functions from this repository to copy without comments (thinking off); chat: 16 chat prompts (thinking on).

| Configuration | Workload | Output tokens | Seconds | Tokens/s | Speed vs nvfp4-triton-fused-int8-head | Draft tokens | Accepted | Acceptance | Outputs equal to nvfp4-triton-fused-int8-head |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-triton-fused-int8-head | edit | 3303 | 15.5 | 212.5 | 1.00x | 0 | 0 | n/a | 8/8 |
| nvfp4-triton-fused-int8-head | chat | 7654 | 34.5 | 222.0 | 1.00x | 0 | 0 | n/a | 16/16 |
| nvfp4-triton-fused-ngram | edit | 3303 | 5.0 | 662.0 | 3.12x | 2924 | 2573 | 88% | 8/8 |
| nvfp4-triton-fused-ngram | chat | 7616 | 36.3 | 210.0 | 0.95x | 23536 | 1727 | 7% | 1/16 |
