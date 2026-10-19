# Wavefront self-speculative decoding, stand-alone prototype (NVIDIA GeForce RTX 4090, 4 loops, drafts at loop 1, greedy)

8 chat prompts, 256 tokens each, decode time only. A: sequential, single-token kernels. B: sequential, multi-token kernels. W: wavefront.

| Prompt | Prompt tokens | A tok/s | B tok/s | W tok/s | Drafts accepted | W equals B | Shared W/B | Shared A/B |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 32 | 137.7 | 119.7 | 264.2 | 97.3% | True | 256 | 256 |
| 1 | 31 | 137.7 | 119.7 | 251.1 | 95.3% | True | 256 | 167 |
| 2 | 27 | 137.7 | 119.8 | 235.0 | 92.5% | True | 256 | 256 |
| 3 | 29 | 137.6 | 119.3 | 263.0 | 97.3% | True | 256 | 256 |
| 4 | 42 | 137.5 | 119.5 | 248.1 | 94.9% | True | 256 | 210 |
| 5 | 29 | 137.6 | 119.6 | 248.5 | 94.9% | True | 256 | 134 |
| 6 | 30 | 137.6 | 119.6 | 230.2 | 91.8% | True | 256 | 153 |
| 7 | 24 | 137.6 | 119.6 | 237.1 | 92.9% | True | 256 | 53 |

- Overall: A 137.6 tok/s (7.27 ms per token), B 119.6 tok/s, W 246.6 tok/s (4.05 ms per token); W is 1.79x A.
- Drafts accepted: 94.6% of 2040 verified drafts; mean tokens in flight per step 3.70.
