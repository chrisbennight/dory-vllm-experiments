# Wavefront self-speculative decoding, stand-alone prototype (NVIDIA GeForce RTX 4090, 4 loops, drafts at loop 1)

8 chat prompts, 256 greedy tokens each, decode time only. A: sequential, single-token kernels. B: sequential, multi-token kernels. W: wavefront.

| Prompt | Prompt tokens | A tok/s | B tok/s | W tok/s | Drafts accepted | W equals B | Shared W/B | Shared A/B |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 32 | 137.5 | 116.2 | 258.0 | 97.3% | True | 256 | 256 |
| 1 | 31 | 137.6 | 116.0 | 249.8 | 96.1% | True | 256 | 170 |
| 2 | 27 | 137.3 | 115.9 | 227.0 | 92.2% | True | 256 | 256 |
| 3 | 29 | 137.6 | 116.1 | 257.5 | 97.3% | True | 256 | 256 |
| 4 | 42 | 137.5 | 116.0 | 239.9 | 94.5% | True | 256 | 251 |
| 5 | 29 | 137.6 | 116.1 | 249.9 | 96.1% | True | 256 | 256 |
| 6 | 30 | 137.6 | 116.1 | 222.8 | 91.4% | True | 256 | 153 |
| 7 | 24 | 137.6 | 115.9 | 242.8 | 94.9% | True | 256 | 61 |

- Overall: A 137.5 tok/s (7.27 ms per token), B 116.0 tok/s, W 242.8 tok/s (4.12 ms per token); W is 1.77x A.
- Drafts accepted: 95.0% of 2040 verified drafts; mean tokens in flight per step 3.72.
