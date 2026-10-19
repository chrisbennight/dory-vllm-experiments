# Wavefront self-speculative decoding, stand-alone prototype (NVIDIA GeForce RTX 4090, 4 loops, drafts at loop 1)

3 chat prompts after about 1000 tokens of notes, 256 greedy tokens each, decode time only. A: sequential, single-token kernels. B: sequential, multi-token kernels. W: wavefront.

| Prompt | Prompt tokens | A tok/s | B tok/s | W tok/s | Drafts accepted | W equals B | Shared W/B | Shared A/B |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 1047 | 130.1 | 110.2 | 220.4 | 95.3% | True | 256 | 21 |
| 1 | 1046 | 130.1 | 110.2 | 217.4 | 94.9% | True | 256 | 256 |
| 2 | 1041 | 130.1 | 110.2 | 197.7 | 90.6% | True | 256 | 256 |

- Overall: A 130.1 tok/s (7.68 ms per token), B 110.2 tok/s, W 211.3 tok/s (4.73 ms per token); W is 1.62x A.
- Drafts accepted: 93.6% of 765 verified drafts; mean tokens in flight per step 3.66.
