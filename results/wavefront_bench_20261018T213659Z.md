# Wavefront self-speculative decoding, stand-alone prototype (NVIDIA GeForce RTX 4090, 4 loops, drafts at loop 1, temperature 1.0, top-p 0.95)

8 chat prompts, 256 tokens each, decode time only. A: sequential, single-token kernels. B: sequential, multi-token kernels. W: wavefront.

| Prompt | Prompt tokens | A tok/s | B tok/s | W tok/s | Drafts accepted | W equals B | Shared W/B | Shared A/B |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 32 | 135.5 | n/a | 257.9 | 99.6% | n/a | n/a | n/a |
| 1 | 31 | 135.6 | n/a | 229.1 | 94.9% | n/a | n/a | n/a |
| 2 | 27 | 135.6 | n/a | 225.0 | 94.1% | n/a | n/a | n/a |
| 3 | 29 | 135.6 | n/a | 235.6 | 96.1% | n/a | n/a | n/a |
| 4 | 42 | 135.5 | n/a | 237.7 | 96.5% | n/a | n/a | n/a |
| 5 | 29 | 135.6 | n/a | 227.0 | 94.5% | n/a | n/a | n/a |
| 6 | 30 | 135.6 | n/a | 211.2 | 91.4% | n/a | n/a | n/a |
| 7 | 24 | 135.6 | n/a | 222.9 | 93.7% | n/a | n/a | n/a |

- Overall: A 135.6 tok/s (7.38 ms per token), W 230.1 tok/s (4.35 ms per token); W is 1.70x A.
- Drafts accepted: 95.1% of 2040 verified drafts; mean tokens in flight per step 3.73.
