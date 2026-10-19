# Wavefront self-speculative decoding, stand-alone prototype (NVIDIA GeForce RTX 4090, 4 loops, drafts at loop 1, temperature 1.0, top-p 0.95)

8 chat prompts, 256 tokens each, decode time only. A: sequential, single-token kernels. B: sequential, multi-token kernels. W: wavefront.

| Prompt | Prompt tokens | A tok/s | B tok/s | W tok/s | Drafts accepted | W equals B | Shared W/B | Shared A/B |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 32 | 135.7 | n/a | 230.7 | 97.6% | n/a | n/a | n/a |
| 1 | 31 | 135.7 | n/a | 226.1 | 96.9% | n/a | n/a | n/a |
| 2 | 27 | 135.8 | n/a | 194.7 | 90.2% | n/a | n/a | n/a |
| 3 | 29 | 135.7 | n/a | 223.9 | 96.5% | n/a | n/a | n/a |
| 4 | 42 | 135.5 | n/a | 218.9 | 95.7% | n/a | n/a | n/a |
| 5 | 29 | 135.6 | n/a | 221.7 | 96.1% | n/a | n/a | n/a |
| 6 | 30 | 135.6 | n/a | 192.7 | 89.8% | n/a | n/a | n/a |
| 7 | 24 | 135.7 | n/a | 208.1 | 93.3% | n/a | n/a | n/a |

- Overall: A 135.7 tok/s (7.37 ms per token), W 213.7 tok/s (4.68 ms per token); W is 1.58x A.
- Drafts accepted: 94.5% of 2040 verified drafts; mean tokens in flight per step 3.70.
