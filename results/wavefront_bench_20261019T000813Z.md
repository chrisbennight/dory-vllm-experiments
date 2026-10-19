# Wavefront self-speculative decoding, stand-alone prototype (NVIDIA GeForce RTX 4090, 4 loops, drafts at loop 1, greedy)

4 chat prompts, 128 tokens each, decode time only. A: sequential, single-token kernels. B: sequential, multi-token kernels. W: wavefront.

| Prompt | Prompt tokens | A tok/s | B tok/s | W tok/s | Drafts accepted | W equals B | Shared W/B | Shared A/B |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 32 | 138.0 | 120.3 | 278.0 | 99.2% | True | 128 | 128 |
| 1 | 31 | 138.0 | 120.1 | 255.2 | 96.1% | True | 128 | 128 |
| 2 | 27 | 138.3 | 120.1 | 250.4 | 95.3% | True | 128 | 128 |
| 3 | 29 | 138.0 | 120.2 | 276.1 | 99.2% | True | 128 | 128 |

- Overall: A 138.1 tok/s (7.24 ms per token), B 120.2 tok/s, W 264.4 tok/s (3.78 ms per token); W is 1.91x A.
- Drafts accepted: 97.4% of 508 verified drafts; mean tokens in flight per step 3.82.
