# Multi-token NVFP4 GEMVs against Marlin (NVIDIA GeForce RTX 4090, 2 loops, Triton 3.8.0)

Microseconds per projection call (decode order, CUDA graph); best configuration (rows per program, words per iteration, warps) in parentheses. Minimum: the weight bytes at 1008 GB/s. Ratio: tensor core over the single-token kernel.

| Projection | Tokens | Minimum | Single-token kernel, 1 token | Marlin | Tensor core | Ratio | Max relative error (tensor core) |
|---|---:|---:|---:|---:|---|---:|---:|
| qkv_gate (2560 x 10240) | 4 | 14.6 | 17.5 (4, 64, 4) | n/a | 20.2 (64, 32, 4, split 2, 2 stages) | 1.15 | 1.67e-03 |
| attn_out (4096 x 2560) | 4 | 5.9 | 8.0 (4, 64, 8) | 10.0 | 10.2 (64, 32, 4, split 4, 2 stages) | 1.27 | 1.69e-03 |
| gate_up (2560 x 14336) | 4 | 20.5 | 24.2 (2, 64, 4) | 24.9 | 26.5 (64, 32, 4, 3 stages) | 1.10 | 1.68e-03 |
| down (7168 x 2560) | 4 | 10.2 | 12.8 (2, 64, 4) | 14.5 | 15.0 (128, 32, 8, split 4, 3 stages) | 1.18 | 1.70e-03 |
