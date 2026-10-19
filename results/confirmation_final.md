# Serving confirmation at the end of the hill-climb (2026-10-18)

Runs, in table order within each group of rows: unmodified vLLM
(20261018T180516Z), fused path after E5 (20261018T181035Z), fused path after
E9 (20261018T185825Z) and fused path after E9 with the INT8 LM head
(20261018T190327Z). All NVFP4, 2 loops, Triton attention, RTX 4090, run one at
a time on an idle GPU. KV cache: 112,666 tokens unmodified, 100,581 fused,
98,548 fused with the INT8 LM head.

Run 20261018T180516Z | NVIDIA GeForce RTX 4090 | vLLM 0.1.dev1+g55076950b | torch 2.14.0+cu130
Run 20261018T181035Z | NVIDIA GeForce RTX 4090 | vLLM 0.1.dev1+g55076950b | torch 2.14.0+cu130
Run 20261018T185825Z | NVIDIA GeForce RTX 4090 | vLLM 0.1.dev1+g55076950b | torch 2.14.0+cu130
Run 20261018T190327Z | NVIDIA GeForce RTX 4090 | vLLM 0.1.dev1+g55076950b | torch 2.14.0+cu130

| Input/Output | Concurrency | Loops | Weights | Attention | Output tok/s | vs BF16 | Per-user tok/s | Median TTFT ms | P99 TTFT ms | Failed |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 2 | nvfp4 | TRITON_ATTN | 161.7 | n/a | 162.1 | 18 | 33 | 0 |
| 128/1024 | 1 | 2 | nvfp4 | TRITON_ATTN + fused decode | 199.2 | n/a | 199.7 | 18 | 21 | 0 |
| 128/1024 | 1 | 2 | nvfp4 | TRITON_ATTN + fused decode | 200.3 | n/a | 200.9 | 18 | 29 | 0 |
| 128/1024 | 1 | 2 | nvfp4 | TRITON_ATTN + fused decode + INT8 LM head | 215.7 | n/a | 216.5 | 17 | 27 | 0 |
| 128/1024 | 16 | 2 | nvfp4 | TRITON_ATTN | 1721.1 | n/a | 109.2 | 152 | 166 | 0 |
| 128/1024 | 16 | 2 | nvfp4 | TRITON_ATTN + fused decode | 1719.3 | n/a | 109.1 | 158 | 162 | 0 |
| 128/1024 | 16 | 2 | nvfp4 | TRITON_ATTN + fused decode | 1721.7 | n/a | 109.1 | 159 | 170 | 0 |
| 128/1024 | 16 | 2 | nvfp4 | TRITON_ATTN + fused decode + INT8 LM head | 1721.5 | n/a | 109.2 | 161 | 167 | 0 |
| 128/1024 | 64 | 2 | nvfp4 | TRITON_ATTN | 3145.9 | n/a | 49.9 | 153 | 591 | 0 |
| 128/1024 | 64 | 2 | nvfp4 | TRITON_ATTN + fused decode | 3142.8 | n/a | 49.9 | 159 | 581 | 0 |
| 128/1024 | 64 | 2 | nvfp4 | TRITON_ATTN + fused decode | 3142.1 | n/a | 49.9 | 177 | 585 | 0 |
| 128/1024 | 64 | 2 | nvfp4 | TRITON_ATTN + fused decode + INT8 LM head | 3144.8 | n/a | 49.9 | 178 | 600 | 0 |
| 2560/300 | 1 | 2 | nvfp4 | TRITON_ATTN | 138.0 | n/a | 152.4 | 236 | 247 | 0 |
| 2560/300 | 1 | 2 | nvfp4 | TRITON_ATTN + fused decode | 169.0 | n/a | 191.7 | 238 | 258 | 0 |
| 2560/300 | 1 | 2 | nvfp4 | TRITON_ATTN + fused decode | 169.2 | n/a | 192.6 | 243 | 264 | 0 |
| 2560/300 | 1 | 2 | nvfp4 | TRITON_ATTN + fused decode + INT8 LM head | 180.8 | n/a | 206.7 | 237 | 253 | 0 |
| 2560/300 | 16 | 2 | nvfp4 | TRITON_ATTN | 697.2 | n/a | 46.0 | 727 | 1851 | 0 |
| 2560/300 | 16 | 2 | nvfp4 | TRITON_ATTN + fused decode | 697.3 | n/a | 46.1 | 727 | 1851 | 0 |
| 2560/300 | 16 | 2 | nvfp4 | TRITON_ATTN + fused decode | 697.0 | n/a | 48.3 | 912 | 1853 | 0 |
| 2560/300 | 16 | 2 | nvfp4 | TRITON_ATTN + fused decode + INT8 LM head | 696.5 | n/a | 46.0 | 727 | 2004 | 0 |
| 2560/300 | 64 | 2 | nvfp4 | TRITON_ATTN | 779.3 | n/a | 15.8 | 3604 | 21848 | 0 |
| 2560/300 | 64 | 2 | nvfp4 | TRITON_ATTN + fused decode | 777.6 | n/a | 17.5 | 5066 | 21416 | 0 |
| 2560/300 | 64 | 2 | nvfp4 | TRITON_ATTN + fused decode | 776.8 | n/a | 17.4 | 5069 | 21479 | 0 |
| 2560/300 | 64 | 2 | nvfp4 | TRITON_ATTN + fused decode + INT8 LM head | 784.9 | n/a | 18.1 | 5071 | 20980 | 0 |
