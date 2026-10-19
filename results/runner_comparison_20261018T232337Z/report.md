# vLLM V2 versus V1 model runner, fused decode path (NVIDIA GeForce RTX 4090)

NVFP4, Triton attention, fused decode with its default LM head, run one at a time on an idle GPU. Changes are relative to V2.

## Single-user decode step (bench/profile_decode.py, 640-token prompt)

| Loops | Runner | Step ms | tok/s | Speed change | Forward busy ms | Idle us per step | Profile |
|---:|---|---:|---:|---:|---:|---:|---|
| 2 | V2 | 4.71 | 212.4 | +0.0% | 4.22 | 40 | profile_20261018T232349Z_nvfp4_loops2_prompt640_triton_attn |
| 2 | V1 | 4.78 | 209.2 | -1.5% | 4.22 | 64 | profile_20261018T232420Z_nvfp4_loops2_prompt640_triton_attn |
| 2 | V1, no async scheduling | 5.66 | 176.5 | -16.9% | 4.23 | 941 | profile_20261018T232454Z_nvfp4_loops2_prompt640_triton_attn |
| 4 | V2 | 7.75 | 129.0 | +0.0% | 7.25 | 52 | profile_20261018T232603Z_nvfp4_loops4_prompt640_triton_attn |
| 4 | V1 | 7.77 | 128.6 | -0.3% | 7.20 | 76 | profile_20261018T232637Z_nvfp4_loops4_prompt640_triton_attn |
| 4 | V1, no async scheduling | 8.84 | 113.2 | -12.3% | 7.28 | 1067 | profile_20261018T232801Z_nvfp4_loops4_prompt640_triton_attn |

## Serving at 2 loops (bench/run_baseline.py), output tok/s

Runs: V2 20261018T232916Z, V1 20261018T233431Z, V1, no async scheduling 20261018T234033Z.

| Input/output | Users | V2 | V1 | V1, no async scheduling | V1 change | V1, no async scheduling change |
|---|---:|---:|---:|---:|---:|---:|
| 128/1024 | 1 | 215.8 | 212.7 | 189.2 | -1.5% | -12.3% |
| 128/1024 | 16 | 1718.1 | 1706.2 | 1564.7 | -0.7% | -8.9% |
| 128/1024 | 64 | 3135.4 | 3114.1 | 2889.3 | -0.7% | -7.8% |
| 2560/300 | 1 | 180.8 | 178.5 | 161.2 | -1.3% | -10.8% |
| 2560/300 | 16 | 694.7 | 694.0 | 664.2 | -0.1% | -4.4% |
| 2560/300 | 64 | 782.1 | 778.9 | 753.5 | -0.4% | -3.7% |
