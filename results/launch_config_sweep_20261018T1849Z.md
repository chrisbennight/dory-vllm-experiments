# Launch configuration sweep inside vLLM (2026-10-18, E10)

Fused decode path after E9, NVFP4, 2 loops, 640-token prompt, Triton attention. Each line changes one launch configuration in kernels/dory_fused_model.py (rows per program, weight words per iteration, warps); the others stay at their defaults (QKVG 4,64,4; GATE_UP 4,64,4; DOWN_FOLDED 4,64,4). Produced by a scratch script (build/, not kept) that edits the configuration, runs bench/profile_decode.py and bench/gap_analysis.py, and restores the file.

```
QKVG=4,64,4 Measured: 5.06 ms profile_20261018T184909Z_nvfp4_loops2_prompt640_triton_attn
QKVG=2,64,4 Measured: 5.13 ms profile_20261018T184927Z_nvfp4_loops2_prompt640_triton_attn
QKVG=4,32,4 Measured: 5.05 ms profile_20261018T184946Z_nvfp4_loops2_prompt640_triton_attn
QKVG=8,64,8 Measured: 5.06 ms profile_20261018T185004Z_nvfp4_loops2_prompt640_triton_attn
4,128,4 profile failed
QKVG=8,32,4 Measured: 5.07 ms profile_20261018T185035Z_nvfp4_loops2_prompt640_triton_attn
GATE_UP=2,64,4 Measured: 5.09 ms profile_20261018T185053Z_nvfp4_loops2_prompt640_triton_attn
GATE_UP=2,32,2 Measured: 5.05 ms profile_20261018T185112Z_nvfp4_loops2_prompt640_triton_attn
GATE_UP=4,128,4 Measured: 5.10 ms profile_20261018T185130Z_nvfp4_loops2_prompt640_triton_attn
GATE_UP=8,64,8 Measured: 5.11 ms profile_20261018T185148Z_nvfp4_loops2_prompt640_triton_attn
GATE_UP=4,32,4 Measured: 5.09 ms profile_20261018T185207Z_nvfp4_loops2_prompt640_triton_attn
DOWN_FOLDED=4,128,4 Measured: 5.14 ms profile_20261018T185225Z_nvfp4_loops2_prompt640_triton_attn
DOWN_FOLDED=2,64,4 Measured: 5.33 ms profile_20261018T185244Z_nvfp4_loops2_prompt640_triton_attn
DOWN_FOLDED=4,32,4 Measured: 5.38 ms profile_20261018T185303Z_nvfp4_loops2_prompt640_triton_attn
DOWN_FOLDED=8,64,4 Measured: 5.08 ms profile_20261018T185321Z_nvfp4_loops2_prompt640_triton_attn
```

QKVG=4,128,4 cannot run: K=2560 is not a multiple of 8 x 128 words.
