"""Benchmarks multi-token NVFP4 GEMVs against vLLM's Marlin kernel for 1-8 tokens.

For each projection kind and token count M, every layer's projection of that
kind runs in decode order (2 loops) inside a CUDA graph, so the weights stream
from DRAM as in a decode step over M tokens. Kernels:

- single-token: today's fused GEMV (w4a16_gemv), run once; the cost that a
  multi-token kernel should stay close to;
- tensor core: w4a16_gemm_tc, which decodes each weight tile once and
  multiplies it with all M inputs on tensor cores, optionally splitting each
  row's K over several programs (the partial sums are left to the consumer and
  their addition is not timed);
- CUDA core (with --cuda-core): w4a16_gemv_tokens;
- Marlin: vLLM's kernel. It cannot run the merged q/k/v + output gate weight
  (its shards have different global scales), so that kind has no Marlin time.

Triton kernels take their M inputs in nibble-major FP32, as a fused pipeline
would produce them. Accuracy compares every token's output with dequantized
FP32 weights.

Usage: .venv/bin/python kernels/bench_gemv_tokens.py
"""

import argparse
import sys
from datetime import datetime, timezone

import torch
import triton
from triton.runtime.errors import OutOfResources

from bench_gemv import (BYTES_PER_PARAM, PEAK_BYTES_PER_S, ROOT, MarlinProjection,
                        decode_order, dequantize, kinds_for_layer, load_projections,
                        relative_error, require_idle_gpu, time_graph)
from w4a16_gemv import to_nibble_major, w4a16_gemm_tc, w4a16_gemv, w4a16_gemv_tokens

TOKENS = (1, 2, 3, 4, 8)
# A wider search over the tensor-core GEMV's launch parameters (--wide).
WIDE_TC_CONFIGS = [dict(block_n=n, block_w=w, num_warps=warps, split_k=s, num_stages=st)
                   for n in (32, 64, 128) for w in (32, 64) for warps in (4, 8)
                   for s in (1, 2, 4) for st in (2, 3, 4)]
KINDS = ("qkv_gate", "qkv", "gate", "attn_out", "gate_up", "down")
CONFIGS = [dict(block_n=n, block_w=w, num_warps=warps) for n, w, warps in
           ((4, 64, 4), (4, 32, 4), (2, 64, 4), (8, 32, 4), (4, 64, 8), (8, 64, 8), (4, 128, 4))]
TC_CONFIGS = [dict(block_n=n, block_w=w, num_warps=warps, split_k=s) for n, w, warps, s in
              ((16, 64, 4, 1), (32, 32, 4, 1), (64, 32, 4, 1), (32, 64, 4, 1),
               (32, 32, 4, 2), (32, 32, 4, 4), (32, 32, 4, 5), (32, 32, 4, 8),
               (64, 32, 4, 2), (64, 32, 4, 4), (16, 32, 4, 4), (16, 64, 4, 2),
               (32, 64, 4, 4), (64, 64, 8, 2))]


def best_time(keys, projections, launch, configs, k, replays):
    """(microseconds per call, configuration) of the fastest configuration."""
    timings = []
    for cfg in configs:
        if k % (8 * cfg["block_w"] * cfg.get("split_k", 1)):
            continue
        calls = [lambda p=projections[key], cfg=cfg: launch(p, cfg) for key in keys]
        try:
            timings.append((time_graph(calls, replays) / len(keys) * 1000, cfg))
        except OutOfResources:
            continue  # the configuration needs more shared memory than the GPU has
    return min(timings, key=lambda t: t[0])


def config_text(cfg: dict) -> str:
    split = f", split {cfg['split_k']}" if cfg.get("split_k", 1) > 1 else ""
    stages = f", {cfg['num_stages']} stages" if "num_stages" in cfg else ""
    return f"{cfg['block_n']}, {cfg['block_w']}, {cfg['num_warps']}{split}{stages}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--cuda-core", action="store_true",
                        help="also time the CUDA-core multi-token kernel")
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    parser.add_argument("--wide", action="store_true",
                        help="search WIDE_TC_CONFIGS instead of TC_CONFIGS")
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    torch.manual_seed(0)

    projections, config = load_projections(device)
    order = decode_order(config, args.loops)
    lines = [f"# Multi-token NVFP4 GEMVs against Marlin ({torch.cuda.get_device_name()}, "
             f"{args.loops} loops, Triton {triton.__version__})", "",
             "Microseconds per projection call (decode order, CUDA graph); best configuration "
             "(rows per program, words per iteration, warps) in parentheses. Minimum: the "
             "weight bytes at 1008 GB/s. Ratio: tensor core over the single-token kernel.", "",
             "| Projection | Tokens | Minimum | Single-token kernel, 1 token | Marlin "
             "| Tensor core | Ratio |" + (" CUDA core |" if args.cuda_core else "")
             + " Max relative error (tensor core) |",
             "|---|---:|---:|---:|---:|---|---:|" + ("---|" if args.cuda_core else "") + "---:|"]
    for kind in args.kinds:
        keys = [(kind, layer) for layer in order
                if kind in kinds_for_layer(config, layer, kind == "qkv_gate")]
        k, n = projections[keys[0]].shape
        ideal_us = k * n * BYTES_PER_PARAM / PEAK_BYTES_PER_S * 1e6
        marlin = ({key: MarlinProjection(projections[key]) for key in dict.fromkeys(keys)}
                  if kind != "qkv_gate" else None)
        reference_weight = dequantize(projections[keys[0]]).float()
        x1 = to_nibble_major(torch.randn(k, device=device))
        out1 = torch.empty(n, dtype=torch.bfloat16, device=device)
        single_us, single_cfg = best_time(
            keys, projections, lambda p, cfg: w4a16_gemv(x1, p.weight, p.scale, p.global_scale,
                                                         out1, **cfg),
            CONFIGS, k, args.replays)
        for m in args.tokens:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            x_nm = x.float().view(m, -1, 8).transpose(1, 2).contiguous()
            out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
            partials = {s: torch.empty(s, m, n, device=device)
                        for s in {c["split_k"] for c in TC_CONFIGS if c["split_k"] > 1}}

            def tc(p, cfg, x_nm=x_nm, out=out, partials=partials):
                target = out if cfg["split_k"] == 1 else partials[cfg["split_k"]]
                return w4a16_gemm_tc(x_nm, p.weight, p.scale, p.global_scale, target, **cfg)
            tc_us, tc_cfg = best_time(keys, projections, tc,
                                      WIDE_TC_CONFIGS if args.wide else TC_CONFIGS, k,
                                      args.replays)
            cuda_core = ""
            if args.cuda_core:
                cc_us, cc_cfg = best_time(
                    keys, projections, lambda p, cfg: w4a16_gemv_tokens(
                        x_nm, p.weight, p.scale, p.global_scale, out, **cfg),
                    CONFIGS, k, args.replays)
                cuda_core = f" {cc_us:.1f} ({config_text(cc_cfg)}) |"
            p = projections[keys[0]]
            result = tc(p, tc_cfg)
            if tc_cfg["split_k"] > 1:
                out = result.sum(dim=0).to(torch.bfloat16)
            reference = x.float() @ reference_weight.T
            error = max(relative_error(out[i], reference[i]) for i in range(m))
            if marlin is not None:
                calls = [lambda mp=marlin[key]: mp(x) for key in keys]
                marlin_cell = f"{time_graph(calls, args.replays) / len(keys) * 1000:.1f}"
            else:
                marlin_cell = "n/a"
            lines.append(f"| {kind} ({k} x {n}) | {m} | {ideal_us:.1f} "
                         f"| {single_us:.1f} ({config_text(single_cfg)}) | {marlin_cell} "
                         f"| {tc_us:.1f} ({config_text(tc_cfg)}) | {tc_us / single_us:.2f} |"
                         f"{cuda_core} {error:.2e} |")
            print(lines[-1], flush=True)
        del marlin
        torch.cuda.empty_cache()

    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"gemv_tokens_bench_{stamp}.md"
    out_path.write_text(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
