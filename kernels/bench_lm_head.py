"""Compares the 8-bit LM heads with vLLM's BF16 LM head for single-token logits.

Both compute logit_scale / initializer_range * (W h) for unit-norm hidden
states h (Dory's final hidden state is normalized). Errors are measured
against FP32 math on the BF16 checkpoint weights; vLLM's path rounds the
logits to BF16 before scaling, as DoryForCausalLM.compute_logits does.
Timing replays each implementation in a CUDA graph.

Usage: .venv/bin/python kernels/bench_lm_head.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import torch
from safetensors import safe_open

from bench_gemv import CHECKPOINT, ROOT, require_idle_gpu, time_graph
from dory_lm_head import QuantizedLmHead

CONFIGS = [dict(block_v=v, block_k=k, num_warps=w) for v, k, w in
           ((8, 512, 4), (16, 512, 4), (4, 512, 4), (8, 256, 4), (16, 256, 8), (32, 256, 8))]


def load(name: str, device: torch.device) -> torch.Tensor:
    for path in sorted(CHECKPOINT.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            if name in f.keys():
                return f.get_tensor(name).to(device)
    raise KeyError(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    torch.manual_seed(0)
    config = json.loads((CHECKPOINT / "config.json").read_text())
    weight = load("lm_head.weight", device)
    logit_scale = load("lm_head.logit_scale", device)
    factor = logit_scale.float() / config["initializer_range"]
    heads = {storage: QuantizedLmHead(weight, logit_scale, config["initializer_range"], storage)
             for storage in ("int8", "fp8")}

    metrics = {"vLLM BF16": [], "INT8": [], "FP8": []}
    for _ in range(args.samples):
        h = torch.randn(1, weight.shape[1], device=device)
        h = (h / h.norm()).to(torch.bfloat16)
        reference = (h.float() @ weight.float().T) * factor
        candidates = {"vLLM BF16": (h @ weight.T).float() * factor, "INT8": heads["int8"](h),
                      "FP8": heads["fp8"](h)}
        ref_logp = reference.log_softmax(dim=-1)
        for name, logits in candidates.items():
            logp = logits.log_softmax(dim=-1)
            metrics[name].append({
                "max_abs": (logits - reference).abs().max().item(),
                "top1": float(logits.argmax() == reference.argmax()),
                "kl": (ref_logp.exp() * (ref_logp - logp)).sum().item(),
                "top_logprob": (logp[0, reference.argmax()] - ref_logp[0, reference.argmax()])
                .abs().item(),
            })

    h = torch.randn(1, weight.shape[1], device=device)
    h = (h / h.norm()).to(torch.bfloat16)
    bf16_us = time_graph([lambda: (h @ weight.T).float() * factor], args.replays) * 1000
    timings = {}
    for storage, head in heads.items():
        for cfg in CONFIGS:
            head.block_v, head.block_k, head.num_warps = (cfg["block_v"], cfg["block_k"],
                                                          cfg["num_warps"])
            timings.setdefault(storage, []).append(
                (time_graph([lambda head=head: head(h)], args.replays) * 1000, cfg))

    vocab, hidden = weight.shape
    lines = [f"# 8-bit LM heads against vLLM's BF16 LM head ({torch.cuda.get_device_name()}, "
             f"{vocab} x {hidden})", "",
             f"## Accuracy against FP32 math ({args.samples} random unit-norm hidden states)", "",
             "| Head | Mean max abs logit error | Worst max abs logit error | Top-1 agreement "
             "| Mean KL divergence | Mean abs top-token logprob error |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, rows in metrics.items():
        mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
        lines.append(f"| {name} | {mean('max_abs'):.4f} | {max(r['max_abs'] for r in rows):.4f} "
                     f"| {mean('top1'):.0%} | {mean('kl'):.2e} | {mean('top_logprob'):.4f} |")
    minimum = lambda bytes_: bytes_ / 1008e9 * 1e6  # noqa: E731
    lines += ["", "## Time per token (CUDA graph)", "",
              "| Head | us | Minimum us |", "|---|---:|---:|",
              f"| vLLM BF16 (matmul and scale) | {bf16_us:.1f} | {minimum(vocab * hidden * 2):.1f} |"]
    for storage, results in timings.items():
        us, cfg = min(results, key=lambda r: r[0])
        lines.append(f"| {storage.upper()} {cfg} | {us:.1f} "
                     f"| {minimum(vocab * hidden + vocab * 4):.1f} |")
    for storage, results in timings.items():
        lines += ["", f"All {storage.upper()} configurations: " + ", ".join(
            f"{c['block_v']}x{c['block_k']}x{c['num_warps']} {us:.1f} us" for us, c in results)]
    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"lm_head_bench_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
