"""Compares measured decode-step time per component with its memory-bandwidth minimum.

Reads a profile directory written by bench/profile_decode.py and, for each part
of a single-user decode step, sets the measured GPU time against the time the
same bytes would take at the RTX 4090's peak memory bandwidth:

- weight GEMMs: the quantized weights of each projection,
- output projection: the BF16 vocabulary matrix,
- attention: the KV cache entries each attention layer application must read
  (sliding-window layers read at most the window, full layers the whole context),
- normalization, gating, sampling and bookkeeping: only small activation
  tensors, so their minimum is effectively zero when fused into neighbours.

Usage: python bench/gap_analysis.py results/profile_<...> --prompt-len 640
"""

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from profile_decode import BYTES_PER_PARAM, MODELS, categorize, decode_steps

PEAK_BYTES_PER_S = 1008e9
KV_BYTES_PER_ELEMENT = 2  # BF16 KV cache
PROJECTIONS = ("q/k/v", "output gate", "attention output", "MLP gate+up", "MLP down")
# Fused decode path (kernels/ in this repository): one kernel per projection, by name.
FUSED_PROJECTIONS = {"_qkvg_rope_kernel": "q/k/v + output gate",
                     "_o_gated_normalized_kernel": "attention output",
                     "_gate_up_swiglu_kernel": "MLP gate+up",
                     "_down_normalized_kernel": "MLP down"}
# Bytes touched by the normalization, gating and sampling kernels are a few
# activation vectors per sublayer plus one pass over the logits: about 0.01 ms.
SMALL_TENSOR_FLOOR_MS = 0.01


def model_shapes(config: dict) -> dict[str, tuple[int, int]]:
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    kv_heads, head_dim = config["num_key_value_heads"], config["head_dim"]
    intermediate = config["intermediate_size"]
    return {
        "q/k/v": (hidden, (heads + 2 * kv_heads) * head_dim),
        "q/k/v + output gate": (hidden, (2 * heads + 2 * kv_heads) * head_dim),
        "output gate": (hidden, heads * head_dim),
        "attention output": (heads * head_dim, hidden),
        "MLP gate+up": (hidden, 2 * intermediate),
        "MLP down": (intermediate, hidden),
    }


def attention_applications(config: dict, loops: int) -> list[int | None]:
    """Window size (None for full attention) of every attention layer application."""
    first_recurrent = config["n_input_layers"]
    after_recurrent = first_recurrent + config["n_recurrent_layers"]
    windows = []
    for index, layer_type in enumerate(config["layer_types"]):
        if layer_type == "mlp":
            continue
        repeats = loops if first_recurrent <= index < after_recurrent else 1
        window = config["sliding_window"] if layer_type == "sliding_attention" else None
        windows += [window] * repeats
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--prompt-len", type=int,
                        help="prompt length of the profiled run; read from the directory "
                             "name when it contains _prompt<N>")
    args = parser.parse_args()
    prompt_len = args.prompt_len
    if prompt_len is None:
        match = re.search(r"_prompt(\d+)", args.profile_dir.name)
        if not match:
            parser.error("--prompt-len is required for this profile directory")
        prompt_len = int(match.group(1))

    summary = json.loads((args.profile_dir / "summary.json").read_text())
    variant, loops = summary["variant"], summary["loops"]
    config = json.loads((MODELS / variant / "config.json").read_text())
    steps = decode_steps(json.load(gzip.open(args.profile_dir / "trace.json.gz")))

    measured = defaultdict(float)  # component -> microseconds summed over steps
    projection_us = defaultdict(float)
    projection_calls = defaultdict(int)
    period_us = 0.0
    for step in steps:
        period_us += step.next_start - step.start
        forward_ids = {id(op) for op in step.forward}
        for op in step.forward:
            if categorize(op["name"], op["cat"]) == "Fused weight GEMV":
                name = FUSED_PROJECTIONS[op["name"]]
                projection_us[name] += op["dur"]
                projection_calls[name] += 1
        gemms = [op for op in step.forward
                 if categorize(op["name"], op["cat"]) in ("Marlin weight GEMM", "Dense GEMM")]
        if len(gemms) % len(PROJECTIONS):
            raise RuntimeError(f"{len(gemms)} GEMMs per forward is not a multiple of "
                               f"{len(PROJECTIONS)}; cannot attribute projections")
        for index, op in enumerate(gemms):
            name = PROJECTIONS[index % len(PROJECTIONS)]
            projection_us[name] += op["dur"]
            projection_calls[name] += 1
        for op in step.ops:
            category = categorize(op["name"], op["cat"])
            if category in ("Marlin weight GEMM", "Dense GEMM", "Fused weight GEMV"):
                component = "Weight GEMMs" if id(op) in forward_ids else "Output projection"
            elif category == "Attention" or op["name"] == "reduce_segments":
                component = "Attention"
            elif category in ("Fused elementwise and norms (torch.compile)",
                              "Fused normalization and residual"):
                component = "Normalization and gating"
            else:
                component = "Sampling, bookkeeping, copies"
            measured[component] += op["dur"]
        busy = sum(op["dur"] for op in step.ops)
        measured["Idle GPU"] += max(0.0, (step.next_start - step.start) - busy)

    count = len(steps)
    to_ms = lambda us: us / count / 1000  # noqa: E731
    shapes = model_shapes(config)
    bytes_per_param = BYTES_PER_PARAM[variant]
    measured_projections = [name for name in shapes if projection_calls[name]]
    calls_per_step = {name: projection_calls[name] / count for name in measured_projections}
    ideal_projection_ms = {
        name: calls_per_step[name] * shapes[name][0] * shapes[name][1] * bytes_per_param
        / PEAK_BYTES_PER_S * 1000 for name in measured_projections}
    lm_head_bytes = config["vocab_size"] * config["hidden_size"] * 2
    kv_bytes_per_token = (2 * config["num_key_value_heads"] * config["head_dim"]
                          * KV_BYTES_PER_ELEMENT)
    windows = attention_applications(config, loops)
    # Context grows by one token per step after a prompt_len-token prefill.
    kv_bytes = sum(
        sum(min(prompt_len + 1 + i, w) if w else prompt_len + 1 + i for w in windows)
        for i in range(count)) / count * kv_bytes_per_token
    ideal = {
        "Weight GEMMs": sum(ideal_projection_ms.values()),
        "Output projection": lm_head_bytes / PEAK_BYTES_PER_S * 1000,
        "Attention": kv_bytes / PEAK_BYTES_PER_S * 1000,
        "Normalization and gating": SMALL_TENSOR_FLOOR_MS,
        "Sampling, bookkeeping, copies": 0.0,
        "Idle GPU": 0.0,
    }

    period_ms = to_ms(period_us)
    ideal_total = sum(ideal.values())
    gap_total = period_ms - ideal_total
    print(f"# Gap to bandwidth minimum: {variant}, {loops} loops, batch size 1, "
          f"{prompt_len}-token prompt ({args.profile_dir.name})")
    print()
    print(f"- Measured: {period_ms:.2f} ms per token ({1000 / period_ms:.0f} tokens/s)")
    print(f"- Bandwidth minimum: {ideal_total:.2f} ms per token ({1000 / ideal_total:.0f} tokens/s)"
          f" at {PEAK_BYTES_PER_S / 1e9:.0f} GB/s")
    print(f"- KV cache read per token: {kv_bytes / 1e6:.0f} MB over {len(windows)} attention"
          f" layer applications")
    print()
    print("| Component | Measured ms | Minimum ms | Gap ms | Share of total gap | Efficiency |")
    print("|---|---:|---:|---:|---:|---:|")
    for component in sorted(ideal, key=lambda c: -(to_ms(measured[c]) - ideal[c])):
        actual = to_ms(measured[component])
        gap = actual - ideal[component]
        efficiency = f"{ideal[component] / actual:.0%}" if ideal[component] > 0 and actual else "n/a"
        print(f"| {component} | {actual:.3f} | {ideal[component]:.3f} | {gap:.3f} "
              f"| {gap / gap_total:.0%} | {efficiency} |")
    print()
    print("| Weight projection | Calls per token | Mean us per call | Minimum us per call "
          "| Gap ms per token | Efficiency |")
    print("|---|---:|---:|---:|---:|---:|")
    for name in measured_projections:
        calls = calls_per_step[name]
        mean_us = projection_us[name] / projection_calls[name]
        min_us = ideal_projection_ms[name] * 1000 / calls
        print(f"| {name} | {calls:.0f} | {mean_us:.1f} | {min_us:.1f} "
              f"| {(mean_us - min_us) * calls / 1000:.3f} | {min_us / mean_us:.0%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
