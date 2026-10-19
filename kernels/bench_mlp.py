"""Validates and times the fused Dory MLP sublayer against vLLM's implementation.

Three implementations of one MLP sublayer plus its residual update, on the
real NVFP4 weights:

- reference: dequantized FP32 weights and FP32 math throughout;
- vllm: vLLM's path, Marlin GEMMs with vLLM's own Dory functions (justnorm,
  slerp_alpha) for the elementwise steps, compiled with torch.compile as vLLM
  compiles them;
- fused: the three kernels in dory_mlp.py.

Accuracy is reported per layer for a random unit-norm input and for a chain of
all MLP sublayers in decode order, each output feeding the next. Timing replays
the decode-order sequence of MLP sublayers (2 loops) in CUDA graphs and breaks
the time down per kernel with the PyTorch profiler.

Usage: .venv/bin/python kernels/bench_mlp.py
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
import triton
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile

from bench_gemv import (CHECKPOINT, ROOT, MarlinProjection, Projection, decode_order, dequantize,
                        require_idle_gpu, time_graph)
from dory_mlp import EPS, FusedMlpSublayer, LaunchConfig, MlpWeights, load_mlp_weights
from w4a16_gemv import to_nibble_major

GATE_UP_CONFIGS = [LaunchConfig(*c) for c in
                   ((2, 64, 4), (2, 32, 4), (2, 64, 8), (4, 64, 4), (4, 32, 4), (2, 128, 4))]
DOWN_CONFIGS = [LaunchConfig(*c) for c in
                ((4, 64, 8), (4, 32, 8), (2, 64, 8), (8, 64, 8), (4, 128, 4), (4, 64, 4))]
RESIDUAL_WARPS = (4, 8, 16)


@dataclass
class MlpLayer:
    index: int
    fused: MlpWeights
    marlin_gate_up: MarlinProjection
    marlin_down: MarlinProjection
    gate_up: Projection  # vLLM order: gate rows, then up rows
    down: Projection


def load_layers(config: dict, device: torch.device) -> dict[int, MlpLayer]:
    layers = {}
    with safe_open(str(CHECKPOINT / "model.safetensors"), framework="pt", device=str(device)) as f:
        for index, layer_type in enumerate(config["layer_types"]):
            if layer_type != "mlp":
                continue
            fused = load_mlp_weights(f, index)
            mixer = f"backbone.layers.{index}.mixer"
            parts = {}
            for name in ("gate_proj", "up_proj", "down_proj"):
                weight = f.get_tensor(f"{mixer}.{name}.weight")
                parts[name] = (weight, f.get_tensor(f"{mixer}.{name}.weight_scale"),
                               f.get_tensor(f"{mixer}.{name}.weight_scale_2").float()
                               .expand(weight.shape[0]))
            gate_up = Projection("gate_up", index,
                                 *(torch.cat((g, u)).contiguous()
                                   for g, u in zip(parts["gate_proj"], parts["up_proj"])))
            down = Projection("down", index, *(t.contiguous() for t in parts["down_proj"]))
            layers[index] = MlpLayer(index, fused, MarlinProjection(gate_up),
                                     MarlinProjection(down), gate_up, down)
    return layers


def reference_sublayer(layer: MlpLayer, h: torch.Tensor, alpha_scale: float) -> torch.Tensor:
    """One MLP sublayer and residual update in FP32 with dequantized weights."""
    w = layer.fused
    suv_factor = math.sqrt(w.hidden_size)
    gate, up = (dequantize(layer.gate_up) @ h).chunk(2)
    m = F.silu(gate * w.suv_gate.float() * suv_factor) * (up * w.suv_up.float() * suv_factor)
    out = dequantize(layer.down) @ (m * torch.rsqrt(m.square().sum() + EPS))
    ya = h * torch.rsqrt(h.square().sum() + EPS)
    yb = out * torch.rsqrt(out.square().sum() + EPS)
    z = ya + (w.mlp_alpha.float() * alpha_scale).abs() * (yb - ya)
    return z * torch.rsqrt(z.square().sum() + EPS)


class VllmSublayer:
    """vLLM's MLP sublayer: Marlin GEMMs and vLLM's Dory functions, compiled like vLLM."""

    def __init__(self, hidden_size: int, alpha_scale: float):
        from vllm.model_executor.models.dory import justnorm, slerp_alpha
        suv_factor = math.sqrt(hidden_size)

        def swiglu_norm(m, suv_gate, suv_up):  # DoryMLP.forward between its two GEMMs
            g, u = m.chunk(2, dim=-1)
            g = (g.float() * (suv_gate.float() * suv_factor)).to(g.dtype)
            u = (u.float() * (suv_up.float() * suv_factor)).to(u.dtype)
            return justnorm((F.silu(g.float()) * u.float()).to(m.dtype))

        def residual(a, b, alpha):
            return slerp_alpha(a, b, alpha, alpha_scale)

        self.swiglu_norm = torch.compile(swiglu_norm, fullgraph=True)
        self.residual = torch.compile(residual, fullgraph=True)

    def __call__(self, layer: MlpLayer, h: torch.Tensor) -> torch.Tensor:
        """h: BF16 [1, H]; returns the sublayer output after the residual update."""
        w = layer.fused
        m = layer.marlin_gate_up(h[0])[None]
        m = self.swiglu_norm(m, w.suv_gate, w.suv_up)
        out = layer.marlin_down(m[0])[None]
        return self.residual(h, out, w.mlp_alpha)


def relative_error(x: torch.Tensor, reference: torch.Tensor) -> float:
    return ((x.float().flatten() - reference.flatten()).norm() / reference.norm()).item()


def kernel_breakdown(run, replays: int = 5) -> dict[str, float]:
    """Microseconds per replay of each kernel in a CUDA graph capturing `run`."""
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    totals = defaultdict(float)
    for event in prof.events():
        if event.device_type.name == "CUDA":
            totals[event.name] += event.device_time / replays
    return dict(totals)


def short_name(name: str) -> str:
    if "marlin" in name.lower():
        return "Marlin GEMM"
    return name if len(name) <= 60 else name[:57] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--chain-inputs", type=int, default=8)
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    torch.manual_seed(0)

    config = json.loads((CHECKPOINT / "config.json").read_text())
    H, F_ = config["hidden_size"], config["intermediate_size"]
    alpha_scale = config["ngpt_alpha_init"] / config["initializer_range"]
    layers = load_layers(config, device)
    order = [i for i in decode_order(config, args.loops) if i in layers]
    vllm = VllmSublayer(H, alpha_scale)

    def unit_input() -> torch.Tensor:
        h = torch.randn(H, device=device)
        return (h / h.norm()).to(torch.bfloat16)

    def make_fused(gate_up: LaunchConfig, down: LaunchConfig, residual_warps: int,
                   fuse_residual: bool = True):
        return FusedMlpSublayer(H, F_, alpha_scale, device, gate_up, down, residual_warps,
                                fuse_residual)

    buffers = [(torch.empty(H, dtype=torch.bfloat16, device=device),
                torch.empty(8, H // 8, dtype=torch.float32, device=device)) for _ in range(2)]

    def fused_sequence(fused, h0: torch.Tensor):
        """Runs every MLP sublayer in decode order, alternating two output buffers."""
        h0_nibble_major = to_nibble_major(h0.float())

        def run():
            h, h_nm = h0, h0_nibble_major
            for step, index in enumerate(order):
                out, out_nm = buffers[step % 2]
                fused(layers[index].fused, h, h_nm, out, out_nm)
                h, h_nm = out, out_nm
            return h
        return run

    def vllm_sequence(h0: torch.Tensor):
        def run():
            h = h0[None]
            for index in order:
                h = vllm(layers[index], h)
            return h[0]
        return run

    lines = [f"# Fused Dory MLP sublayer ({torch.cuda.get_device_name()}, {args.loops} loops, "
             f"{len(order)} MLP sublayers per token)", ""]

    # Tune launch configurations by coordinate descent on the whole sequence.
    h0 = unit_input()
    best_gate_up, best_down, best_warps = GATE_UP_CONFIGS[0], DOWN_CONFIGS[0], 8
    tuning = []
    for stage, candidates in (("gate_up", GATE_UP_CONFIGS), ("down", DOWN_CONFIGS),
                              ("residual", RESIDUAL_WARPS)):
        timings = []
        for candidate in candidates:
            cfg = {"gate_up": best_gate_up, "down": best_down, "residual": best_warps}
            cfg[stage] = candidate
            # Residual warps only matter with the separate residual kernel.
            fused = make_fused(cfg["gate_up"], cfg["down"], cfg["residual"],
                               fuse_residual=stage != "residual")
            timings.append((time_graph([fused_sequence(fused, h0)], args.replays), candidate))
        ms, chosen = min(timings, key=lambda t: t[0])
        tuning.append(f"- {stage}: {chosen} ({ms * 1000 / len(order):.2f} us per sublayer)")
        if stage == "gate_up":
            best_gate_up = chosen
        elif stage == "down":
            best_down = chosen
        else:
            best_warps = chosen
    fused = make_fused(best_gate_up, best_down, best_warps)
    # The same kernels with the residual update as its own kernel.
    separate = make_fused(best_gate_up, best_down, best_warps, fuse_residual=False)
    lines += ["## Chosen launch configurations", ""] + tuning

    # Accuracy, one sublayer per layer.
    lines += ["", "## Accuracy: one sublayer, random unit-norm input", "",
              "Relative error of the sublayer output against the FP32 reference.", "",
              "| Layer | vLLM path | Fused | Fused vs vLLM path "
              "| Fused vs separate residual kernel |", "|---:|---:|---:|---:|---:|"]
    worst_vllm = worst_fused = 0.0
    for index in sorted(layers):
        h = unit_input()
        reference = reference_sublayer(layers[index], h.float(), alpha_scale)
        vllm_out = vllm(layers[index], h[None])[0]
        out, out_nm = buffers[0]
        fused(layers[index].fused, h, to_nibble_major(h.float()), out, out_nm)
        out_separate, out_separate_nm = buffers[1]
        separate(layers[index].fused, h, to_nibble_major(h.float()), out_separate,
                 out_separate_nm)
        e_vllm, e_fused = relative_error(vllm_out, reference), relative_error(out, reference)
        worst_vllm, worst_fused = max(worst_vllm, e_vllm), max(worst_fused, e_fused)
        lines.append(f"| {index} | {e_vllm:.2e} | {e_fused:.2e} "
                     f"| {relative_error(out, vllm_out.float()):.2e} "
                     f"| {relative_error(out, out_separate.float()):.2e} |")
    lines.append(f"| worst | {worst_vllm:.2e} | {worst_fused:.2e} | | |")

    # Accuracy along the whole decode-order chain of MLP sublayers. Small
    # differences grow along a chain, so several inputs are averaged.
    errors = defaultdict(list)
    for _ in range(args.chain_inputs):
        h = unit_input()
        reference = h.float()
        for index in order:
            reference = reference_sublayer(layers[index], reference, alpha_scale)
        vllm_final = vllm_sequence(h)().float()
        fused_final = fused_sequence(fused, h)().float()
        errors["vLLM path vs FP32 reference"].append(relative_error(vllm_final, reference))
        errors["Fused vs FP32 reference"].append(relative_error(fused_final, reference))
        errors["Fused vs vLLM path"].append(relative_error(fused_final, vllm_final))
    lines += ["", f"## Accuracy: chain of all {len(order)} MLP sublayers in decode order "
                  f"({args.chain_inputs} random inputs)", "",
              "| Comparison | Mean relative error | Worst relative error |", "|---|---:|---:|"]
    for name, values in errors.items():
        lines.append(f"| {name} | {sum(values) / len(values):.2e} | {max(values):.2e} |")

    # Timing of the decode-order sequence.
    vllm_ms = time_graph([vllm_sequence(h)], args.replays)
    fused_ms = time_graph([fused_sequence(fused, h)], args.replays)
    separate_ms = time_graph([fused_sequence(separate, h)], args.replays)
    per = 1000 / len(order)
    lines += ["", "## Time per token for all MLP sublayers (CUDA graph)", "",
              "| Implementation | Kernels per sublayer | ms per token | us per sublayer |",
              "|---|---:|---:|---:|",
              f"| vLLM path (Marlin + torch.compile) | 4 | {vllm_ms:.3f} | {vllm_ms * per:.1f} |",
              f"| Fused, separate residual kernel | 3 | {separate_ms:.3f} "
              f"| {separate_ms * per:.1f} |",
              f"| Fused | 2 | {fused_ms:.3f} | {fused_ms * per:.1f} |",
              f"| Saving | | {vllm_ms - fused_ms:.3f} | {(vllm_ms - fused_ms) * per:.1f} |"]

    lines += ["", "## Kernel time per sublayer (profiler, us)", "",
              "| Implementation | Kernel | us per sublayer |", "|---|---|---:|"]
    for name, run in (("vLLM path", vllm_sequence(h)), ("Fused", fused_sequence(fused, h))):
        totals = defaultdict(float)
        for kernel, us in kernel_breakdown(run).items():
            totals[short_name(kernel)] += us
        for kernel, us in sorted(totals.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | `{kernel}` | {us / len(order):.1f} |")

    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"mlp_bench_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
