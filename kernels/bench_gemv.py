"""Benchmarks single-token NVFP4 GEMV kernels on Dory's real weights.

Loads the NVFP4 projections of every attention and MLP layer from the
checkpoint, checks each implementation against a dequantized FP32 reference,
and times the projections in decode order (input layers, the recurrent layers
once per loop, output layers) captured in CUDA graphs. Consecutive calls use
different weights, so, as in real decoding, weights stream from DRAM instead
of the 72 MB L2 cache.

Implementations: vLLM's Marlin path (the production kernel on this GPU) and
the Triton prototype in w4a16_gemv.py, with a block-size sweep per shape. An
empty-kernel graph gives the per-launch floor.

Usage: .venv/bin/python kernels/bench_gemv.py [--loops 2]
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from w4a16_gemv import GROUP_SIZE, to_nibble_major, w4a16_gemv

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "nvfp4"
PEAK_BYTES_PER_S = 1008e9
BYTES_PER_PARAM = 0.5 + 1 / GROUP_SIZE  # FP4 value plus its share of an FP8 block scale

# Projection kinds and the checkpoint shards merged into each, in vLLM's order.
ATTENTION_KINDS = {"qkv": ("q_proj", "k_proj", "v_proj"), "gate": ("g_proj",),
                   "attn_out": ("o_proj",)}
MERGED_ATTENTION_KINDS = {"qkv_gate": ("q_proj", "k_proj", "v_proj", "g_proj")}
MLP_KINDS = {"gate_up": ("gate_proj", "up_proj"), "down": ("down_proj",)}
KIND_ORDER = ("qkv", "gate", "attn_out", "gate_up", "down", "qkv_gate")

TRITON_CONFIGS = [
    {"block_n": n, "block_w": w_block, "num_warps": w}
    for n, w_block, w in itertools.product((2, 4, 8, 16), (32, 64, 128), (2, 4, 8))
]


@dataclass
class Projection:
    kind: str
    layer: int
    weight: torch.Tensor  # uint8 [N, K/2]
    scale: torch.Tensor  # float8_e4m3fn [N, K/16]
    global_scale: torch.Tensor  # float32 [N]

    @property
    def shape(self) -> tuple[int, int]:
        return self.weight.shape[1] * 2, self.weight.shape[0]  # (K, N)

    @property
    def scalar_global_scale(self) -> float | None:
        values = torch.unique(self.global_scale)
        return values.item() if values.numel() == 1 else None


def require_idle_gpu() -> None:
    sys.path.insert(0, str(ROOT / "bench"))
    from run_baseline import gpu_busy_reason
    if (reason := gpu_busy_reason()) is not None:
        raise RuntimeError(f"the GPU is not idle ({reason}); refusing to benchmark")


def load_projections(device: torch.device) -> tuple[dict, dict]:
    """Returns ({(kind, layer): Projection}, config) for every projection kind."""
    config = json.loads((CHECKPOINT / "config.json").read_text())
    projections = {}
    with safe_open(str(CHECKPOINT / "model.safetensors"), framework="pt", device=str(device)) as f:
        for layer, layer_type in enumerate(config["layer_types"]):
            kinds = MLP_KINDS if layer_type == "mlp" else {**ATTENTION_KINDS, **MERGED_ATTENTION_KINDS}
            prefix = f"backbone.layers.{layer}.mixer"
            for kind, shards in kinds.items():
                weights, scales, globals_ = [], [], []
                for shard in shards:
                    weight = f.get_tensor(f"{prefix}.{shard}.weight")
                    weights.append(weight)
                    scales.append(f.get_tensor(f"{prefix}.{shard}.weight_scale"))
                    globals_.append(f.get_tensor(f"{prefix}.{shard}.weight_scale_2")
                                    .float().expand(weight.shape[0]))
                projections[(kind, layer)] = Projection(
                    kind, layer, torch.cat(weights).contiguous(), torch.cat(scales).contiguous(),
                    torch.cat(globals_).contiguous())
    return projections, config


def decode_order(config: dict, loops: int) -> list[int]:
    first = config["n_input_layers"]
    after = first + config["n_recurrent_layers"]
    layers = list(range(first))
    layers += list(range(first, after)) * loops
    layers += list(range(after, config["num_hidden_layers"]))
    return layers


def kinds_for_layer(config: dict, layer: int, merged_qkv_gate: bool) -> tuple[str, ...]:
    if config["layer_types"][layer] == "mlp":
        return ("gate_up", "down")
    return ("qkv_gate", "attn_out") if merged_qkv_gate else ("qkv", "gate", "attn_out")


def dequantize(p: Projection) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        break_fp4_bytes)
    values = break_fp4_bytes(p.weight, torch.float32)
    scales = p.scale.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=1)
    return values * scales * p.global_scale[:, None]


_MARLIN_WORKSPACES: dict[torch.device, torch.Tensor] = {}


def marlin_workspace(device: torch.device) -> torch.Tensor:
    """One persistent zeroed lock buffer per device, as vLLM's workspace manager keeps.

    Marlin resets its locks after each call, so sequential calls can share it.
    Passing workspace=None outside the engine would allocate and zero a new
    buffer on every call, adding a fill kernel that vLLM does not run.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        MARLIN_MAX_BLOCKS_PER_SM, marlin_make_workspace_new)
    if device not in _MARLIN_WORKSPACES:
        _MARLIN_WORKSPACES[device] = marlin_make_workspace_new(device, MARLIN_MAX_BLOCKS_PER_SM)
    return _MARLIN_WORKSPACES[device]


class MarlinProjection(torch.nn.Module):
    """One projection prepared and called exactly as vLLM's NVFP4 Marlin kernel does."""

    def __init__(self, p: Projection):
        super().__init__()
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_fp4_layer_for_marlin)
        global_scale = p.scalar_global_scale
        if global_scale is None:
            raise ValueError(f"{p.kind} layer {p.layer}: Marlin needs one global scale")
        k, n = p.shape
        self.input_size_per_partition, self.output_size_per_partition = k, n
        self.params_dtype = torch.bfloat16
        self.weight = torch.nn.Parameter(p.weight.clone(), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(p.scale.clone(), requires_grad=False)
        self.weight_global_scale = torch.nn.Parameter(
            torch.tensor(global_scale, dtype=torch.float32, device=p.weight.device),
            requires_grad=False)
        prepare_fp4_layer_for_marlin(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [K] for one token, returning [N]; or [M, K], returning [M, N]."""
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear)
        y = apply_fp4_marlin_linear(
            input=x if x.dim() == 2 else x[None, :], weight=self.weight,
            weight_scale=self.weight_scale, weight_global_scale=self.weight_global_scale,
            workspace=marlin_workspace(x.device), size_n=self.output_size_per_partition,
            size_k=self.input_size_per_partition)
        return y if x.dim() == 2 else y[0]


def relative_error(y: torch.Tensor, reference: torch.Tensor) -> float:
    return ((y.float() - reference).norm() / reference.norm()).item()


def time_graph(calls: list, replays: int) -> float:
    """Milliseconds per replay of a CUDA graph that runs `calls` in order."""
    for call in calls:  # compiles Triton kernels and allocates outputs before capture
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / replays


@triton.jit
def _empty_kernel(out_ptr):
    tl.store(out_ptr, 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--check-layers", type=int, nargs="+", default=[0, 1, 30, 31])
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    torch.manual_seed(0)

    projections, config = load_projections(device)
    order = decode_order(config, args.loops)
    shapes = {kind: next(p.shape for (k, _), p in projections.items() if k == kind)
              for kind in KIND_ORDER}
    inputs = {k: torch.randn(k, dtype=torch.bfloat16, device=device)
              for k in {shape[0] for shape in shapes.values()}}
    # The Triton kernel takes x as FP32 in nibble-major order; in a fused pipeline
    # the producing kernel writes that layout, so it is prepared outside the timing.
    nibble_major = {k: to_nibble_major(x) for k, x in inputs.items()}
    lines = [f"# Single-token NVFP4 GEMV benchmark ({torch.cuda.get_device_name()}, "
             f"{args.loops} loops, Triton {triton.__version__}, torch {torch.__version__})", ""]

    # Global scales: Marlin needs one per merged weight; record where shards differ.
    differing = sorted({p.kind for p in projections.values() if p.scalar_global_scale is None})
    lines.append(f"- Kinds whose merged shards have different global scales: {differing or 'none'}")

    marlin = {key: MarlinProjection(p) for key, p in projections.items()
              if p.scalar_global_scale is not None}
    best = {}

    # Sweep Triton configurations per kind on the decode-order sequence of that kind.
    sweep_rows = []
    for kind in KIND_ORDER:
        merged = kind == "qkv_gate"
        keys = [(kind, layer) for layer in order if kind in kinds_for_layer(config, layer, merged)]
        k, n = shapes[kind]
        x, xn = inputs[k], nibble_major[k]
        outs = [torch.empty(n, dtype=torch.bfloat16, device=device) for _ in keys]
        results = []
        for cfg in TRITON_CONFIGS:
            if k % (8 * cfg["block_w"]):
                continue
            calls = [lambda p=projections[key], o=o, cfg=cfg: w4a16_gemv(
                xn, p.weight, p.scale, p.global_scale, o, **cfg) for key, o in zip(keys, outs)]
            results.append((time_graph(calls, args.replays) / len(keys) * 1000, cfg))
        results.sort(key=lambda r: r[0])
        best[kind] = results[0][1]
        marlin_us = None
        if all(key in marlin for key in keys):
            calls = [lambda m=marlin[key]: m(x) for key in keys]
            marlin_us = time_graph(calls, args.replays) / len(keys) * 1000
        ideal_us = k * n * BYTES_PER_PARAM / PEAK_BYTES_PER_S * 1e6
        sweep_rows.append((kind, k, n, len(keys), ideal_us, marlin_us, results[0][0], results[0][1]))

    lines += ["", "## Per-call time in decode order (CUDA graph, wall time per call)", "",
              "| Projection | K x N | Calls per token | Minimum us | Marlin us | Triton best us "
              "| Triton efficiency | Best Triton config |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for kind, k, n, calls, ideal, marlin_us, triton_us, cfg in sweep_rows:
        marlin_text = f"{marlin_us:.1f}" if marlin_us is not None else "n/a"
        lines.append(f"| {kind} | {k} x {n} | {calls} | {ideal:.1f} | {marlin_text} | {triton_us:.1f} "
                     f"| {ideal / triton_us:.0%} | BLOCK_N={cfg['block_n']}, BLOCK_W={cfg['block_w']},"
                     f" warps={cfg['num_warps']} |")

    # Full decode-order sequences, as the model would issue them.
    def full_sequence(implementation: str, merged: bool) -> list:
        calls = []
        for layer in order:
            for kind in kinds_for_layer(config, layer, merged):
                p = projections[(kind, layer)]
                x = inputs[p.shape[0]]
                if implementation == "marlin":
                    calls.append(lambda m=marlin[(kind, layer)], x=x: m(x))
                else:
                    o = torch.empty(p.shape[1], dtype=torch.bfloat16, device=device)
                    calls.append(lambda p=p, xn=nibble_major[p.shape[0]], o=o, cfg=best[kind]:
                                 w4a16_gemv(xn, p.weight, p.scale, p.global_scale, o, **cfg))
        return calls

    ideal_total_ms = sum(projections[(kind, layer)].shape[0] * projections[(kind, layer)].shape[1]
                         for layer in order for kind in kinds_for_layer(config, layer, False)
                         ) * BYTES_PER_PARAM / PEAK_BYTES_PER_S * 1000
    lines += ["", "## Whole decode-order sequence per token", "",
              "| Implementation | Kernels | ms per token | Efficiency |", "|---|---:|---:|---:|"]
    for name, implementation, merged in (("Marlin (vLLM)", "marlin", False),
                                         ("Triton, separate q/k/v and gate", "triton", False),
                                         ("Triton, merged q/k/v + gate", "triton", True)):
        calls = full_sequence(implementation, merged)
        ms = time_graph(calls, args.replays)
        lines.append(f"| {name} | {len(calls)} | {ms:.3f} | {ideal_total_ms / ms:.0%} |")
    lines.append(f"| Bandwidth minimum | | {ideal_total_ms:.3f} | 100% |")

    flag = torch.zeros(1, device=device)
    empty_ms = time_graph([lambda: _empty_kernel[(1,)](flag)] * 280, args.replays)
    lines += ["", f"- Empty-kernel floor: {empty_ms / 280 * 1000:.2f} us per launch "
                  "(280 launches in one CUDA graph)"]

    # Correctness against the dequantized FP32 reference.
    lines += ["", "## Accuracy against a dequantized FP32 reference", "",
              "| Projection | Layer | Marlin relative error | Triton relative error |",
              "|---|---:|---:|---:|"]
    for layer in args.check_layers:
        for kind in kinds_for_layer(config, layer, False) + (
                ("qkv_gate",) if config["layer_types"][layer] != "mlp" else ()):
            p = projections[(kind, layer)]
            x = inputs[p.shape[0]]
            reference = dequantize(p) @ x.float()
            triton_out = w4a16_gemv(nibble_major[p.shape[0]], p.weight, p.scale, p.global_scale,
                                    torch.empty(p.shape[1], dtype=torch.bfloat16, device=device),
                                    **best[kind])
            marlin_text = (f"{relative_error(marlin[(kind, layer)](x), reference):.2e}"
                           if (kind, layer) in marlin else "n/a")
            lines.append(f"| {kind} | {layer} | {marlin_text} "
                         f"| {relative_error(triton_out, reference):.2e} |")

    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"gemv_bench_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
