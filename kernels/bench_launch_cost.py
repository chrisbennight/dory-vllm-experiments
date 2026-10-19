"""Measures the fixed cost of a kernel launch for the fused weight GEMVs.

A decode step launches each projection's GEMV once per layer (56 times per
token at 2 loops). Every launch pays start-up, ramp-up and drain time that a
persistent kernel could in principle overlap with useful work. This benchmark
streams the same weight bytes with the single-token NVFP4 GEMV (w4a16_gemv) in
56, 14, 4 and 1 launches per shape, by stacking the layers' weights (random
data of the real shapes; timing does not depend on the values). The time
saved per removed launch bounds what removing kernel boundaries can recover.

A persistent kernel replaces each boundary with a grid-wide barrier, so the
benchmark also times a barrier built from atomics, with every program
resident.

Usage: .venv/bin/python kernels/bench_launch_cost.py
"""

import argparse
import sys
from datetime import datetime, timezone

import torch
import triton
import triton.language as tl

from bench_gemv import BYTES_PER_PARAM, PEAK_BYTES_PER_S, ROOT, require_idle_gpu, time_graph
from w4a16_gemv import w4a16_gemv

LAYERS = 56
LAYERS_PER_LAUNCH = (1, 4, 14, 56)
SHAPES = {"q/k/v + output gate": (2560, 10240), "attention output": (4096, 2560),
          "MLP gate+up": (2560, 14336), "MLP down": (7168, 2560)}  # (K, N)
CONFIG = dict(block_n=4, block_w=64, num_warps=4)


@triton.jit
def _grid_barrier_kernel(counter_ptr, ROUNDS: tl.constexpr, NUM_PROGRAMS: tl.constexpr):
    for round_index in range(ROUNDS):
        tl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
        target = (round_index + 1) * NUM_PROGRAMS
        arrived = tl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")
        while arrived < target:
            arrived = tl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")


def barrier_us(programs: int, rounds: int, replays: int) -> float:
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    def run():
        counter.zero_()
        _grid_barrier_kernel[(programs,)](counter, ROUNDS=rounds, NUM_PROGRAMS=programs,
                                          num_warps=4)
    single = time_graph([run], replays)
    counter_one = torch.zeros(1, dtype=torch.int32, device="cuda")

    def run_one():
        counter_one.zero_()
        _grid_barrier_kernel[(programs,)](counter_one, ROUNDS=1, NUM_PROGRAMS=programs,
                                          num_warps=4)
    base = time_graph([run_one], replays)
    return (single - base) * 1000 / (rounds - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=int, default=30)
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    lines = [f"# Fixed cost per GEMV launch ({torch.cuda.get_device_name()}, {LAYERS} layers "
             f"per projection, launch configuration {CONFIG})", "",
             "Microseconds per layer for the same weight bytes split into fewer, larger "
             "launches. Minimum: the bytes at 1008 GB/s.", "",
             "| Projection | Minimum | " + " | ".join(
                 f"{LAYERS // g} launches" for g in LAYERS_PER_LAUNCH)
             + f" | Saving per removed launch ({LAYERS} to 1) |",
             "|---|---:|" + "---:|" * len(LAYERS_PER_LAUNCH) + "---:|"]
    total_saving_ms = 0.0
    for name, (k, n) in SHAPES.items():
        weight = torch.randint(0, 256, (LAYERS * n, k // 2), dtype=torch.uint8, device=device)
        scale = (torch.rand(LAYERS * n, k // 16, device=device) + 0.5).to(torch.float8_e4m3fn)
        global_scale = torch.full((LAYERS * n,), 1e-3, device=device)
        x = torch.randn(k, device=device)
        x_nm = x.view(-1, 8).T.contiguous()
        out = torch.empty(LAYERS * n, dtype=torch.bfloat16, device=device)
        per_layer = []
        for group in LAYERS_PER_LAUNCH:
            rows = group * n
            calls = [lambda s=slice(i * rows, (i + 1) * rows): w4a16_gemv(
                x_nm, weight[s], scale[s], global_scale[s], out[s], **CONFIG)
                for i in range(LAYERS // group)]
            per_layer.append(time_graph(calls, args.replays) * 1000 / LAYERS)
        minimum = k * n * BYTES_PER_PARAM / PEAK_BYTES_PER_S * 1e6
        saving = (per_layer[0] - per_layer[-1]) * LAYERS / (LAYERS - 1)
        total_saving_ms += saving * LAYERS / 1000
        lines.append(f"| {name} ({k} x {n}) | {minimum:.1f} | "
                     + " | ".join(f"{t:.1f}" for t in per_layer) + f" | {saving:.2f} |")
        print(lines[-1], flush=True)
        del weight, scale, global_scale, out
        torch.cuda.empty_cache()
    lines += ["", f"Removing every GEMV launch boundary of a decode step (4 x {LAYERS} launches) "
              f"would save at most about {total_saving_ms:.2f} ms per token at this launch "
              "configuration, before the cost of synchronizing the steps inside one kernel.",
              "", "## Grid-wide barrier inside one kernel", "",
              "| Resident programs | us per barrier |", "|---:|---:|"]
    for programs in (sms, 2 * sms, 4 * sms):
        us = barrier_us(programs, rounds=200, replays=args.replays)
        lines.append(f"| {programs} | {us:.2f} |")
        print(lines[-1], flush=True)

    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"launch_cost_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
