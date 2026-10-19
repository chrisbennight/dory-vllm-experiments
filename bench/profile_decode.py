"""Profiles single-user Dory decode steps under vLLM and attributes GPU time.

Runs the vLLM engine in-process so torch.profiler records every CUDA kernel,
generates from a fixed-length random prompt at batch size 1 with the same
sampling settings as the serving benchmark, and splits each decode step into
the model forward (one full CUDA graph replay) and the work that follows it
(output projection, sampling, bookkeeping). Writes a gzipped Chrome trace
(viewable in https://ui.perfetto.dev) and a Markdown report.

Usage: python bench/profile_decode.py --variant nvfp4 --loops 2
"""

import argparse
import gzip
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# Must be set before vLLM is imported: the in-process engine lets the profiler
# see GPU work. FlashInfer's sampler is JIT-compiled and needs a CUDA compiler
# (see bench/setup_cuda_toolkit.sh), so it stays off unless the environment
# sets VLLM_USE_FLASHINFER_SAMPLER.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# JIT builds run tools installed in the environment (FlashInfer runs ninja),
# which are only found on PATH when the environment is activated.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"

HIDDEN, INTERMEDIATE, HEADS, KV_HEADS, HEAD_DIM = 2560, 7168, 16, 4, 256
INPUT_PAIRS, OUTPUT_PAIRS, RECURRENT_PAIRS = 8, 8, 20
PAIR_PARAMS = (HIDDEN * (2 * HEADS + 2 * KV_HEADS) * HEAD_DIM + HEADS * HEAD_DIM * HIDDEN
               + 3 * HIDDEN * INTERMEDIATE)
BYTES_PER_PARAM = {"bf16": 2.0, "mxfp8": 1 + 1 / 32, "nvfp4": 0.5 + 1 / 16}
RTX_4090_PEAK_BYTES_PER_S = 1008e9

# First matching pattern wins, so specific families come before generic ones.
CATEGORIES = (
    # Kernels from kernels/ in the investigation repository (fused decode path).
    ("Fused weight GEMV", re.compile(
        r"^_(qkvg_rope|o_gated_normalized|gate_up_swiglu|down_normalized|w4a16_gemv"
        r"|quantized_logits)_kernel$")),
    ("Fused normalization and residual", re.compile(
        r"^_(residual_update|to_residual|qk_norm_cache)_kernel$")),
    ("Marlin weight GEMM", re.compile(r"marlin", re.I)),
    ("Attention", re.compile(r"flash|fmha|attn|attention|reshape_and_cache|paged"
                             r"|^_combine_gate_kernel$", re.I)),
    ("Sampling", re.compile(r"topk|top_k|topp|top_p|sampl|gumbel|exponential|multinomial"
                            r"|argmax|_softmax|softmax_", re.I)),
    ("Dense GEMM", re.compile(r"gemm|gemv|cutlass|cublas|xmma|nvjet|splitk|matmul", re.I)),
    ("Rotary embedding", re.compile(r"rotary", re.I)),
    ("Fused elementwise and norms (torch.compile)", re.compile(r"^triton_")),
)


def categorize(name: str, kind: str) -> str:
    if kind != "kernel":
        return "Memory copies and sets"
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "Other"


def gpu_is_idle() -> bool:
    from run_baseline import gpu_busy_reason
    return gpu_busy_reason() is None


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


@dataclass
class DecodeStep:
    """One decode step: the model forward graph replay plus everything up to the next one."""
    forward: list[dict]  # kernels of the full CUDA graph replay, in time order
    ops: list[dict]  # every GPU op in the step, including other streams
    start: float
    forward_end: float
    next_start: float


def busy_time(ops: list[dict]) -> float:
    return union_length([(op["ts"], op["ts"] + op["dur"]) for op in ops])


def decode_steps(trace: dict) -> list[DecodeStep]:
    """Segments a profiler trace into decode steps, bounded by consecutive forwards."""
    events = trace["traceEvents"]
    graph_launches = {e["args"]["correlation"] for e in events
                      if e.get("cat") == "cuda_runtime" and e.get("name") == "cudaGraphLaunch"}
    gpu_ops = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    gpu_ops.sort(key=lambda e: e["ts"])

    groups = defaultdict(list)
    for op in gpu_ops:
        correlation = op.get("args", {}).get("correlation")
        if correlation in graph_launches:
            groups[correlation].append(op)
    if not groups:
        raise RuntimeError("no CUDA graph kernels in the trace; cannot segment decode steps")
    # Decode steps replay one full CUDA graph holding the whole model; prefill
    # uses many small piecewise graphs, so the largest common size is decode.
    size_counts = Counter(len(ops) for ops in groups.values())
    decode_size = max(size for size, count in size_counts.items() if count >= 4)
    forwards = sorted((sorted(ops, key=lambda op: op["ts"]) for ops in groups.values()
                       if len(ops) == decode_size), key=lambda ops: ops[0]["ts"])
    # The last forward has no following step to bound its post-forward work.
    steps = []
    for forward, next_forward in zip(forwards, forwards[1:]):
        start, next_start = forward[0]["ts"], next_forward[0]["ts"]
        steps.append(DecodeStep(
            forward=forward,
            ops=[op for op in gpu_ops if start <= op["ts"] < next_start],
            start=start,
            forward_end=max(op["ts"] + op["dur"] for op in forward),
            next_start=next_start,
        ))
    return steps


def analyze(trace: dict, variant: str, loops: int, prompt_len: int,
            backend: str) -> tuple[str, dict]:
    segmented = decode_steps(trace)
    steps = len(segmented)
    decode_size = len(segmented[0].forward)

    per_category = defaultdict(lambda: {"us": 0.0, "count": 0})
    per_kernel = defaultdict(lambda: {"us": 0.0, "count": 0, "category": ""})
    forward_span = forward_busy = post_busy = period_total = idle_total = 0.0
    forward_gemm_us = 0.0
    for step in segmented:
        forward_ids = {id(op) for op in step.forward}
        period_total += step.next_start - step.start
        idle_total += (step.next_start - step.start) - busy_time(step.ops)
        forward_span += step.forward_end - step.start
        forward_busy += busy_time(step.forward)
        post_busy += busy_time([op for op in step.ops if op["ts"] >= step.forward_end])
        for op in step.ops:
            category = categorize(op["name"], op["cat"])
            per_category[category]["us"] += op["dur"]
            per_category[category]["count"] += 1
            kernel = per_kernel[op["name"]]
            kernel["us"] += op["dur"]
            kernel["count"] += 1
            kernel["category"] = category
            if id(op) in forward_ids and category in ("Marlin weight GEMM", "Dense GEMM",
                                                      "Fused weight GEMV"):
                forward_gemm_us += op["dur"]

    sublayers = 2 * (INPUT_PAIRS + OUTPUT_PAIRS + RECURRENT_PAIRS * loops)
    pairs = sublayers // 2
    weight_bytes = pairs * PAIR_PARAMS * BYTES_PER_PARAM[variant]
    period = period_total / steps
    span = forward_span / steps
    busy = forward_busy / steps
    gemm = forward_gemm_us / steps
    lines = [
        f"# Decode profile: {variant}, {loops} loops, batch size 1, "
        f"{prompt_len}-token prompt, attention backend {backend}",
        "",
        f"- Decode steps analyzed: {steps}; kernels per model forward: {decode_size}"
        f" ({decode_size / sublayers:.1f} per sublayer, {sublayers} sublayers)",
        f"- Step period: {period / 1000:.2f} ms ({1e6 / period:.1f} tokens/s)",
        f"- Model forward: {span / 1000:.2f} ms wall time, {busy / 1000:.2f} ms with a kernel"
        f" running, {(span - busy) / 1000:.2f} ms idle between kernels"
        f" ({(span - busy) / decode_size:.2f} us per kernel boundary)",
        f"- After the forward (output projection, sampling, bookkeeping):"
        f" {(period - span) / 1000:.2f} ms wall time, {post_busy / steps / 1000:.2f} ms busy",
        f"- Weight GEMMs inside the forward: {gemm / 1000:.2f} ms for"
        f" {weight_bytes / 1e9:.2f} GB = {weight_bytes / (gemm / 1e6) / 1e9:.0f} GB/s"
        f" ({weight_bytes / (gemm / 1e6) / RTX_4090_PEAK_BYTES_PER_S:.0%} of the 1008 GB/s peak)",
        "",
        "## Time per decode step by category",
        "",
        "| Category | ms per step | Share of step | us per sublayer | Kernels per step |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered = sorted(per_category.items(), key=lambda kv: -kv[1]["us"])
    idle = idle_total / steps
    for label, value in ordered + [("Idle GPU (gaps between kernels and steps)",
                                    {"us": idle_total, "count": 0})]:
        per_step = value["us"] / steps
        lines.append(f"| {label} | {per_step / 1000:.3f} | {per_step / period:.0%} "
                     f"| {per_step / sublayers:.1f} | {value['count'] / steps:.0f} |")
    lines += ["", "## Top kernels", "",
              "| Kernel | Category | Calls per step | Mean us | ms per step |",
              "|---|---|---:|---:|---:|"]
    for name, value in sorted(per_kernel.items(), key=lambda kv: -kv[1]["us"])[:25]:
        short = name if len(name) <= 90 else name[:87] + "..."
        short = short.replace("|", "\\|")
        lines.append(f"| `{short}` | {value['category']} | {value['count'] / steps:.0f} "
                     f"| {value['us'] / value['count']:.1f} | {value['us'] / steps / 1000:.3f} |")
    summary = {
        "variant": variant, "loops": loops, "steps": steps, "kernels_per_forward": decode_size,
        "period_us": period, "forward_span_us": span, "forward_busy_us": busy,
        "forward_gemm_us": gemm, "weight_bytes": weight_bytes,
        "categories_us_per_step": {k: v["us"] / steps for k, v in per_category.items()},
        "idle_us_per_step": idle,
    }
    return "\n".join(lines) + "\n", summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=list(BYTES_PER_PARAM))
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--attention-backend", default=None,
                        help="vLLM attention backend name; vLLM chooses when omitted")
    parser.add_argument("--no-async-scheduling", action="store_true",
                        help="run without asynchronous scheduling, as vLLM does for "
                             "speculative methods that do not support it")
    args = parser.parse_args()
    if not gpu_is_idle():
        raise RuntimeError("another process is using the GPU; refusing to profile")

    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backend_tag = f"_{args.attention_backend.lower()}" if args.attention_backend else ""
    out_dir = (ROOT / "results" / f"profile_{stamp}_{args.variant}_loops{args.loops}"
               f"_prompt{args.prompt_len}{backend_tag}")
    out_dir.mkdir(parents=True)

    llm = LLM(
        model=str(MODELS / args.variant),
        hf_overrides={"n_recurrent_loops": args.loops, "recurrent_kv_cache_mode": "per_loop"},
        max_model_len=8192, max_num_seqs=64, gpu_memory_utilization=0.90,
        async_scheduling=not args.no_async_scheduling, seed=0,
        attention_backend=args.attention_backend,
    )
    rng = random.Random(0)
    prompt = TokensPrompt(prompt_token_ids=[rng.randrange(1000, 100000)
                                            for _ in range(args.prompt_len)])
    # Same sampling as the serving benchmark, which used the model's defaults.
    sampling = SamplingParams(max_tokens=args.decode_steps + 1, ignore_eos=True,
                              temperature=1.0, top_p=0.95, seed=0)
    for _ in range(2):
        llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        llm.generate([prompt], sampling, use_tqdm=False)
        torch.cuda.synchronize()
    trace_path = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    with trace_path.open("rb") as source, gzip.open(f"{trace_path}.gz", "wb") as target:
        shutil.copyfileobj(source, target)
    trace_path.unlink()

    report, summary = analyze(trace, args.variant, args.loops, args.prompt_len,
                              args.attention_backend or "vLLM default")
    summary["flashinfer_sampler"] = os.environ["VLLM_USE_FLASHINFER_SAMPLER"]
    summary["dory_fused_decode"] = os.environ.get("DORY_FUSED_DECODE")
    summary["dory_fold_residual"] = os.environ.get("DORY_FOLD_RESIDUAL")
    summary["dory_fused_kv_write"] = os.environ.get("DORY_FUSED_KV_WRITE")
    summary["dory_fused_attention"] = os.environ.get("DORY_FUSED_ATTENTION")
    # The fused path's default LM head is INT8 (kernels/dory_fused_model.py).
    summary["dory_lm_head_8bit"] = (os.environ.get("DORY_LM_HEAD_8BIT", "int8")
                                    if os.environ.get("DORY_FUSED_DECODE") else None)
    summary["cuda_home"] = os.environ.get("CUDA_HOME")
    # "0" forces vLLM's V1 model runner; unset, vLLM picks V2 when it supports
    # the configuration.
    summary["vllm_use_v2_model_runner"] = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    summary["async_scheduling"] = not args.no_async_scheduling
    report += (f"\nFlashInfer sampler: {summary['flashinfer_sampler']};"
               f" CUDA_HOME: {summary['cuda_home']};"
               f" VLLM_USE_V2_MODEL_RUNNER: {summary['vllm_use_v2_model_runner']};"
               f" async scheduling: {summary['async_scheduling']}\n")
    (out_dir / "report.md").write_text(report)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(report)
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
