"""Runs the vLLM serving benchmark matrix for the Dory checkpoints.

For every (checkpoint, loop count) pair this starts a dedicated vLLM server,
records greedy reference generations, runs `vllm bench serve` for each
workload shape and concurrency level, and stops the server. Results are
written under results/<run id>/ and never overwrite an earlier run.
"""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = ROOT / ".venv" / "bin"
MODELS = ROOT / "models"

VLLM_PR_SHA = "55076950b6c155f384d6766d63563c3bc20a0e3f"
MODEL_REVISIONS = {
    "bf16": "b366b87f762ea334ba3eb55912031baba1515810",
    "mxfp8": "f35880f52368271136b48e3a4365f91e70a92308",
    "nvfp4": "41f02335e1a1ad3289b8b32485e2265cbe97cfdf",
}
LOOPS = (1, 2, 4)
# (input tokens, output tokens); both match NVIDIA's published DGX Spark table.
SHAPES = ((2560, 300), (128, 1024))
# Concurrency -> number of measured requests. Concurrency 1 is the
# single-user case that a custom decode engine would target.
CONCURRENCY_PROMPTS = {1: 8, 16: 64, 64: 128}
WARMUP_REQUESTS = 2

PORT = 8011
SERVED_NAME = "dory"
# FlashInfer's sampler is JIT-compiled on first use, which needs a CUDA compiler
# (see bench/setup_cuda_toolkit.sh). It stays off unless the environment sets
# VLLM_USE_FLASHINFER_SAMPLER, so results stay comparable with earlier runs.
SERVER_ENV_DEFAULTS = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
SERVER_ENV = {name: os.environ.get(name, value)
              for name, value in SERVER_ENV_DEFAULTS.items()}
SERVER_READY_TIMEOUT_S = 1200
# Anything above this much resident GPU memory before a server starts means
# another workload is sharing the GPU and the numbers would be contaminated.
IDLE_GPU_MEMORY_LIMIT_MIB = 1024
# The desktop alone stays well below this average load; a game, launcher or
# video left running does not, and the GPU then time-slices it with the
# benchmark.
IDLE_GPU_UTILIZATION_LIMIT_PCT = 10

REFERENCE_PROMPTS = (
    "What is 17 * 23? Answer with just the number.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in two sentences why the sky is blue.",
    "List the first five prime numbers greater than 100.",
)
REFERENCE_MAX_TOKENS = 256


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def gpu_snapshot() -> dict:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,"
         "utilization.gpu,temperature.gpu,power.limit,clocks.max.sm",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(", ")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    keys = ("name", "driver", "memory_used_mib", "memory_total_mib",
            "utilization_pct", "temperature_c", "power_limit_w", "max_sm_clock_mhz")
    snapshot = dict(zip(keys, query))
    snapshot["compute_apps"] = [line for line in apps.splitlines() if line]
    return snapshot


class GpuBusyError(RuntimeError):
    """Another workload holds the GPU, so measurements would be contaminated."""


def gpu_busy_reason(samples: int = 5) -> str | None:
    """Why the GPU is not idle, or None when it is. A game or other graphics
    program is not listed as a compute process, so memory use and the load
    averaged over `samples` seconds are checked too."""
    snapshot = gpu_snapshot()
    if snapshot["compute_apps"]:
        return f"compute processes {snapshot['compute_apps']}"
    if int(snapshot["memory_used_mib"]) > IDLE_GPU_MEMORY_LIMIT_MIB:
        return f"{snapshot['memory_used_mib']} MiB used"
    loads = [int(snapshot["utilization_pct"])]
    for _ in range(samples - 1):
        time.sleep(1)
        loads.append(int(gpu_snapshot()["utilization_pct"]))
    load = sum(loads) / len(loads)
    if load > IDLE_GPU_UTILIZATION_LIMIT_PCT:
        return f"{load:.0f}% average load with no compute process (a graphics program?)"
    return None


def require_idle_gpu(grace_s: float = 90) -> None:
    # A server stopped moments ago can still be releasing its memory.
    deadline = time.monotonic() + grace_s
    while True:
        reason = gpu_busy_reason()
        if reason is None:
            return
        if time.monotonic() >= deadline:
            raise GpuBusyError(
                f"GPU is not idle ({reason}); refusing to benchmark on a shared GPU")
        time.sleep(5)


def http_json(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_until_ready(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5):
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    raise RuntimeError(f"server not ready after {SERVER_READY_TIMEOUT_S}s")


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def start_server(variant: str, loops: int, attention_backend: str | None,
                 server_log: Path, extra_args: list[str] | None = None,
                 async_scheduling: bool = True,
                 extra_overrides: dict | None = None) -> subprocess.Popen:
    overrides = {"n_recurrent_loops": loops, "recurrent_kv_cache_mode": "per_loop",
                 **(extra_overrides or {})}
    command = [
        str(VENV_BIN / "vllm"), "serve", str(MODELS / variant),
        "--served-model-name", SERVED_NAME,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--max-model-len", "8192",
        "--max-num-seqs", str(max(CONCURRENCY_PROMPTS)),
        "--gpu-memory-utilization", "0.90",
        "--hf-overrides", json.dumps(overrides),
        "--async-scheduling" if async_scheduling else "--no-async-scheduling",
        "--seed", "0",
    ]
    if attention_backend:
        command += ["--attention-backend", attention_backend]
    command += extra_args or []
    log(f"starting server: {variant} loops={loops} attention={attention_backend or 'default'}")
    handle = server_log.open("w")
    return subprocess.Popen(
        command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT,
        env={**os.environ, **SERVER_ENV,
             # JIT builds run tools installed in the environment (FlashInfer
             # runs ninja), which are only found on PATH when it is activated.
             "PATH": f"{VENV_BIN}{os.pathsep}{os.environ['PATH']}"},
        start_new_session=True)


def capture_reference(out_path: Path) -> None:
    generations = []
    for prompt in REFERENCE_PROMPTS:
        response = http_json(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            {
                "model": SERVED_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": REFERENCE_MAX_TOKENS,
                "logprobs": True,
                "top_logprobs": 5,
            },
        )
        generations.append({"prompt": prompt, "response": response})
    out_path.write_text(json.dumps(generations, indent=1))


def run_benchmark(variant: str, shape: tuple[int, int], concurrency: int,
                  result_dir: Path, filename: str, bench_log: Path) -> None:
    input_len, output_len = shape
    command = [
        str(VENV_BIN / "vllm"), "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://127.0.0.1:{PORT}",
        "--endpoint", "/v1/completions",
        "--model", SERVED_NAME,
        "--tokenizer", str(MODELS / variant),
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", "0",
        "--ignore-eos",
        "--num-prompts", str(CONCURRENCY_PROMPTS[concurrency]),
        "--max-concurrency", str(concurrency),
        "--num-warmups", str(WARMUP_REQUESTS),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--seed", "0",
        "--save-result",
        "--result-dir", str(result_dir),
        "--result-filename", filename,
    ]
    with bench_log.open("a") as handle:
        handle.write(f"\n===== {filename}\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT,
                       cwd=ROOT, timeout=3600)


def vllm_versions() -> dict:
    output = subprocess.run(
        [str(VENV_BIN / "python"), "-c",
         "import json, torch, vllm; print(json.dumps({'vllm': vllm.__version__, "
         "'torch': torch.__version__, 'torch_cuda': torch.version.cuda}))"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    ).stdout
    return json.loads(output.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(MODEL_REVISIONS),
                        choices=list(MODEL_REVISIONS))
    parser.add_argument("--loops", nargs="+", type=int, default=list(LOOPS))
    parser.add_argument("--attention-backend", default=None,
                        help="vLLM attention backend name; vLLM chooses when omitted")
    parser.add_argument("--no-async-scheduling", action="store_true",
                        help="run the server without asynchronous scheduling, as vLLM does "
                             "for speculative methods that do not support it")
    parser.add_argument("--concurrency", nargs="+", type=int,
                        default=list(CONCURRENCY_PROMPTS), choices=list(CONCURRENCY_PROMPTS),
                        help="run only these concurrency levels")
    parser.add_argument("--shapes", nargs="+", default=[f"{i}/{o}" for i, o in SHAPES],
                        choices=[f"{i}/{o}" for i, o in SHAPES],
                        help="run only these input/output lengths")
    parser.add_argument("--wavefront", action="store_true",
                        help="Dory's wavefront decoding for a request decoding alone (needs "
                             "DORY_FUSED_DECODE and the patched vLLM checkout)")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "vllm_pr_sha": VLLM_PR_SHA,
        "model_revisions": MODEL_REVISIONS,
        "versions": vllm_versions(),
        "gpu": gpu_snapshot(),
        "shapes": SHAPES,
        "concurrency_prompts": CONCURRENCY_PROMPTS,
        "kv_cache_mode": "per_loop",
        "attention_backend": args.attention_backend,
        "server_env_overrides": SERVER_ENV,
        "cuda_home": os.environ.get("CUDA_HOME"),
        # "0" forces vLLM's V1 model runner; unset, vLLM picks V2 when it supports
        # the configuration (the server log then says "Using V2 Model Runner").
        "vllm_use_v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
        "async_scheduling": not args.no_async_scheduling,
        "wavefront_decode": args.wavefront,
        # Set when the patched vLLM runs single-token decode with the fused kernels.
        "dory_fused_decode": os.environ.get("DORY_FUSED_DECODE"),
        # The fused path's LM head for single-token logits; its default is INT8
        # (LM_HEAD_STORAGE in kernels/dory_fused_model.py).
        "dory_lm_head_8bit": (os.environ.get("DORY_LM_HEAD_8BIT", "int8")
                              if os.environ.get("DORY_FUSED_DECODE") else None),
        "configs": [],
    }
    log(f"run directory: {run_dir}")

    for variant, loops in itertools.product(args.variants, args.loops):
        try:
            require_idle_gpu()
        except GpuBusyError as error:
            manifest["aborted"] = str(error)
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
            raise
        config = {"variant": variant, "loops": loops, "status": "ok", "benchmarks": []}
        manifest["configs"].append(config)
        tag = f"{variant}_loops{loops}"
        server = start_server(variant, loops, args.attention_backend,
                              run_dir / f"server_{tag}.log",
                              async_scheduling=not args.no_async_scheduling,
                              extra_overrides={"wavefront_decode": True} if args.wavefront
                              else None)
        try:
            wait_until_ready(server)
            log(f"server ready: {tag}; capturing reference outputs")
            capture_reference(run_dir / f"reference_{tag}.json")
            for shape, concurrency in itertools.product(SHAPES, args.concurrency):
                if f"{shape[0]}/{shape[1]}" not in args.shapes:
                    continue
                filename = f"bench_{tag}_in{shape[0]}_out{shape[1]}_c{concurrency}.json"
                log(f"benchmark: {filename}")
                try:
                    run_benchmark(variant, shape, concurrency, run_dir, filename,
                                  run_dir / f"bench_{tag}.log")
                    config["benchmarks"].append(filename)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    config["status"] = "partial"
                    config.setdefault("errors", []).append(f"{filename}: {error}")
                    log(f"benchmark failed: {filename}: {error}")
        except (RuntimeError, urllib.error.URLError, TimeoutError) as error:
            config["status"] = "failed"
            config.setdefault("errors", []).append(str(error))
            log(f"config failed: {tag}: {error}")
        finally:
            stop_server(server)
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    failed = [c for c in manifest["configs"] if c["status"] != "ok"]
    log(f"done: {len(manifest['configs']) - len(failed)} ok, {len(failed)} not ok")
    print(run_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
