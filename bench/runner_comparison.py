"""Compares vLLM's V2 and V1 model runners for Dory with the fused decode path.

Serving the wavefront through vLLM's custom speculative-proposer hook would
move every request to the V1 model runner, and vLLM turns off asynchronous
scheduling for that speculative method. This runs, one job at a time, in
three configurations (V2, the default; V1; V1 without asynchronous
scheduling), all NVFP4, Triton attention, fused decode with its default LM
head:

- bench/profile_decode.py at 2 and 4 loops (640-token prompt): single-user
  decode step time;
- bench/run_baseline.py at 2 loops: serving at 1, 16 and 64 users.

Each step must start on an idle GPU and leave it idle, so a game or another
job that starts mid-run stops the comparison instead of contaminating it.
Each step's log must show the runner it was asked for. The report goes to
results/runner_comparison_<time>/report.md.

Usage: .venv/bin/python bench/runner_comparison.py [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_baseline import ROOT, VENV_BIN, gpu_busy_reason, gpu_snapshot
from summarize import load_rows

PYTHON = str(VENV_BIN / "python")
PROFILE = [PYTHON, "bench/profile_decode.py", "--variant", "nvfp4", "--prompt-len", "640",
           "--attention-backend", "TRITON_ATTN"]
SERVING = [PYTHON, "bench/run_baseline.py", "--variants", "nvfp4", "--loops", "2",
           "--attention-backend", "TRITON_ATTN"]
# Name, VLLM_USE_V2_MODEL_RUNNER (None leaves vLLM's default, V2), extra arguments.
CONFIGS = (("V2", None, []),
           ("V1", "0", []),
           ("V1, no async scheduling", "0", ["--no-async-scheduling"]))
V2_LOG_LINE = "Using V2 Model Runner"
# A finished step's server can take a moment to release its memory.
IDLE_GRACE_S = 90


def wait_for_idle_gpu(grace_s: float) -> str | None:
    """None once the GPU is idle, else why it is still busy after grace_s."""
    deadline = time.monotonic() + grace_s
    while (reason := gpu_busy_reason()) is not None:
        if time.monotonic() >= deadline:
            return reason
        time.sleep(5)
    return None


def steps() -> list[dict]:
    plan = [{"kind": "profile", "loops": loops, "config": name, "runner": runner,
             "argv": PROFILE + ["--loops", str(loops)] + extra}
            for loops in (2, 4) for name, runner, extra in CONFIGS]
    plan += [{"kind": "serving", "loops": 2, "config": name, "runner": runner,
              "argv": SERVING + extra} for name, runner, extra in CONFIGS]
    return plan


def run_step(step: dict, log_path: Path) -> tuple[int, Path | None]:
    """Runs one step; returns its exit status and the results directory it made."""
    env = {**os.environ, "DORY_FUSED_DECODE": str(ROOT / "kernels")}
    env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    if step["runner"] is not None:
        env["VLLM_USE_V2_MODEL_RUNNER"] = step["runner"]
    before = set((ROOT / "results").iterdir())
    with log_path.open("w") as log:
        status = subprocess.run(step["argv"], cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    made = [path for path in set((ROOT / "results").iterdir()) - before if path.is_dir()]
    return status, (made[0] if len(made) == 1 else None)


def runner_matches(step: dict, log_path: Path, out_dir: Path) -> bool:
    if step["kind"] == "serving":
        log_path = out_dir / "server_nvfp4_loops2.log"
    uses_v2 = V2_LOG_LINE in log_path.read_text(errors="replace")
    return uses_v2 == (step["runner"] is None)


def change(value: float, reference: float) -> str:
    return f"{(value / reference - 1) * 100:+.1f}%"


def report(done: list[dict], problems: list[str]) -> str:
    names = [name for name, _, _ in CONFIGS]
    lines = ["# vLLM V2 versus V1 model runner, fused decode path "
             f"({gpu_snapshot()['name']})", "",
             "NVFP4, Triton attention, fused decode with its default LM head, run one at "
             "a time on an idle GPU. Changes are relative to V2.", "",
             "## Single-user decode step (bench/profile_decode.py, 640-token prompt)", "",
             "| Loops | Runner | Step ms | tok/s | Speed change | Forward busy ms "
             "| Idle us per step | Profile |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    for loops in (2, 4):
        rows = {s["config"]: s for s in done if s["kind"] == "profile" and s["loops"] == loops}
        if "V2" not in rows:
            continue
        reference = rows["V2"]["summary"]["period_us"]
        for name in names:
            if name not in rows:
                continue
            summary = rows[name]["summary"]
            lines.append(
                f"| {loops} | {name} | {summary['period_us'] / 1000:.2f} "
                f"| {1e6 / summary['period_us']:.1f} | {change(reference, summary['period_us'])} "
                f"| {summary['forward_busy_us'] / 1000:.2f} | {summary['idle_us_per_step']:.0f} "
                f"| {rows[name]['out_dir'].name} |")
    serving = {s["config"]: s for s in done if s["kind"] == "serving"}
    if "V2" in serving:
        table = {name: {(r["input_len"], r["output_len"], r["concurrency"]): r
                        for r in load_rows(serving[name]["out_dir"])[1]}
                 for name in names if name in serving}
        columns = [name for name in names if name in table]
        lines += ["", "## Serving at 2 loops (bench/run_baseline.py), output tok/s", "",
                  "Runs: " + ", ".join(f"{name} {serving[name]['out_dir'].name}"
                                       for name in columns) + ".", "",
                  "| Input/output | Users | " + " | ".join(columns) + " | "
                  + " | ".join(f"{name} change" for name in columns[1:]) + " |",
                  "|---|---:|" + "---:|" * (2 * len(columns) - 1)]
        for key in sorted(table["V2"], key=lambda k: (k[0], k[2])):
            values = [table[name][key]["output_tps"] if key in table[name] else None
                      for name in columns]
            lines.append(
                f"| {key[0]}/{key[1]} | {key[2]} | "
                + " | ".join(f"{v:.1f}" if v is not None else "n/a" for v in values) + " | "
                + " | ".join(change(v, values[0]) if v is not None else "n/a"
                             for v in values[1:]) + " |")
    if problems:
        lines += ["", "## Problems", ""] + [f"- {problem}" for problem in problems]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the GPU state without running anything")
    args = parser.parse_args()
    plan = steps()
    if args.dry_run:
        for index, step in enumerate(plan, 1):
            runner = step["runner"] if step["runner"] is not None else "unset"
            print(f"{index}. {step['kind']}, {step['loops']} loops, {step['config']}: "
                  f"VLLM_USE_V2_MODEL_RUNNER={runner} {' '.join(step['argv'][1:])}")
        print(f"GPU now: {gpu_busy_reason() or 'idle'}")
        return 0

    if (reason := gpu_busy_reason()) is not None:
        print(f"GPU is not idle ({reason}); refusing to start", file=sys.stderr)
        return 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / f"runner_comparison_{stamp}"
    out_dir.mkdir()
    done, problems = [], []
    for index, step in enumerate(plan, 1):
        label = f"{step['kind']}, {step['loops']} loops, {step['config']}"
        if (reason := wait_for_idle_gpu(IDLE_GRACE_S)) is not None:
            problems.append(f"stopped before step {index} ({label}): GPU not idle ({reason})")
            break
        print(f"[{datetime.now(timezone.utc):%H:%M:%SZ}] step {index}/{len(plan)}: {label}",
              flush=True)
        log_path = out_dir / f"step{index:02d}.log"
        status, step_dir = run_step(step, log_path)
        if status != 0 or step_dir is None:
            problems.append(f"step {index} ({label}) failed: exit status {status}, "
                            f"results directory {step_dir}; see {log_path.name}")
            break
        if not runner_matches(step, log_path, step_dir):
            problems.append(f"step {index} ({label}) did not use the requested runner")
            break
        # A job that started during the step shows up once ours has exited.
        if (reason := wait_for_idle_gpu(IDLE_GRACE_S)) is not None:
            problems.append(f"step {index} ({label}) discarded: the GPU was busy after it "
                            f"ended ({reason}), so another job may have overlapped it")
            break
        step["out_dir"] = step_dir
        if step["kind"] == "profile":
            step["summary"] = json.loads((step_dir / "summary.json").read_text())
        done.append(step)
    text = report(done, problems)
    (out_dir / "report.md").write_text(text)
    print(text)
    print(out_dir / "report.md")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
