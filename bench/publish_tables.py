"""Fills the benchmark tables of README.md and docs/BENCHMARKS.md from the serving
runs in results/.

Each table sits between `<!-- table:NAME -->` and `<!-- /table -->` markers in a
document and is rebuilt from the run directories named here (written by
bench/run_baseline.py), so the published numbers are the measured ones. Values
are total output tokens per second; with one user that is the user's speed.

Usage: .venv/bin/python bench/publish_tables.py [--check]
"""

import argparse
from functools import cache
import math
import re
import sys
from pathlib import Path

from summarize import load_rows

ROOT = Path(__file__).resolve().parent.parent
DOCS = (ROOT / "README.md", ROOT / "docs" / "BENCHMARKS.md")
SHAPES = ((128, 1024), (2560, 300))
CONCURRENCY = (1, 16, 64)

# Serving runs (results/<run id>), by configuration.
DEFAULT_BF16 = "20261018T041933Z"  # FLASH_ATTN, vLLM's default backend
DEFAULT_MXFP8 = "20261018T035249Z"
DEFAULT_NVFP4 = "20261018T045319Z"
TRITON_NVFP4 = "20261018T114127Z"
TRITON_BF16_MXFP8 = "20261018T121113Z"
FLASHINFER_NVFP4 = "20261018T130843Z"
FUSED_FIRST = "20261018T163313Z"  # the first fused decode path
TRITON_2 = "20261018T180516Z"  # unmodified, rerun in the 2-loop session
FUSED_E5 = "20261018T181035Z"
FUSED_E9 = "20261018T185825Z"  # the finished fused path, BF16 LM head
FUSED_INT8_2 = "20261018T190327Z"
WAVEFRONT_2 = "20261019T022147Z"
# The corrected suite disables prefix caching for independent random prompts.
SUITE_USERS = (1, 2, 4, 8, 16, 32, 64)
SUITE_SHIPPED = "20261019T152606Z"
SUITE_TRITON = "20261019T232759Z"
SUITE_TRITON_SHARED = "20261019T160704Z"
SUITE_FUSED_BF16_HEAD = "20261020T002938Z"
SUITE_FUSED = "20261020T010402Z"
SUITE_FUSED_SHARED = "20261020T000323Z"
SUITE_WAVEFRONT = "20261019T143422Z"
SHARED_WAVEFRONT_RUNS = {2: "20261020T015236Z", 4: "20261019T150009Z"}

# From vLLM as shipped to this work, one feature per step; None where a step
# does not apply (the wavefront needs at least two loops, and with one loop
# there is nothing to share).
STEP_LABELS = ("vLLM as shipped", "+ Triton attention", "+ fused decode path",
               "+ INT8 LM head", "+ wavefront", "+ shared KV cache")
STEPS = {
    1: (SUITE_SHIPPED, SUITE_TRITON, SUITE_FUSED_BF16_HEAD, SUITE_FUSED, None, None),
    2: (SUITE_SHIPPED, SUITE_TRITON, SUITE_FUSED_BF16_HEAD, SUITE_FUSED,
        SUITE_WAVEFRONT, SHARED_WAVEFRONT_RUNS[2]),
    4: (SUITE_SHIPPED, SUITE_TRITON, SUITE_FUSED_BF16_HEAD, SUITE_FUSED,
        SUITE_WAVEFRONT, SHARED_WAVEFRONT_RUNS[4]),
}

# Tables of plain columns: name -> (columns, change). A column is (label, run,
# variant, loops); with change set, a last column gives the last column's
# change over the first.
COLUMN_TABLES = {
    **{f"backends-{loops}": ([("FLASH_ATTN (default)", DEFAULT_NVFP4, "nvfp4", loops),
                              ("TRITON_ATTN", TRITON_NVFP4, "nvfp4", loops),
                              ("FLASHINFER", FLASHINFER_NVFP4, "nvfp4", loops)], False)
       for loops in (1, 2, 4)},
    **{f"quants-{loops}": ([("BF16", TRITON_BF16_MXFP8, "bf16", loops),
                            ("MXFP8", TRITON_BF16_MXFP8, "mxfp8", loops),
                            ("NVFP4", TRITON_NVFP4, "nvfp4", loops)], True)
       for loops in (1, 2, 4)},
    **{f"quants-default-{loops}": ([("BF16", DEFAULT_BF16, "bf16", loops),
                                    ("MXFP8", DEFAULT_MXFP8, "mxfp8", loops),
                                    ("NVFP4", DEFAULT_NVFP4, "nvfp4", loops)], True)
       for loops in (1, 2, 4)},
    **{f"shared-kv-{loops}": (
        [("Unmodified, per-loop KV", SUITE_TRITON, "nvfp4", loops),
         ("This work, per-loop KV", SUITE_WAVEFRONT, "nvfp4", loops),
         ("Unmodified, shared KV", SUITE_TRITON_SHARED, "nvfp4", loops),
         ("Fused path, shared KV", SUITE_FUSED_SHARED, "nvfp4", loops),
         ("This work, shared KV", SHARED_WAVEFRONT_RUNS[loops], "nvfp4", loops)], True)
       for loops in (2, 4)},
    "features-2": ([("vLLM default", DEFAULT_NVFP4, "nvfp4", 2),
                    ("Triton attention", TRITON_2, "nvfp4", 2),
                    ("+ first fused decode path", FUSED_FIRST, "nvfp4", 2),
                    ("+ own attention kernel", FUSED_E5, "nvfp4", 2),
                    ("+ q/k norm in attention", FUSED_E9, "nvfp4", 2),
                    ("+ INT8 LM head", FUSED_INT8_2, "nvfp4", 2),
                    ("+ wavefront", WAVEFRONT_2, "nvfp4", 2)], False),
}


@cache
def output_rates(run: str) -> dict[tuple, float]:
    """(variant, loops, input, output, concurrency) -> output tokens/s."""
    return {(r["variant"], r["loops"], r["input_len"], r["output_len"], r["concurrency"]):
            r["output_tps"] for r in load_rows(ROOT / "results" / run)[1]}


def runs_line(runs) -> str:
    return f"Runs: {', '.join(sorted({run for run in runs if run}))}."


def column_table(columns: list[tuple], change: bool) -> str:
    rates = {run: output_rates(run) for _, run, _, _ in columns}
    header = ["Input/output", "Users"] + [label for label, *_ in columns]
    if change:
        header.append("Change")
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|" + "---:|" * (len(header) - 2)]
    for inp, out in SHAPES:
        for users in CONCURRENCY:
            values = [rates[run][(variant, loops, inp, out, users)]
                      for _, run, variant, loops in columns]
            cells = [f"{inp}/{out}", str(users)] + [f"{v:.1f}" for v in values]
            if change:
                cells.append(f"{(values[-1] / values[0] - 1) * 100:+.0f}%")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines + ["", runs_line(run for _, run, _, _ in columns)])


def step_cells(loops: int, inp: int, out: int, users: int) -> list[str]:
    """Each step's value with its change over the previous step that applies,
    then the total change."""
    cells, first, previous = [], None, None
    for run in STEPS[loops]:
        if run is None:
            cells.append("n/a")
            continue
        value = output_rates(run)[("nvfp4", loops, inp, out, users)]
        cells.append(f"{value:.1f}" if previous is None
                     else f"{value:.1f} ({(value / previous - 1) * 100:+.0f}%)")
        first = value if first is None else first
        previous = value
    return cells + [f"{(previous / first - 1) * 100:+.0f}%"]


def steps_table(loops: int) -> str:
    header = ["Input/output", "Users", *STEP_LABELS, "Total"]
    lines = ["| " + " | ".join(header) + " |", "|---|---:|" + "---:|" * (len(header) - 2)]
    for inp, out in SHAPES:
        for users in CONCURRENCY:
            lines.append("| " + " | ".join([f"{inp}/{out}", str(users),
                                            *step_cells(loops, inp, out, users)]) + " |")
    return "\n".join(lines + ["", runs_line(STEPS[loops])])


def headline_table() -> str:
    header = ["Loops", "Input/output", *STEP_LABELS, "Total"]
    lines = ["| " + " | ".join(header) + " |", "|---:|---|" + "---:|" * (len(header) - 2)]
    for loops in STEPS:
        for inp, out in SHAPES:
            lines.append("| " + " | ".join([str(loops), f"{inp}/{out}",
                                            *step_cells(loops, inp, out, 1)]) + " |")
    return "\n".join(lines)


def final_table() -> str:
    """vLLM as shipped and unmodified vLLM with Triton attention against this
    work (the last step that applies), for every loop count."""
    header = ["Loops", "Input/output", "Users", "vLLM as shipped",
              "Unmodified vLLM, Triton attention", "This work", "Over vLLM as shipped",
              "Over Triton attention"]
    lines = ["| " + " | ".join(header) + " |", "|---:|---|---:|" + "---:|" * 5]
    runs = []
    for loops, steps in STEPS.items():
        shipped, triton = steps[0], steps[1]
        best = [r for r in steps if r][-1]
        runs += [shipped, triton, best]
        for inp, out in SHAPES:
            for users in CONCURRENCY:
                key = ("nvfp4", loops, inp, out, users)
                a, b, c = (output_rates(run)[key] for run in (shipped, triton, best))
                lines.append("| " + " | ".join([
                    str(loops), f"{inp}/{out}", str(users), f"{a:.1f}", f"{b:.1f}",
                    f"{c:.1f}", f"{(c / a - 1) * 100:+.0f}%",
                    f"{(c / b - 1) * 100:+.0f}%"]) + " |")
    return "\n".join(lines + ["", runs_line(runs)])


SUITE_COLUMNS = (
    ("vLLM as shipped", {n: SUITE_SHIPPED for n in STEPS}),
    ("Triton attention", {n: SUITE_TRITON for n in STEPS}),
    ("Fused path", {n: SUITE_FUSED for n in STEPS}),
    ("Fused + wavefront", {n: SUITE_WAVEFRONT for n in (2, 4)}),
    ("Triton, shared KV", {n: SUITE_TRITON_SHARED for n in (2, 4)}),
    ("Fused, shared KV", {n: SUITE_FUSED_SHARED for n in (2, 4)}),
    ("Fused + wavefront, shared KV", SHARED_WAVEFRONT_RUNS),
)


def validate_suite() -> None:
    """Require complete, comparable measurements before publishing the suite."""
    configurations = (
        (SUITE_SHIPPED, (1, 2, 4), None, "per_loop", False, None, False, SHAPES),
        (SUITE_TRITON, (1, 2, 4), "TRITON_ATTN", "per_loop", False, None, False, SHAPES),
        (SUITE_TRITON_SHARED, (2, 4), "TRITON_ATTN", "last_loop", False, None, False, SHAPES),
        (SUITE_FUSED_BF16_HEAD, (1, 2, 4), "TRITON_ATTN", "per_loop", True, "none", False,
         SHAPES),
        (SUITE_FUSED, (1, 2, 4), "TRITON_ATTN", "per_loop", True, "int8", False, SHAPES),
        (SUITE_FUSED_SHARED, (2, 4), "TRITON_ATTN", "last_loop", True, "int8", False, SHAPES),
        (SUITE_WAVEFRONT, (2, 4), "TRITON_ATTN", "per_loop", True, "int8", True, SHAPES),
        (SHARED_WAVEFRONT_RUNS[2], (2,), "TRITON_ATTN", "last_loop", True, "int8", True,
         SHAPES),
        (SHARED_WAVEFRONT_RUNS[4], (4,), "TRITON_ATTN", "last_loop", True, "int8", True,
         SHAPES),
    )
    reference = None
    for run, loops, backend, mode, fused, head, wavefront, shapes in configurations:
        manifest, rows = load_rows(ROOT / "results" / run)
        expected_settings = {
            "prefix_caching": False, "attention_backend": backend,
            "kv_cache_mode": mode, "wavefront_decode": wavefront,
            "dory_lm_head_8bit": head, "async_scheduling": True,
        }
        if (manifest.get("aborted") or
                bool(manifest.get("dory_fused_decode")) != fused or
                any(manifest.get(k) != v for k, v in expected_settings.items())):
            raise ValueError(f"{run}: unexpected benchmark settings")
        provenance = (manifest["versions"], manifest["vllm_pr_sha"],
                      manifest["model_revisions"], manifest["gpu"]["name"],
                      manifest["gpu"]["driver"], manifest["server_env_overrides"],
                      manifest.get("vllm_use_v2_model_runner"))
        if reference is None:
            reference = provenance
        elif provenance != reference:
            raise ValueError(f"{run}: software, model, GPU or sampler differs")
        configs = manifest["configs"]
        expected_configs = {("nvfp4", n) for n in loops}
        if (len(configs) != len(expected_configs) or
                {(c["variant"], c["loops"]) for c in configs} != expected_configs or
                any(c["status"] != "ok" for c in configs)):
            raise ValueError(f"{run}: incomplete configurations")
        expected_keys = {("nvfp4", n, i, o, u)
                         for n in loops for i, o in shapes for u in SUITE_USERS}
        keys = {(r["variant"], r["loops"], r["input_len"], r["output_len"],
                 r["concurrency"]) for r in rows}
        if keys != expected_keys or len(rows) != len(expected_keys):
            raise ValueError(f"{run}: missing or duplicate benchmark rows")
        for row in rows:
            expected_requests = {1: 8, 2: 8, 4: 16, 8: 32, 16: 64, 32: 128, 64: 128}[
                row["concurrency"]]
            if (row["failed"] != 0 or row["completed"] != expected_requests or
                    not math.isfinite(row["output_tps"]) or row["output_tps"] <= 0):
                raise ValueError(f"{run}: failed or invalid benchmark row")


def suite_table(loops: int) -> str:
    """Every configuration of the complete suite at one loop count; columns that
    do not apply (a shared cache or the wavefront at 1 loop) are n/a. "Best" is
    the fastest configuration of this work against vLLM as shipped."""
    columns = [(label, runs.get(loops)) for label, runs in SUITE_COLUMNS]
    rates = {run: output_rates(run) for _, run in columns if run}
    header = ["Input/output", "Users", *(label for label, _ in columns), "Best over shipped"]
    lines = ["| " + " | ".join(header) + " |", "|---|---:|" + "---:|" * (len(header) - 2)]
    for inp, out in SHAPES:
        for users in SUITE_USERS:
            key = ("nvfp4", loops, inp, out, users)
            values = [rates[run][key] if run else None for _, run in columns]
            ours = [v for (label, _), v in zip(columns, values)
                    if v is not None and label not in ("vLLM as shipped", "Triton attention",
                                                       "Triton, shared KV")]
            cells = [f"{inp}/{out}", str(users)] + [
                f"{v:.1f}" if v is not None else "n/a" for v in values]
            cells.append(f"{(max(ours) / values[0] - 1) * 100:+.0f}%")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines + ["", runs_line(run for _, run in columns)])


def build(name: str) -> str:
    if name in COLUMN_TABLES:
        return column_table(*COLUMN_TABLES[name])
    if name.startswith("steps-") and int(name.removeprefix("steps-")) in STEPS:
        return steps_table(int(name.removeprefix("steps-")))
    if name == "headline":
        return headline_table()
    if name == "final":
        return final_table()
    if name.startswith("suite-") and int(name.removeprefix("suite-")) in STEPS:
        return suite_table(int(name.removeprefix("suite-")))
    raise SystemExit(f"unknown table {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if a document's tables differ from the runs")
    args = parser.parse_args()
    validate_suite()
    pattern = re.compile(r"(<!-- table:([\w-]+) -->\n).*?(<!-- /table -->)", re.S)
    stale = []
    for doc in DOCS:
        text = doc.read_text()
        updated = pattern.sub(
            lambda m: m.group(1) + build(m.group(2)) + "\n" + m.group(3), text)
        if updated == text:
            continue
        if args.check:
            stale.append(str(doc.relative_to(ROOT)))
        else:
            doc.write_text(updated)
            print(f"{doc.relative_to(ROOT)}: updated")
    if stale:
        print(f"out of date, run bench/publish_tables.py: {stale}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
