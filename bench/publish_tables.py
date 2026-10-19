"""Fills the benchmark tables of README.md and docs/BENCHMARKS.md from the serving
runs in results/.

Each table sits between `<!-- table:NAME -->` and `<!-- /table -->` markers in a
document and is rebuilt from the run directories named here (written by
bench/run_baseline.py), so the published numbers are the measured ones. Values
are total output tokens per second; with one user that is the user's speed.

Usage: .venv/bin/python bench/publish_tables.py [--check]
"""

import argparse
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
TRITON_4 = "20261019T003224Z"  # unmodified, rerun in the 4-loop session
FUSED_BF16_HEAD_1_4 = "20261019T023002Z"  # the finished fused path, BF16 LM head
FUSED_INT8_1 = "20261019T024248Z"
FUSED_INT8_4 = "20261019T004110Z"
WAVEFRONT_4 = "20261019T013129Z"
# Dory's shared KV cache ("last_loop"), 4 loops.
UNMODIFIED_SHARED_4 = "20261019T111134Z"
FUSED_SHARED_4 = "20261019T122346Z"
WAVEFRONT_SHARED_4 = "20261019T123105Z"

# A step whose run has not been made yet.
PENDING = "pending"

# From vLLM as shipped to this work, one feature per step; None where a step
# does not apply (the wavefront needs at least two loops, and with one loop
# there is nothing to share).
STEP_LABELS = ("vLLM as shipped", "+ Triton attention", "+ fused decode path",
               "+ INT8 LM head", "+ wavefront", "+ shared KV cache")
STEPS = {
    1: (DEFAULT_NVFP4, TRITON_NVFP4, FUSED_BF16_HEAD_1_4, FUSED_INT8_1, None, None),
    2: (DEFAULT_NVFP4, TRITON_2, FUSED_E9, FUSED_INT8_2, WAVEFRONT_2, PENDING),
    4: (DEFAULT_NVFP4, TRITON_4, FUSED_BF16_HEAD_1_4, FUSED_INT8_4, WAVEFRONT_4,
        WAVEFRONT_SHARED_4),
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
    "shared-kv-4": ([("Unmodified, per-loop KV", TRITON_4, "nvfp4", 4),
                     ("This work, per-loop KV", WAVEFRONT_4, "nvfp4", 4),
                     ("Unmodified, shared KV", UNMODIFIED_SHARED_4, "nvfp4", 4),
                     ("Fused path, shared KV", FUSED_SHARED_4, "nvfp4", 4),
                     ("This work, shared KV", WAVEFRONT_SHARED_4, "nvfp4", 4)], True),
    "features-2": ([("vLLM default", DEFAULT_NVFP4, "nvfp4", 2),
                    ("Triton attention", TRITON_2, "nvfp4", 2),
                    ("+ first fused decode path", FUSED_FIRST, "nvfp4", 2),
                    ("+ own attention kernel", FUSED_E5, "nvfp4", 2),
                    ("+ q/k norm in attention", FUSED_E9, "nvfp4", 2),
                    ("+ INT8 LM head", FUSED_INT8_2, "nvfp4", 2),
                    ("+ wavefront", WAVEFRONT_2, "nvfp4", 2)], False),
}


def output_rates(run: str) -> dict[tuple, float]:
    """(variant, loops, input, output, concurrency) -> output tokens/s."""
    return {(r["variant"], r["loops"], r["input_len"], r["output_len"], r["concurrency"]):
            r["output_tps"] for r in load_rows(ROOT / "results" / run)[1]}


def runs_line(runs) -> str:
    return f"Runs: {', '.join(sorted({run for run in runs if run and run != PENDING}))}."


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
        if run is None or run == PENDING:
            cells.append("n/a" if run is None else PENDING)
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
        best = [r for r in steps if r and r != PENDING][-1]
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


# The complete suite: every configuration at 1 to 64 users, run in one session.
SUITE_USERS = (1, 2, 4, 8, 16, 32, 64)
SUITE_SHIPPED = "PENDING_SHIPPED"  # vLLM as shipped (FLASH_ATTN), per-loop KV
SUITE_TRITON = "PENDING_TRITON"
SUITE_TRITON_SHARED = "PENDING_TRITON_SHARED"
SUITE_FUSED = "PENDING_FUSED"  # fused path, INT8 LM head
SUITE_FUSED_SHARED = "PENDING_FUSED_SHARED"
SUITE_WAVEFRONT = "PENDING_WAVEFRONT"
SUITE_WAVEFRONT_SHARED = "PENDING_WAVEFRONT_SHARED"
SUITE_COLUMNS = (("vLLM as shipped", SUITE_SHIPPED, False, False),
                 ("Triton attention", SUITE_TRITON, False, False),
                 ("Fused path", SUITE_FUSED, False, False),
                 ("Fused + wavefront", SUITE_WAVEFRONT, False, True),
                 ("Triton, shared KV", SUITE_TRITON_SHARED, True, False),
                 ("Fused, shared KV", SUITE_FUSED_SHARED, True, False),
                 ("Fused + wavefront, shared KV", SUITE_WAVEFRONT_SHARED, True, True))


def suite_table(loops: int) -> str:
    """Every configuration of the complete suite at one loop count; columns that
    do not apply (a shared cache or the wavefront at 1 loop) are n/a. "Best" is
    the fastest configuration of this work against vLLM as shipped."""
    columns = [(label, run if loops > 1 or not (shared or wavefront) else None)
               for label, run, shared, wavefront in SUITE_COLUMNS]
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
