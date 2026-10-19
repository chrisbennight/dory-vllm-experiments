"""Measures single-user speed with and without vLLM's prompt-lookup speculation.

Prompt-lookup (n-gram) speculation proposes draft tokens by copying text that
follows an earlier occurrence of the latest tokens, and verifies them in one
multi-token step. It costs no draft model, and pays off when outputs repeat
their context. Two greedy single-user workloads, one request at a time:

- edit: rewrite a real function from this repository (comments removed,
  docstring kept), thinking disabled; the output mostly copies the input;
- chat: the first 16 prompts of bench/compare_outputs.py with thinking on, as in
  normal chat use.

For each configuration and workload the report gives output tokens per second,
vLLM's speculation counters (draft tokens proposed and accepted) and whether
the greedy outputs match the first configuration's.

Usage: .venv/bin/python -I bench/spec_ngram.py nvfp4-triton-fused nvfp4-triton-fused-ngram
"""

import argparse
import ast
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Run with -I, which leaves this directory off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_outputs as co  # noqa: E402
import run_baseline as rb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EDIT_SOURCES = ("bench/compare_outputs.py", "bench/gap_analysis.py", "bench/run_baseline.py",
                "kernels/dory_fused_model.py", "kernels/dory_decode_attention.py")
EDIT_COUNT, CHAT_COUNT = 8, 16
MAX_TOKENS = 512


def edit_prompts() -> list[str]:
    """Functions of 20 to 60 lines from this repository, in a fixed order."""
    functions = []
    for relative in EDIT_SOURCES:
        source = (ROOT / relative).read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and 20 <= node.end_lineno - node.lineno + 1 <= 60:
                functions.append(ast.get_source_segment(source, node))
    if len(functions) < EDIT_COUNT:
        raise ValueError("not enough functions to edit")
    return [("Return the following Python function with every '#' comment removed. Keep "
             "everything else exactly as it is, and return only the code.\n\n```python\n"
             f"{code}\n```") for code in functions[:EDIT_COUNT]]


def spec_counters() -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{rb.PORT}/metrics", timeout=30) as response:
        text = response.read().decode()
    counters = {}
    for line in text.splitlines():
        match = re.match(r"(vllm:spec_decode_num_\w+?)(?:_total)?(?:\{[^}]*\})? ([0-9.eE+-]+)$",
                         line)
        if match:
            counters[match.group(1)] = counters.get(match.group(1), 0.0) + float(match.group(2))
    return counters


def run_workload(prompts: list[str], thinking: bool) -> dict:
    before = spec_counters()
    texts, tokens, start = [], 0, time.monotonic()
    for prompt in prompts:
        response = rb.http_json(f"http://127.0.0.1:{rb.PORT}/v1/chat/completions", {
            "model": rb.SERVED_NAME, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": thinking}})
        texts.append(response["choices"][0]["message"]["content"] or "")
        tokens += response["usage"]["completion_tokens"]
    seconds = time.monotonic() - start
    after = spec_counters()
    return {"texts": texts, "tokens": tokens, "seconds": seconds,
            "counters": {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", choices=list(co.CONFIGS))
    parser.add_argument("--loops", type=int, default=2)
    args = parser.parse_args()
    workloads = {"edit": (edit_prompts(), False), "chat": (list(co.PROMPTS[:CHAT_COUNT]), True)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / f"spec_ngram_{stamp}"
    out_dir.mkdir(parents=True)

    results = {}
    for name in args.configs:
        with co.serving(name, args.loops, out_dir / f"server_{name}.log"):
            # One short request first, so that compilation and warm-up are not timed.
            run_workload(["Say hello."], False)
            results[name] = {workload: run_workload(prompts, thinking)
                             for workload, (prompts, thinking) in workloads.items()}
            for workload, result in results[name].items():
                print(f"{name} {workload}: {result['tokens'] / result['seconds']:.1f} tok/s",
                      flush=True)
        (out_dir / f"outputs_{name}.json").write_text(json.dumps(results[name]))

    reference = args.configs[0]
    lines = [f"# Prompt-lookup speculation at one user ({args.loops} loops, greedy, up to "
             f"{MAX_TOKENS} tokens)", "",
             f"edit: {EDIT_COUNT} functions from this repository to copy without comments "
             f"(thinking off); chat: {CHAT_COUNT} chat prompts (thinking on).", "",
             "| Configuration | Workload | Output tokens | Seconds | Tokens/s | Speed vs "
             f"{reference} | Draft tokens | Accepted | Acceptance | Outputs equal to "
             f"{reference} |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in args.configs:
        for workload in workloads:
            r, ref = results[name][workload], results[reference][workload]
            rate = r["tokens"] / r["seconds"]
            drafts = r["counters"].get("vllm:spec_decode_num_draft_tokens", 0.0)
            accepted = r["counters"].get("vllm:spec_decode_num_accepted_tokens", 0.0)
            equal = sum(a == b for a, b in zip(r["texts"], ref["texts"]))
            lines.append(f"| {name} | {workload} | {r['tokens']} | {r['seconds']:.1f} "
                         f"| {rate:.1f} | {rate / (ref['tokens'] / ref['seconds']):.2f}x "
                         f"| {drafts:.0f} | {accepted:.0f} "
                         f"| {accepted / drafts:.0%} | {equal}/{len(r['texts'])} |"
                         if drafts else
                         f"| {name} | {workload} | {r['tokens']} | {r['seconds']:.1f} "
                         f"| {rate:.1f} | {rate / (ref['tokens'] / ref['seconds']):.2f}x "
                         f"| 0 | 0 | n/a | {equal}/{len(r['texts'])} |")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report)
    print(report)
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
