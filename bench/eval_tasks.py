"""Measures task accuracy of server configurations on GSM8K and MMLU samples.

Checks whether a change to the fused decode path (such as the INT8 LM head)
changes task quality. Requests run one at a time, because the fused path and
its LM head only apply when a step computes logits for a single token. Thinking
is disabled through the chat template, so answers come directly.

- GSM8K (bench/fetch_eval_data.py sample): greedy chat completion asking for a
  final line "Answer: <number>"; exact match of the number.
- MMLU (sample spread over subjects): a one-token chat completion with the
  top-20 log-probabilities; the prediction is the most likely of A, B, C, D.
  Prefill runs vLLM's own kernels in every configuration, so MMLU differences
  between fused configurations come from the LM head alone.

The report compares every configuration with the first one on the same items:
accuracy, items each got right that the other got wrong, the exact McNemar
p-value of that split, and for MMLU how often the chosen letter agrees and how
much its log-probability moves.

Usage: .venv/bin/python -I bench/eval_tasks.py nvfp4-triton-fused nvfp4-triton-fused-int8-head
"""

import argparse
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Run with -I, which leaves this directory off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_outputs as co  # noqa: E402
import run_baseline as rb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ABCD"
GSM8K_MAX_TOKENS = 768
NUMBER = r"-?\$?\s*[\d,]*\.?\d+"


def chat(messages: list[dict], **params) -> dict:
    return rb.http_json(f"http://127.0.0.1:{rb.PORT}/v1/chat/completions", {
        "model": rb.SERVED_NAME, "messages": messages, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}, **params})


def parse_number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def gsm8k_answer(text: str) -> float | None:
    marked = re.findall(r"Answer:\s*(" + NUMBER + ")", text)
    found = marked or re.findall(NUMBER, text)
    return parse_number(found[-1]) if found else None


def run_gsm8k(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        prompt = (f"{item['question']}\n\nSolve the problem step by step. End with a final "
                  "line of the form 'Answer: <number>'.")
        response = chat([{"role": "user", "content": prompt}], max_tokens=GSM8K_MAX_TOKENS)
        text = response["choices"][0]["message"]["content"] or ""
        expected = parse_number(item["answer"].split("####")[-1])
        predicted = gsm8k_answer(text)
        results.append({"expected": expected, "predicted": predicted,
                        "correct": predicted is not None and expected is not None
                        and abs(predicted - expected) < 1e-6,
                        "completion_tokens": response["usage"]["completion_tokens"]})
    return results


def run_mmlu(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"]))
        prompt = (f"The following is a multiple-choice question about "
                  f"{item['subject'].replace('_', ' ')}.\n\n{item['question']}\n\n{options}\n\n"
                  "Answer with the letter of the correct option only.")
        response = chat([{"role": "user", "content": prompt}], max_tokens=1, logprobs=True,
                        top_logprobs=20)
        top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        letters = {}
        for entry in top:
            letter = entry["token"].strip()
            if letter in LETTERS and letter not in letters:
                letters[letter] = entry["logprob"]
        predicted = max(letters, key=letters.get) if letters else None
        results.append({"expected": LETTERS[item["answer"]], "predicted": predicted,
                        "correct": predicted == LETTERS[item["answer"]],
                        "letter_logprobs": letters})
    return results


def mcnemar_p(first_only: int, second_only: int) -> float:
    """Exact two-sided McNemar p-value for the discordant counts."""
    n = first_only + second_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(first_only, second_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", choices=list(co.CONFIGS))
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--limit", type=int, help="items per task, for a quick check")
    args = parser.parse_args()
    data = {name: json.loads((ROOT / "data" / name / f"{name}_sample.json").read_text())
            [:args.limit] for name in ("gsm8k", "mmlu")}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / f"task_eval_{stamp}"
    out_dir.mkdir(parents=True)

    outputs, seconds = {}, {}
    for name in args.configs:
        with co.serving(name, args.loops, out_dir / f"server_{name}.log"):
            outputs[name] = {}
            for task, run in (("mmlu", run_mmlu), ("gsm8k", run_gsm8k)):
                start = time.monotonic()
                outputs[name][task] = run(data[task])
                seconds[(name, task)] = time.monotonic() - start
                accuracy = statistics.mean(r["correct"] for r in outputs[name][task])
                print(f"{name} {task}: {accuracy:.1%} in {seconds[(name, task)]:.0f} s",
                      flush=True)
        (out_dir / f"outputs_{name}.json").write_text(json.dumps(outputs[name]))

    reference = args.configs[0]
    lines = [f"# Task accuracy ({args.loops} loops, thinking disabled, one request at a time)",
             "", f"GSM8K: {len(data['gsm8k'])} problems, greedy, up to {GSM8K_MAX_TOKENS} tokens. "
             f"MMLU: {len(data['mmlu'])} questions over "
             f"{len({r['subject'] for r in data['mmlu']})} subjects, best answer letter by "
             "log-probability.", "",
             "| Configuration | Task | Accuracy | Right only here | Right only in "
             f"{reference} | McNemar p | Seconds |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name in args.configs:
        for task in ("gsm8k", "mmlu"):
            mine, theirs = outputs[name][task], outputs[reference][task]
            accuracy = statistics.mean(r["correct"] for r in mine)
            only_here = sum(m["correct"] and not t["correct"] for m, t in zip(mine, theirs))
            only_ref = sum(t["correct"] and not m["correct"] for m, t in zip(mine, theirs))
            p = "" if name == reference else f"{mcnemar_p(only_here, only_ref):.2f}"
            lines.append(f"| {name} | {task} | {accuracy:.1%} | {only_here} | {only_ref} | {p} "
                         f"| {seconds[(name, task)]:.0f} |")
    lines += ["", f"## MMLU answer letters compared with {reference}", "",
              "| Configuration | Same letter | Mean abs logprob change of the reference's letter "
              "| Largest change |", "|---|---:|---:|---:|"]
    for name in args.configs[1:]:
        pairs = list(zip(outputs[name]["mmlu"], outputs[reference]["mmlu"]))
        same = statistics.mean(m["predicted"] == t["predicted"] for m, t in pairs)
        changes = [abs(m["letter_logprobs"][t["predicted"]] - t["letter_logprobs"][t["predicted"]])
                   for m, t in pairs
                   if t["predicted"] is not None and t["predicted"] in m["letter_logprobs"]]
        lines.append(f"| {name} | {same:.1%} | {statistics.mean(changes):.4f} "
                     f"| {max(changes):.4f} |")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report)
    print(report)
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
