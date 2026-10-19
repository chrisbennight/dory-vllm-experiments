"""Compares greedy reference generations of quantized checkpoints against BF16.

For each loop count and prompt, reports how many leading tokens a quantized
checkpoint's greedy output shares with the BF16 output. An early divergence
is a coarse signal of quantization error; near-tied tokens can also diverge.

Usage: python bench/compare_reference.py results/<run id> [results/<run id> ...]
"""

import json
import re
import sys
from pathlib import Path

REFERENCE_NAME = re.compile(r"reference_(?P<variant>[a-z0-9]+)_loops(?P<loops>\d+)\.json")


def tokens_of(entry: dict) -> list[str]:
    return [t["token"] for t in entry["response"]["choices"][0]["logprobs"]["content"]]


def shared_prefix(a: list[str], b: list[str]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def main() -> int:
    # Later run directories override earlier ones for the same configuration.
    references: dict[tuple[str, int], list[dict]] = {}
    for run_dir in sys.argv[1:]:
        for path in sorted(Path(run_dir).glob("reference_*.json")):
            match = REFERENCE_NAME.fullmatch(path.name)
            key = (match["variant"], int(match["loops"]))
            references[key] = json.loads(path.read_text())

    print("| Loops | Weights | Shared greedy tokens with BF16, per prompt | Identical outputs |")
    print("|---:|---|---|---:|")
    for (variant, loops), entries in sorted(references.items(), key=lambda kv: kv[0][::-1]):
        baseline = references.get(("bf16", loops))
        if variant == "bf16" or baseline is None:
            continue
        prefixes, identical = [], 0
        for entry, base in zip(entries, baseline, strict=True):
            mine, theirs = tokens_of(entry), tokens_of(base)
            prefix = shared_prefix(mine, theirs)
            prefixes.append(f"{prefix}/{len(theirs)}")
            identical += mine == theirs
        print(f"| {loops} | {variant} | {', '.join(prefixes)} | {identical}/{len(entries)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
