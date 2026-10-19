"""Compares log-probability differences over token positions shared by every configuration.

compare_outputs.py averages each configuration's log-probability difference
over its own shared prefix with the reference, so a configuration that agrees
for longer also averages over later tokens, where differences grow. This
script restricts every configuration to the positions that all of them share
with the reference (per prompt, the shortest shared prefix), which makes the
means comparable.

Usage: .venv/bin/python -I bench/common_prefix_delta.py REFERENCE.json NAME=OUTPUTS.json ...
(outputs_<configuration>.json files written by compare_outputs.py)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def shared_prefix(a: list[dict], b: list[dict]) -> int:
    limit = min(len(a), len(b))
    return next((i for i in range(limit) if a[i]["token"] != b[i]["token"]), limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", help="NAME=path/to/outputs.json")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidates = {}
    for item in args.candidates:
        name, _, path = item.partition("=")
        candidates[name] = json.loads(Path(path).read_text())

    deltas = {name: [] for name in candidates}
    for prompt, ref in enumerate(reference):
        tokens = ref["tokens"]
        common = min(shared_prefix(tokens, c[prompt]["tokens"]) for c in candidates.values())
        for name, outputs in candidates.items():
            other = outputs[prompt]["tokens"]
            deltas[name] += [abs(tokens[i]["logprob"] - other[i]["logprob"])
                             for i in range(common)]
    count = len(next(iter(deltas.values())))
    print(f"| Configuration | Mean abs logprob difference over {count} common positions "
          "| Largest |")
    print("|---|---:|---:|")
    for name, values in deltas.items():
        print(f"| {name} | {statistics.mean(values):.4f} | {max(values):.4f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
