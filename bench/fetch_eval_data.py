"""Downloads the task-evaluation samples used by bench/eval_tasks.py.

Fetches rows as JSON from the Hugging Face dataset viewer API, so no Parquet
reader is needed:

- GSM8K (openai/gsm8k, main, test): the first 250 problems;
- MMLU (cais/mmlu, all, test): 1000 questions, 20 consecutive rows at 50 evenly
  spaced offsets, which covers every subject (the split is sorted by subject).

Each dataset is written to its own directory under data/ (ignored by Git). The
files are data only and are parsed as JSON.

Usage: .venv/bin/python -I bench/fetch_eval_data.py
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://datasets-server.huggingface.co/rows"
GSM8K_COUNT = 250
MMLU_TOTAL, MMLU_GROUPS, MMLU_GROUP_SIZE = 14042, 50, 20


def rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": offset, "length": length})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
                payload = json.load(response)
            return [item["row"] for item in payload["rows"]]
        except (OSError, KeyError, json.JSONDecodeError) as error:
            if attempt == 3:
                raise
            print(f"retrying {dataset} offset {offset}: {error}", file=sys.stderr)
            time.sleep(2 + 2 * attempt)
    raise AssertionError("unreachable")


def main() -> int:
    gsm8k = []
    for offset in range(0, GSM8K_COUNT, 100):
        gsm8k += rows("openai/gsm8k", "main", "test", offset, min(100, GSM8K_COUNT - offset))
    if len(gsm8k) != GSM8K_COUNT or not all({"question", "answer"} <= set(r) for r in gsm8k):
        raise ValueError("unexpected GSM8K rows")

    mmlu = []
    stride = MMLU_TOTAL // MMLU_GROUPS
    for group in range(MMLU_GROUPS):
        mmlu += rows("cais/mmlu", "all", "test", group * stride, MMLU_GROUP_SIZE)
    if (len(mmlu) != MMLU_GROUPS * MMLU_GROUP_SIZE
            or not all({"question", "subject", "choices", "answer"} <= set(r) for r in mmlu)
            or not all(len(r["choices"]) == 4 and r["answer"] in range(4) for r in mmlu)):
        raise ValueError("unexpected MMLU rows")

    for name, data in (("gsm8k", gsm8k), ("mmlu", mmlu)):
        directory = ROOT / "data" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}_sample.json").write_text(json.dumps(data))
        print(f"{name}: {len(data)} rows -> {directory}")
    print(f"MMLU subjects: {len({r['subject'] for r in mmlu})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
