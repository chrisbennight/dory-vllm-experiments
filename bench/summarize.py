"""Summarizes one or more benchmark run directories into a Markdown table.

Usage: python bench/summarize.py results/<run id> [results/<run id> ...]
"""

import json
import sys
from pathlib import Path

VARIANT_ORDER = {"bf16": 0, "mxfp8": 1, "nvfp4": 2}


def load_rows(run_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = []
    for config in manifest["configs"]:
        for filename in config["benchmarks"]:
            result = json.loads((run_dir / filename).read_text())
            # Filenames look like bench_<variant>_loops<n>_in<len>_out<len>_c<n>.json
            parts = filename.removesuffix(".json").split("_")
            input_len = int(parts[3].removeprefix("in"))
            output_len = int(parts[4].removeprefix("out"))
            rows.append({
                "attention": (manifest.get("attention_backend") or "default")
                + (" + fused decode" if manifest.get("dory_fused_decode") else "")
                + (f" + {manifest['dory_lm_head_8bit'].upper()} LM head"
                   if manifest.get("dory_lm_head_8bit") not in (None, "none") else "")
                + (" + wavefront" if manifest.get("wavefront_decode") else "")
                + (" (V1 model runner)" if manifest.get("vllm_use_v2_model_runner") == "0"
                   else "")
                + (" (no async scheduling)" if manifest.get("async_scheduling") is False
                   else ""),
                "variant": config["variant"],
                "loops": config["loops"],
                "input_len": input_len,
                "output_len": output_len,
                "concurrency": result["max_concurrency"],
                "completed": result["completed"],
                "failed": result["failed"],
                "output_tps": result["output_throughput"],
                "per_user_tps": 1000 / result["median_tpot_ms"],
                "median_ttft_ms": result["median_ttft_ms"],
                "p99_ttft_ms": result["p99_ttft_ms"],
                "median_tpot_ms": result["median_tpot_ms"],
            })
    return manifest, rows


def main() -> int:
    manifests, rows = [], []
    for run_dir in sys.argv[1:]:
        manifest, run_rows = load_rows(Path(run_dir))
        manifests.append(manifest)
        rows.extend(run_rows)
    rows.sort(key=lambda r: (r["input_len"], r["output_len"], r["concurrency"],
                             r["loops"], VARIANT_ORDER[r["variant"]], r["attention"]))
    bf16_tps = {(r["attention"], r["loops"], r["input_len"], r["output_len"],
                 r["concurrency"]): r["output_tps"] for r in rows if r["variant"] == "bf16"}

    for manifest in manifests:
        print(f"Run {manifest['run_id']} | {manifest['gpu']['name']} | "
              f"vLLM {manifest['versions']['vllm']} | torch {manifest['versions']['torch']}")
    print()
    print("| Input/Output | Concurrency | Loops | Weights | Attention | Output tok/s "
          "| vs BF16 | Per-user tok/s | Median TTFT ms | P99 TTFT ms | Failed |")
    print("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        key = (r["attention"], r["loops"], r["input_len"], r["output_len"], r["concurrency"])
        ratio = f"{r['output_tps'] / bf16_tps[key]:.2f}x" if key in bf16_tps else "n/a"
        print(f"| {r['input_len']}/{r['output_len']} | {r['concurrency']} | {r['loops']} "
              f"| {r['variant']} | {r['attention']} | {r['output_tps']:.1f} | {ratio} "
              f"| {r['per_user_tps']:.1f} | {r['median_ttft_ms']:.0f} "
              f"| {r['p99_ttft_ms']:.0f} | {r['failed']} |")

    problems = [f"- {m['run_id']}: run aborted: {m['aborted']}"
                for m in manifests if m.get("aborted")]
    problems += [f"- {c['variant']} loops={c['loops']}: {c['status']}: "
                 f"{'; '.join(c.get('errors', []))}"
                 for m in manifests for c in m["configs"] if c["status"] != "ok"]
    if problems:
        print()
        print("Problems:")
        print("\n".join(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
