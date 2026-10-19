"""Splits single-user decode time into weight streaming and per-layer overhead.

Fits, across the single-user 128/1024 runs of every checkpoint and loop count:

    time per output token = overhead per sublayer * sublayers + bytes read / bandwidth

where a sublayer is one attention block or one MLP block, and bytes read are
the quantized linear weights, the BF16 output projection, and the KV cache.
Parameter counts come from the Dory config. This is a model, not a profile:
it assumes the per-sublayer overhead does not depend on the weight format.

Usage: python bench/fit_overhead.py results/<run id> [results/<run id> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

HIDDEN, INTERMEDIATE, HEADS, KV_HEADS, HEAD_DIM, VOCAB = 2560, 7168, 16, 4, 256, 131072
INPUT_PAIRS, OUTPUT_PAIRS, RECURRENT_PAIRS = 8, 8, 20
# q, k, v and output-gate projections, plus the output projection.
ATTENTION_PARAMS = (HIDDEN * (2 * HEADS + 2 * KV_HEADS) * HEAD_DIM
                    + HEADS * HEAD_DIM * HIDDEN)
MLP_PARAMS = 3 * HIDDEN * INTERMEDIATE
PAIR_PARAMS = ATTENTION_PARAMS + MLP_PARAMS
LM_HEAD_BYTES = VOCAB * HIDDEN * 2
# Weight bytes per parameter including block scales.
BYTES_PER_PARAM = {"bf16": 2.0, "mxfp8": 1 + 1 / 32, "nvfp4": 0.5 + 1 / 16}
KV_BYTES_PER_TOKEN_PER_LAYER = 2 * KV_HEADS * HEAD_DIM * 2
INPUT_LEN, OUTPUT_LEN = 128, 1024
RTX_4090_PEAK_BYTES_PER_S = 1008e9


def main() -> int:
    rows = []
    for run_dir in sys.argv[1:]:
        for path in Path(run_dir).glob(f"bench_*_in{INPUT_LEN}_out{OUTPUT_LEN}_c1.json"):
            _, variant, loops_part, *_ = path.name.split("_")
            loops = int(loops_part.removeprefix("loops"))
            pairs = INPUT_PAIRS + OUTPUT_PAIRS + RECURRENT_PAIRS * loops
            average_context = INPUT_LEN + OUTPUT_LEN / 2
            bytes_read = (pairs * PAIR_PARAMS * BYTES_PER_PARAM[variant] + LM_HEAD_BYTES
                          + pairs * average_context * KV_BYTES_PER_TOKEN_PER_LAYER)
            seconds = json.loads(path.read_text())["median_tpot_ms"] / 1000
            rows.append((variant, loops, 2 * pairs, bytes_read, seconds))

    design = np.array([[sublayers, bytes_read] for _, _, sublayers, bytes_read, _ in rows])
    measured = np.array([seconds for *_, seconds in rows])
    (overhead, inverse_bandwidth), *_ = np.linalg.lstsq(design, measured, rcond=None)

    print(f"Overhead per sublayer: {overhead * 1e6:.1f} us")
    print(f"Effective weight bandwidth: {1 / inverse_bandwidth / 1e9:.0f} GB/s")
    print()
    print("| Weights | Loops | Measured ms/token | Fitted ms/token | Overhead share "
          "| GB read/token | Bandwidth ceiling tok/s |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for variant, loops, sublayers, bytes_read, seconds in sorted(rows, key=lambda r: (r[1], r[0])):
        fitted = overhead * sublayers + bytes_read * inverse_bandwidth
        print(f"| {variant} | {loops} | {seconds * 1e3:.2f} | {fitted * 1e3:.2f} "
              f"| {overhead * sublayers / fitted:.0%} | {bytes_read / 1e9:.2f} "
              f"| {RTX_4090_PEAK_BYTES_PER_S / bytes_read:.0f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
