#!/usr/bin/env bash
# Builds the Python environment for the vLLM Dory baseline.
# vLLM is installed from the pinned head of the Dory support PR, reusing the
# prebuilt kernels published for that PR's base commit on main.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_PR_SHA=55076950b6c155f384d6766d63563c3bc20a0e3f
VLLM_BASE_SHA=153a17166e5d07bc38184dedeea542d2e8885e55
SRC="$ROOT/src/vllm"

if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/vllm-project/vllm.git
fi
git -C "$SRC" fetch -q --depth 1 origin "$VLLM_PR_SHA"
git -C "$SRC" checkout -q --detach "$VLLM_PR_SHA"
test "$(git -C "$SRC" rev-parse HEAD)" = "$VLLM_PR_SHA"

uv venv --allow-existing --python 3.12 "$ROOT/.venv"
cd "$SRC"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_BASE_SHA" \
  uv pip install --python "$ROOT/.venv/bin/python" --editable .
"$ROOT/.venv/bin/python" -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__, "| gpu", torch.cuda.get_device_name(0))'
