#!/usr/bin/env bash
# Installs a CUDA 13.0 compiler toolchain for JIT-compiled kernels (FlashInfer,
# PyTorch extensions) without touching the benchmark environment or the system.
#
# NVIDIA's pip packages are installed into cuda/site. Their layout differs from
# a standard CUDA install only in naming the library directory lib instead of
# lib64 and omitting the unversioned libcudart.so, so both are added as links.
# The version matches PyTorch's CUDA 13.0 runtime, which the installed driver
# supports. Use it with:
#   export CUDA_HOME="$PWD/cuda/home" PATH="$PWD/cuda/home/bin:$PATH"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$ROOT/cuda/site"
TOOLKIT_VERSION=13.0.3.0

uv pip install --python "$ROOT/.venv/bin/python" --target "$SITE" \
  "cuda-toolkit[nvcc,cudart,cccl]==$TOOLKIT_VERSION"

CU="$SITE/nvidia/cu13"
ln -sfn lib "$CU/lib64"
ln -sfn libcudart.so.13 "$CU/lib/libcudart.so"
ln -sfn "$CU" "$ROOT/cuda/home"

"$ROOT/cuda/home/bin/nvcc" --version | tail -2
