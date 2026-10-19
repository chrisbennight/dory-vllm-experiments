#!/usr/bin/env bash
# Downloads the three official Dory checkpoints at pinned revisions.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
declare -A REVISIONS=(
  [bf16]="BF16 b366b87f762ea334ba3eb55912031baba1515810"
  [mxfp8]="MXFP8 f35880f52368271136b48e3a4365f91e70a92308"
  [nvfp4]="NVFP4 41f02335e1a1ad3289b8b32485e2265cbe97cfdf"
)
for variant in nvfp4 mxfp8 bf16; do
  read -r suffix revision <<<"${REVISIONS[$variant]}"
  uvx --from huggingface_hub hf download \
    "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-$suffix" \
    --revision "$revision" \
    --local-dir "$ROOT/models/$variant" \
    --exclude '*.png' --exclude '*.jpeg'
done
du -sh "$ROOT"/models/*
