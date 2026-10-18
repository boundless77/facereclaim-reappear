#!/usr/bin/env bash
set -euo pipefail
# Run from any working directory. All paths and sizes can be overridden via environment variables.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ARGS=()
if [[ -n "${SAVE_ROOT:-}" ]]; then
  OUTPUT_ARGS+=(--save_root "$SAVE_ROOT")
fi
# Preserve the GPU visibility assigned by Slurm; do not select a physical GPU here.
python "$PROJECT_DIR/reclaim_from_clip.py" \
  --fake_root "${FAKE_ROOT:-/mnt/home/chenxiaofu/facetrack/data/deepfakes/image/celeba64_std_100c/infoswap}" \
  --weights_root "${WEIGHTS_ROOT:-/mnt/data/zhonghuanlong/models/faceReclaim}" \
  --clip_path "${CLIP_PATH:-/mnt/data/zhonghuanlong/models/clip-vit-large-patch14}" \
  --batch_size "${BATCH_SIZE:-4}" --num_workers "${NUM_WORKERS:-4}" \
  --save_workers "${SAVE_WORKERS:-2}" --ratio -5 "${OUTPUT_ARGS[@]}" "$@"
