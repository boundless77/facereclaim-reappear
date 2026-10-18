#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ARGS=()
if [[ -n "${OUTPUT_BASE:-}" ]]; then
  OUTPUT_ARGS+=(--output_base "$OUTPUT_BASE")
fi
# Preserve the GPU visibility assigned by Slurm; do not select a physical GPU here.
python "$PROJECT_DIR/reclaim_from_clip.py" --all_datasets \
  --input_base "${INPUT_BASE:-/mnt/home/chenxiaofu/facetrack/data/deepfakes/image}" \
  --weights_root "${WEIGHTS_ROOT:-/mnt/data/zhonghuanlong/models/faceReclaim}" \
  --clip_path "${CLIP_PATH:-/mnt/data/zhonghuanlong/models/clip-vit-large-patch14}" \
  --batch_size "${BATCH_SIZE:-8}" --num_workers "${NUM_WORKERS:-4}" \
  --save_workers "${SAVE_WORKERS:-2}" --ratio -5 "${OUTPUT_ARGS[@]}" "$@"
