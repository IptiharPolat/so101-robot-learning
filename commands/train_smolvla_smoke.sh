#!/usr/bin/env bash
set -euo pipefail

# Local-only 200-step SmolVLA smoke test. This script does not upload the
# dataset or checkpoint to Hugging Face and does not enable W&B logging.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${PROJECT_ROOT}/outputs/datasets/vla_pink_cyan_order_clean_v1"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/train/smolvla_smoke_clean_v1"

test -f "${DATASET_ROOT}/meta/info.json"

exec conda run --no-capture-output -n so101-ordering lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=iptihar/so101_vla_pink_cyan_order_clean_v1 \
  --dataset.root="${DATASET_ROOT}" \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2"}' \
  --policy.device=cuda \
  --batch_size=4 \
  --steps=200 \
  --save_freq=200 \
  --eval_freq=200 \
  --log_freq=10 \
  --seed=20260824 \
  --num_workers=2 \
  --output_dir="${OUTPUT_ROOT}" \
  --job_name=smolvla_smoke_clean_v1 \
  --policy.push_to_hub=false \
  --wandb.enable=false
