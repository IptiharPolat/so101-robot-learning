#!/usr/bin/env bash
set -euo pipefail

# Local-only 30K-step SmolVLA fine-tuning command. Run only after the smoke
# test is reviewed and explicitly approved. No dataset/checkpoint upload and
# scalar W&B logging is enabled, while model artifacts and Hub uploads remain disabled.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${PROJECT_ROOT}/outputs/datasets/vla_pink_cyan_order_clean_v1"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/train/smolvla_full_30k_b8_wandb_v1"

test -f "${DATASET_ROOT}/meta/info.json"

exec conda run --no-capture-output -n so101-ordering lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=iptihar/so101_vla_pink_cyan_order_clean_v1 \
  --dataset.root="${DATASET_ROOT}" \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2"}' \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=5000 \
  --log_freq=100 \
  --seed=20260824 \
  --num_workers=2 \
  --output_dir="${OUTPUT_ROOT}" \
  --job_name=smolvla_full_30k_b8_wandb_v1 \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=so101-two-cube-ordering \
  --wandb.disable_artifact=true
