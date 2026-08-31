#!/usr/bin/env bash
set -euo pipefail

# Draft-only by default. This command never touches the robot, but training still
# requires a fresh explicit user gate and a QC-approved derived v2 dataset.

MODE="${1:-}"
EXECUTE="${2:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 {smoke|full} [--execute]" >&2
  exit 2
fi
if [[ -n "$EXECUTE" && "$EXECUTE" != "--execute" ]]; then
  echo "Second argument must be --execute or omitted." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${PROJECT_ROOT}/outputs/datasets/vla_pink_cyan_order_recovery_v2"
DATASET_ID="iptihar/so101_vla_pink_cyan_order_recovery_v2"
SEED=20260829

if [[ "$MODE" == "smoke" ]]; then
  STEPS=200
  BATCH_SIZE=4
  SAVE_FREQ=200
  LOG_FREQ=10
  SCHEDULER_DECAY=30000
  JOB_NAME=smolvla_recovery_v2_smoke_b4
  WANDB_ENABLE=false
else
  STEPS=20000
  BATCH_SIZE=8
  SAVE_FREQ=2500
  LOG_FREQ=100
  SCHEDULER_DECAY=20000
  JOB_NAME=smolvla_recovery_v2_20k_b8
  WANDB_ENABLE=true
fi
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/train/${JOB_NAME}"

command=(
  conda run --no-capture-output -n so101-ordering lerobot-train
  --policy.path=lerobot/smolvla_base
  --dataset.repo_id="${DATASET_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.image_transforms.enable=false
  '--rename_map={"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2"}'
  --policy.device=cuda
  --policy.train_expert_only=true
  --policy.freeze_vision_encoder=true
  --policy.optimizer_lr=0.0001
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --eval_freq=0
  --log_freq="${LOG_FREQ}"
  --seed="${SEED}"
  --num_workers=2
  --output_dir="${OUTPUT_ROOT}"
  --job_name="${JOB_NAME}"
  --policy.push_to_hub=false
  --wandb.enable="${WANDB_ENABLE}"
  --wandb.project=so101-two-cube-ordering
  --wandb.disable_artifact=true
)

printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "$EXECUTE" != "--execute" ]]; then
  echo "DRY RUN ONLY: no training was started."
  exit 0
fi
if [[ "${SMOLVLA_RECOVERY_TRAIN_APPROVED:-}" != "YES" ]]; then
  echo "Execution refused: SMOLVLA_RECOVERY_TRAIN_APPROVED is not YES." >&2
  exit 2
fi
if [[ "$MODE" == "full" && "${SMOLVLA_RECOVERY_FULL_APPROVED:-}" != "YES" ]]; then
  echo "Full run refused: SMOLVLA_RECOVERY_FULL_APPROVED is not YES." >&2
  exit 2
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Execution refused: recovery dataset has not been built and QC-approved." >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Execution refused: output already exists: ${OUTPUT_ROOT}" >&2
  exit 2
fi

exec "${command[@]}"
