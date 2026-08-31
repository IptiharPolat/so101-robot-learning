#!/usr/bin/env bash
set -euo pipefail

# Recovery-v2 cloud launcher. It is offline/dry-run by default: no SSH, rsync,
# authentication, upload, or training happens unless --execute and both approval
# guards are supplied. Dataset/model Hub upload remains disabled.

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
LOCAL_DATASET="${PROJECT_ROOT}/outputs/datasets/vla_pink_cyan_order_recovery_v2"
DATASET_ID="iptihar/so101_vla_pink_cyan_order_recovery_v2"
SEED=20260829

REMOTE_USER="${CLOUD_SSH_USER:-root}"
REMOTE_HOST="${CLOUD_SSH_HOST:-REPLACE_WITH_HOST}"
REMOTE_PORT="${CLOUD_SSH_PORT:-22}"
REMOTE_ENV="${CLOUD_CONDA_ENV:-so101-ordering}"
REMOTE_DATASET="${CLOUD_DATASET_ROOT:-/root/so101-two-cube-ordering-data/vla_pink_cyan_order_recovery_v2}"
LOCAL_POLICY_DIR="${CLOUD_POLICY_LOCAL_DIR:-}"
REMOTE_POLICY_DIR="${CLOUD_POLICY_REMOTE_DIR:-/root/so101-two-cube-ordering-models/smolvla_base}"
POLICY_PATH="lerobot/smolvla_base"

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
WANDB_PROJECT="${SMOLVLA_WANDB_PROJECT:-so101-two-cube-ordering}"
REMOTE_OUTPUT="${CLOUD_OUTPUT_ROOT:-/root/so101-two-cube-ordering/outputs/train/${JOB_NAME}}"
SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_OPTS=(-p "$REMOTE_PORT" -o ServerAliveInterval=60 -o ServerAliveCountMax=10)

echo "Mode: ${MODE}"
echo "Local dataset: ${LOCAL_DATASET}"
echo "Remote: ${SSH_TARGET}:${REMOTE_PORT}"
echo "Remote dataset: ${REMOTE_DATASET}"
echo "Remote output: ${REMOTE_OUTPUT}"
echo "Steps/batch/save: ${STEPS}/${BATCH_SIZE}/${SAVE_FREQ}"
echo "Dataset/Hub upload: disabled"
echo "Model/Hub upload: disabled"
echo "W&B scalars: ${WANDB_ENABLE}; W&B artifacts: disabled"

if [[ "$EXECUTE" != "--execute" ]]; then
  echo "DRY RUN ONLY: no network connection, sync, or training was started."
  exit 0
fi
if [[ "${SMOLVLA_RECOVERY_CLOUD_APPROVED:-}" != "YES" ]]; then
  echo "Execution refused: SMOLVLA_RECOVERY_CLOUD_APPROVED is not YES." >&2
  exit 2
fi
if [[ "$REMOTE_HOST" == "REPLACE_WITH_HOST" ]]; then
  echo "Execution refused: set CLOUD_SSH_HOST to the rented GPU endpoint." >&2
  exit 2
fi
if [[ "$MODE" == "full" && "${SMOLVLA_RECOVERY_CLOUD_FULL_APPROVED:-}" != "YES" ]]; then
  echo "Full run refused: SMOLVLA_RECOVERY_CLOUD_FULL_APPROVED is not YES." >&2
  exit 2
fi
if [[ ! -f "${LOCAL_DATASET}/meta/info.json" ]]; then
  echo "Execution refused: recovery-v2 dataset has not been built and QC-approved." >&2
  exit 2
fi
command -v ssh >/dev/null
command -v rsync >/dev/null

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p -- '${REMOTE_DATASET}'"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" nvidia-smi

rsync -avP \
  -e "ssh -p ${REMOTE_PORT} -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  "${LOCAL_DATASET}/" "${SSH_TARGET}:${REMOTE_DATASET}/"

if [[ -n "$LOCAL_POLICY_DIR" ]]; then
  if [[ ! -f "${LOCAL_POLICY_DIR}/config.json" ]]; then
    echo "Local policy directory lacks config.json: ${LOCAL_POLICY_DIR}" >&2
    exit 2
  fi
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p -- '${REMOTE_POLICY_DIR}'"
  rsync -avP \
    -e "ssh -p ${REMOTE_PORT} -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
    "${LOCAL_POLICY_DIR}/" "${SSH_TARGET}:${REMOTE_POLICY_DIR}/"
  POLICY_PATH="$REMOTE_POLICY_DIR"
fi

remote_args=(
  "$REMOTE_ENV" "$REMOTE_DATASET" "$REMOTE_OUTPUT" "$JOB_NAME"
  "$STEPS" "$BATCH_SIZE" "$SAVE_FREQ" "$LOG_FREQ" "$SCHEDULER_DECAY"
  "$POLICY_PATH" "$WANDB_ENABLE" "$WANDB_PROJECT" "$DATASET_ID" "$SEED"
)
printf -v remote_arg_string ' %q' "${remote_args[@]}"

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "bash -s --${remote_arg_string}" <<'REMOTE_SCRIPT'
set -euo pipefail
trap 'rc=$?; echo "Remote recovery training exit status: ${rc}" >&2' EXIT

REMOTE_ENV="$1"
REMOTE_DATASET="$2"
REMOTE_OUTPUT="$3"
JOB_NAME="$4"
STEPS="$5"
BATCH_SIZE="$6"
SAVE_FREQ="$7"
LOG_FREQ="$8"
SCHEDULER_DECAY="$9"
POLICY_PATH="${10}"
WANDB_ENABLE="${11}"
WANDB_PROJECT="${12}"
DATASET_ID="${13}"
SEED="${14}"

if [[ -e "$REMOTE_OUTPUT" ]]; then
  echo "Refusing to overwrite remote output: ${REMOTE_OUTPUT}" >&2
  exit 2
fi
if [[ ! -f "${REMOTE_DATASET}/meta/info.json" ]]; then
  echo "Remote dataset is incomplete: ${REMOTE_DATASET}" >&2
  exit 2
fi

if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x /root/miniconda3/bin/conda ]]; then
  CONDA_BIN=/root/miniconda3/bin/conda
elif [[ -x /root/miniforge3/bin/conda ]]; then
  CONDA_BIN=/root/miniforge3/bin/conda
elif [[ -x /opt/conda/bin/conda ]]; then
  CONDA_BIN=/opt/conda/bin/conda
else
  echo "Conda was not found on the remote host." >&2
  exit 127
fi

nvidia-smi
df -h "$(dirname "$REMOTE_OUTPUT")" || true
"${CONDA_BIN}" run --no-capture-output -n "$REMOTE_ENV" python -c \
  'import torch, lerobot, transformers, num2words, docopt; from torchcodec.decoders import VideoDecoder; assert torch.cuda.is_available(); print("torch", torch.__version__); print("lerobot", lerobot.__file__); print("transformers", transformers.__version__)'

"${CONDA_BIN}" run --no-capture-output -n "$REMOTE_ENV" python - "$REMOTE_DATASET" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
episode_files = sorted((root / "meta/episodes").glob("**/*.parquet"))
assert episode_files, "no episode metadata parquet files"
episodes = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
counts = Counter(item[0] for item in episodes["tasks"])
pink = "First pick up the pink cube and place it in the center target area, then pick up the cyan cube and place it in the center target area."
cyan = "First pick up the cyan cube and place it in the center target area, then pick up the pink cube and place it in the center target area."
assert info["total_episodes"] == 122, info["total_episodes"]
assert counts == Counter({pink: 61, cyan: 61}), counts
assert info["features"]["action"]["shape"] == [6]
assert info["features"]["observation.state"]["shape"] == [6]
for key in ("observation.images.front", "observation.images.side"):
    assert key in info["features"], key
print("recovery dataset gate: 122 episodes, 61/61 tasks, two cameras, 6D state/action")
PY

"${CONDA_BIN}" run --no-capture-output -n "$REMOTE_ENV" lerobot-train \
  --policy.path="$POLICY_PATH" \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$REMOTE_DATASET" \
  --dataset.image_transforms.enable=false \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2"}' \
  --policy.device=cuda \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="$SCHEDULER_DECAY" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=0 \
  --log_freq="$LOG_FREQ" \
  --seed="$SEED" \
  --num_workers=2 \
  --output_dir="$REMOTE_OUTPUT" \
  --job_name="$JOB_NAME" \
  --policy.push_to_hub=false \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.disable_artifact=true
REMOTE_SCRIPT
