#!/usr/bin/env bash
set -euo pipefail

# Run from the local workstation. The script copies the cleaned dataset to the
# rented GPU host over SSH, runs a read-only GPU/PyTorch preflight there, and
# then starts SmolVLA training in the remote so101-ordering environment.
# It never touches the robot and disables both W&B and Hub uploads.
#
# Usage (after explicit training approval):
#   SMOLVLA_CLOUD_APPROVED=YES bash commands/train_smolvla_cloud.sh smoke
#   SMOLVLA_BATCH_SIZE=8 SMOLVLA_CLOUD_APPROVED=YES bash commands/train_smolvla_cloud.sh smoke
#   SMOLVLA_CLOUD_APPROVED=YES bash commands/train_smolvla_cloud.sh full

MODE="${1:-}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Usage: SMOLVLA_CLOUD_APPROVED=YES $0 {smoke|full}" >&2
  exit 2
fi
if [[ "${SMOLVLA_CLOUD_APPROVED:-}" != "YES" ]]; then
  echo "Refusing to start training. Set SMOLVLA_CLOUD_APPROVED=YES after review." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATASET="${PROJECT_ROOT}/outputs/datasets/vla_pink_cyan_order_clean_v1"

# Override these without editing the script if the provider assigns a new endpoint.
REMOTE_USER="${CLOUD_SSH_USER:-root}"
REMOTE_HOST="${CLOUD_SSH_HOST:?Set CLOUD_SSH_HOST to the rented GPU endpoint}"
REMOTE_PORT="${CLOUD_SSH_PORT:-22}"
REMOTE_DATASET="${CLOUD_DATASET_ROOT:-/root/so101-two-cube-ordering-data/vla_pink_cyan_order_clean_v1}"
REMOTE_ENV="${CLOUD_CONDA_ENV:-so101-ordering}"
LOCAL_POLICY_DIR="${CLOUD_POLICY_LOCAL_DIR:-}"
REMOTE_POLICY_DIR="${CLOUD_POLICY_REMOTE_DIR:-/root/so101-two-cube-ordering-models/smolvla_base}"
REMOTE_POLICY_PATH="lerobot/smolvla_base"

test -f "${LOCAL_DATASET}/meta/info.json"
command -v ssh >/dev/null
command -v rsync >/dev/null

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_OPTS=(-p "${REMOTE_PORT}" -o ServerAliveInterval=60 -o ServerAliveCountMax=10)

echo "Checking remote GPU and creating ${REMOTE_DATASET} ..."
ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "mkdir -p '${REMOTE_DATASET}' && nvidia-smi"

echo "Syncing cleaned dataset (local source is unchanged) ..."
rsync -avP -e "ssh -p ${REMOTE_PORT} -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  "${LOCAL_DATASET}/" "${SSH_TARGET}:${REMOTE_DATASET}/"

if [[ -n "${LOCAL_POLICY_DIR}" ]]; then
  test -f "${LOCAL_POLICY_DIR}/config.json"
  echo "Syncing local SmolVLA base model to ${REMOTE_POLICY_DIR} ..."
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "mkdir -p '${REMOTE_POLICY_DIR}'"
  rsync -avP -e "ssh -p ${REMOTE_PORT} -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
    "${LOCAL_POLICY_DIR}/" "${SSH_TARGET}:${REMOTE_POLICY_DIR}/"
  REMOTE_POLICY_PATH="${REMOTE_POLICY_DIR}"
fi

if [[ "${MODE}" == "smoke" ]]; then
  BATCH_SIZE="${SMOLVLA_BATCH_SIZE:-4}"
  STEPS="${SMOLVLA_STEPS:-200}"
  SAVE_FREQ=200
  EVAL_FREQ=200
  LOG_FREQ=10
  if [[ "${BATCH_SIZE}" == "4" ]]; then
    JOB_NAME=smolvla_smoke_clean_v1
  else
    JOB_NAME="smolvla_smoke_b${BATCH_SIZE}_clean_v1"
  fi
  WANDB_ENABLE=false
  WANDB_PROJECT=so101-two-cube-ordering
  WANDB_DISABLE_ARTIFACT=true
else
  BATCH_SIZE="${SMOLVLA_BATCH_SIZE:-8}"
  STEPS="${SMOLVLA_STEPS:-30000}"
  SAVE_FREQ=5000
  EVAL_FREQ=5000
  LOG_FREQ=100
  JOB_NAME="smolvla_full_30k_b${BATCH_SIZE}_wandb_v1"
  WANDB_ENABLE=true
  WANDB_PROJECT="${SMOLVLA_WANDB_PROJECT:-so101-two-cube-ordering}"
  WANDB_DISABLE_ARTIFACT=true
fi
REMOTE_OUTPUT="${CLOUD_OUTPUT_ROOT:-/root/so101-two-cube-ordering/outputs/train/${JOB_NAME}}"

echo "Starting remote ${MODE} training in conda environment ${REMOTE_ENV} ..."
ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "bash -s -- '${REMOTE_ENV}' '${REMOTE_DATASET}' '${REMOTE_OUTPUT}' '${JOB_NAME}' '${STEPS}' '${SAVE_FREQ}' '${EVAL_FREQ}' '${LOG_FREQ}' '${REMOTE_POLICY_PATH}' '${BATCH_SIZE}' '${WANDB_ENABLE}' '${WANDB_PROJECT}' '${WANDB_DISABLE_ARTIFACT}'" <<'REMOTE_SCRIPT'
set -euo pipefail
trap 'rc=$?; echo "Remote training command exit status: ${rc}" >&2' EXIT
REMOTE_ENV="$1"
REMOTE_DATASET="$2"
REMOTE_OUTPUT="$3"
JOB_NAME="$4"
STEPS="$5"
SAVE_FREQ="$6"
EVAL_FREQ="$7"
LOG_FREQ="$8"
POLICY_PATH="$9"
BATCH_SIZE="${10}"
WANDB_ENABLE="${11}"
WANDB_PROJECT="${12}"
WANDB_DISABLE_ARTIFACT="${13}"

echo "Remote host: $(hostname)"
echo "Remote user: $(id -un)"
echo "Remote dataset: ${REMOTE_DATASET}"
echo "Remote output: ${REMOTE_OUTPUT}"
echo "Policy path: ${POLICY_PATH}"
test -f "${REMOTE_DATASET}/meta/info.json"

# AutoDL's network accelerator is exposed as /etc/network_turbo. Source it
# only on the remote training shell so local proxy settings are untouched.
if [[ -f /etc/network_turbo ]]; then
  echo "Enabling AutoDL network turbo ..."
  # shellcheck disable=SC1091
  source /etc/network_turbo
else
  echo "AutoDL network turbo file not found; using the existing remote network." >&2
fi
if command -v curl >/dev/null 2>&1; then
  curl -fsSI --connect-timeout 15 \
    https://huggingface.co/lerobot/smolvla_base/resolve/main/config.json \
    >/dev/null || echo "WARNING: Hugging Face connectivity check failed; model download may retry." >&2
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
  echo "ERROR: conda was not found in the non-interactive SSH PATH." >&2
  echo "PATH=${PATH}" >&2
  exit 127
fi
echo "Conda: ${CONDA_BIN}"
"${CONDA_BIN}" run --no-capture-output -n "${REMOTE_ENV}" python -c \
  'import torch, lerobot, transformers, num2words, docopt; from torchcodec.decoders import VideoDecoder; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("lerobot", lerobot.__file__); print("transformers", transformers.__version__); print("SmolVLA processor and TorchCodec dependencies: OK")'

"${CONDA_BIN}" run --no-capture-output -n "${REMOTE_ENV}" lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --dataset.repo_id=iptihar/so101_vla_pink_cyan_order_clean_v1 \
  --dataset.root="${REMOTE_DATASET}" \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2"}' \
  --policy.device=cuda \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --eval_freq="${EVAL_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --seed=20260824 \
  --num_workers=2 \
  --output_dir="${REMOTE_OUTPUT}" \
  --job_name="${JOB_NAME}" \
  --policy.push_to_hub=false \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}"
REMOTE_SCRIPT
