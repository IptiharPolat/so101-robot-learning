#!/usr/bin/env bash
set -euo pipefail

# Read-only reload check for the completed cloud smoke checkpoint.
REMOTE_USER="${CLOUD_SSH_USER:-root}"
REMOTE_HOST="${CLOUD_SSH_HOST:?Set CLOUD_SSH_HOST to the rented GPU endpoint}"
REMOTE_PORT="${CLOUD_SSH_PORT:-22}"
REMOTE_ENV="${CLOUD_CONDA_ENV:-so101-ordering}"
CHECKPOINT="${CLOUD_SMOKE_CHECKPOINT:-/root/so101-two-cube-ordering/outputs/train/smolvla_smoke_clean_v1/checkpoints/last/pretrained_model}"

ssh -p "${REMOTE_PORT}" \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=10 \
  "${REMOTE_USER}@${REMOTE_HOST}" \
  "bash -s -- '${REMOTE_ENV}' '${CHECKPOINT}'" <<'REMOTE_SCRIPT'
set -euo pipefail
REMOTE_ENV="$1"
CHECKPOINT="$2"

test -f "${CHECKPOINT}/config.json"
test -f "${CHECKPOINT}/model.safetensors"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo >/dev/null
fi
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x /root/miniconda3/bin/conda ]]; then
  CONDA_BIN=/root/miniconda3/bin/conda
elif [[ -x /root/miniforge3/bin/conda ]]; then
  CONDA_BIN=/root/miniforge3/bin/conda
else
  echo "ERROR: conda not found" >&2
  exit 127
fi

"${CONDA_BIN}" run --no-capture-output -n "${REMOTE_ENV}" \
  python - "${CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

checkpoint = Path(sys.argv[1])
policy = SmolVLAPolicy.from_pretrained(checkpoint)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=checkpoint,
)
total = sum(parameter.numel() for parameter in policy.parameters())
trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
print(f"checkpoint={checkpoint}")
print(f"policy_type={policy.config.type}")
print(f"total_parameters={total}")
print(f"trainable_parameters={trainable}")
print(f"preprocessor={type(preprocessor).__name__}")
print(f"postprocessor={type(postprocessor).__name__}")
print("SMOLVLA_CHECKPOINT_RELOAD_OK")
PY
REMOTE_SCRIPT
