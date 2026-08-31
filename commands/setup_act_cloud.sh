#!/usr/bin/env bash
set -euo pipefail

LERO_REPO="https://github.com/Seeed-Projects/lerobot.git"
LERO_SHA="0f392484458cb5ebca0310c0c4c47390a31c80ed"
CLOUD_ROOT="${SO101_CLOUD_ROOT:-/root/autodl-tmp}"
LERO_ROOT="$CLOUD_ROOT/lerobot"
CONDA_ENV="so101-ordering"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"

if [[ "${1:-}" != "--execute" ]]; then
  echo "DRY RUN: this creates $CONDA_ENV and installs the pinned LeRobot checkout."
  echo "Run: bash commands/setup_act_cloud.sh --execute"
  exit 0
fi

command -v git >/dev/null
command -v nvidia-smi >/dev/null
if [[ ! -x "$CONDA_BIN" ]]; then
  echo "REFUSED: Conda executable not found: $CONDA_BIN" >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
  echo "REFUSED: Conda environment already exists: $CONDA_ENV" >&2
  exit 1
fi
if [[ -e "$LERO_ROOT" ]]; then
  echo "REFUSED: target already exists: $LERO_ROOT" >&2
  exit 1
fi

nvidia-smi
"$CONDA_BIN" create --name "$CONDA_ENV" python=3.10 pip -y
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -m pip install \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
git clone "$LERO_REPO" "$LERO_ROOT"
git -C "$LERO_ROOT" checkout "$LERO_SHA"
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -m pip install -e "$LERO_ROOT"
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -m pip install \
  transformers==4.57.6 tokenizers==0.22.2
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -m pip check
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -c \
  'import torch, lerobot; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(lerobot.__version__, lerobot.__file__)'

echo "Environment ready. Log in interactively with 'hf auth login' and 'wandb login'; never paste tokens into project files."
