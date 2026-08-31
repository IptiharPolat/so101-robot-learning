#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_user="${CLOUD_SSH_USER:-root}"
remote_host="${CLOUD_SSH_HOST:-REPLACE_WITH_HOST}"
remote_port="${CLOUD_SSH_PORT:-22}"
remote_run="${CLOUD_RECOVERY_RUN:-/root/so101-two-cube-ordering/outputs/train/smolvla_recovery_v2_20k_b8}"
local_root="${project_root}/outputs/models/smolvla_recovery_v2"
steps=(002500 005000 007500 010000 012500 015000 017500)

echo "Remote: ${remote_user}@${remote_host}:${remote_port}"
echo "Remote run: ${remote_run}"
echo "Local root: ${local_root}"
echo "Checkpoints: ${steps[*]}"
echo "Scope: pretrained_model only; no training_state; no remote deletion"

if [[ "${1:-}" != "--execute" ]]; then
  echo "DRY RUN ONLY: no SSH, rsync, or filesystem change was performed."
  exit 0
fi
if [[ "${SMOLVLA_RECOVERY_FETCH_APPROVED:-}" != "YES" ]]; then
  echo "Execution refused: SMOLVLA_RECOVERY_FETCH_APPROVED is not YES." >&2
  exit 2
fi
if [[ "$remote_host" == "REPLACE_WITH_HOST" ]]; then
  echo "Execution refused: set CLOUD_SSH_HOST to the rented GPU endpoint." >&2
  exit 2
fi

mkdir -p "$local_root"
for step in "${steps[@]}"; do
  destination="${local_root}/${step}"
  if [[ -s "${destination}/model.safetensors" ]]; then
    echo "Already present, skipping ${step}: ${destination}"
    continue
  fi
  mkdir -p "$destination"
  rsync -avP \
    -e "ssh -p ${remote_port} -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
    "${remote_user}@${remote_host}:${remote_run}/checkpoints/${step}/pretrained_model/" \
    "${destination}/"
  for required in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [[ ! -s "${destination}/${required}" ]]; then
      echo "Incomplete checkpoint ${step}: missing ${required}" >&2
      exit 1
    fi
  done
done

du -sh "${local_root}"/*
