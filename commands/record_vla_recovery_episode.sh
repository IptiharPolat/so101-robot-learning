#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Dry-run unless --execute is explicitly supplied to the Python guard.
exec python scripts/run_vla_record.py \
  --config configs/rig.local.yaml \
  --experiment configs/smolvla_recovery_experiment.yaml \
  --manifest manifests/vla_recovery_episode_schedule.csv \
  "$@"
