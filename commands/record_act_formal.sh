#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Default behavior only prints the 50-episode formal command. Hardware access
# requires both --execute and the explicit SO101_HARDWARE_APPROVED=YES gate.
if [[ "${1:-}" == "--execute" ]]; then
  shift
  exec python scripts/run_record.py --config configs/rig.local.yaml --experiment configs/act_experiment.yaml --phase formal --execute "$@"
fi

exec python scripts/run_record.py --config configs/rig.local.yaml --experiment configs/act_experiment.yaml --phase formal --allow-placeholders "$@"
