#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Default behavior prints a copy-pasteable draft with explicit placeholders and
# never opens hardware. Explicit execution uses only verified config values and
# is still guarded by audit_status plus SO101_HARDWARE_APPROVED=YES.
if [[ "${1:-}" == "--execute" ]]; then
  shift
  exec python scripts/run_record.py --config configs/rig.local.yaml --experiment configs/act_experiment.yaml --phase pilot --execute "$@"
fi

exec python scripts/run_record.py --config configs/rig.local.yaml --experiment configs/act_experiment.yaml --phase pilot --allow-placeholders "$@"
