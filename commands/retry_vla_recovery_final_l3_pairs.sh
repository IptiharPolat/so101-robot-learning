#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

pairs=(recovery_L3_M09_R02 recovery_L3_M09_R01)

if [[ "${1:-}" != "--execute" ]]; then
  echo "DRY RUN ONLY: two L3_M09 pairs / four episodes would be appended locally."
  printf '  %s\n' "${pairs[@]}"
  echo "No camera, serial port, robot, dataset, upload, or training process was opened."
  exit 0
fi

if [[ "${SO101_VLA_RECOVERY_FINAL_L3_APPROVED:-}" != "YES" ]]; then
  echo "Execution refused: SO101_VLA_RECOVERY_FINAL_L3_APPROVED is not YES." >&2
  exit 2
fi

echo "Final targeted retry: two L3_M09 pairs / four episodes."
echo "Keep gripper tip, shoulder pan, elbow, wrist, and gripper opening identical within each pair."
echo "Uploads remain disabled."

for index in "${!pairs[@]}"; do
  pair_id="${pairs[$index]}"
  echo "============================================================"
  echo "Final pair $((index + 1))/${#pairs[@]}: ${pair_id}"
  SO101_VLA_RECOVERY_RETRY_APPROVED=YES \
    commands/retry_vla_recovery_pair.sh "$pair_id" --execute
done

echo "Both final L3 pairs were appended locally. Run final whole-pair QC; do not derive or train yet."
