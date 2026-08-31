#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "${1:-}" != "--execute" ]]; then
  echo "DRY RUN ONLY: 12 rejected strict recovery pairs would be rerecorded:"
  printf '  %s\n' \
    recovery_L4_M09_R04 \
    recovery_L3_M09_R04 \
    recovery_L5_M09_R03 \
    recovery_L1_M09_R03 \
    recovery_L3_M09_R02 \
    recovery_L5_M09_R01 \
    recovery_L3_M09_R01 \
    recovery_L4_M09_R02 \
    recovery_L5_M09_R04 \
    recovery_L2_M09_R03 \
    recovery_L5_M09_R02 \
    recovery_L2_M09_R02
  echo "No camera, serial port, robot, dataset, upload, or training process was opened."
  exit 0
fi

if [[ "${SO101_VLA_RECOVERY_RETRY_BATCH_APPROVED:-}" != "YES" ]]; then
  echo "Execution refused: SO101_VLA_RECOVERY_RETRY_BATCH_APPROVED is not YES." >&2
  exit 2
fi

pairs=(
  recovery_L4_M09_R04
  recovery_L3_M09_R04
  recovery_L5_M09_R03
  recovery_L1_M09_R03
  recovery_L3_M09_R02
  recovery_L5_M09_R01
  recovery_L3_M09_R01
  recovery_L4_M09_R02
  recovery_L5_M09_R04
  recovery_L2_M09_R03
  recovery_L5_M09_R02
  recovery_L2_M09_R02
)

echo "Targeted retry batch: ${#pairs[@]} whole pairs / 24 episodes."
echo "Raw takes are appended locally; upload remains disabled."

for index in "${!pairs[@]}"; do
  pair_id="${pairs[$index]}"
  echo "============================================================"
  echo "Pair $((index + 1))/${#pairs[@]}: ${pair_id}"
  pair_status="$({ python3 - "$pair_id" <<'PY'
import csv
import sys

pair_id = sys.argv[1]
with open("manifests/vla_recovery_episode_schedule.csv", newline="", encoding="utf-8") as stream:
    pair = sorted(
        (row for row in csv.DictReader(stream) if row["pair_id"] == pair_id),
        key=lambda row: int(row["pair_position"]),
    )
if len(pair) != 2:
    raise SystemExit(f"{pair_id}: expected two manifest rows")
print(",".join(row["status"] for row in pair))
PY
  } )"
  case "$pair_status" in
    accepted,accepted|recorded_pending_qc,recorded_pending_qc)
      echo "Skip ${pair_id}: both replacement takes are already recorded (${pair_status})."
      continue
      ;;
    rejected_needs_retry,rejected_needs_retry)
      ;;
    *)
      echo "Refuse ambiguous partial pair ${pair_id}: ${pair_status}" >&2
      exit 2
      ;;
  esac
  SO101_VLA_RECOVERY_RETRY_APPROVED=YES \
    commands/retry_vla_recovery_pair.sh "$pair_id" --execute
  echo "Completed ${pair_id}; offline pair QC is still required."
done

echo "All 12 targeted pairs were appended locally. Run whole-pair QC; do not derive or train yet."
