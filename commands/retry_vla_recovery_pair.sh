#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

pair_id="${1:-}"
mode="${2:-}"
if [[ -z "$pair_id" || ( -n "$mode" && "$mode" != "--execute" ) ]]; then
  echo "Usage: $0 PAIR_ID [--execute]" >&2
  exit 2
fi

mapfile -t rows < <(
  python3 - "$pair_id" <<'PY'
import csv
import sys
from pathlib import Path

pair_id = sys.argv[1]
with Path("manifests/vla_recovery_episode_schedule.csv").open(newline="", encoding="utf-8") as stream:
    pair = sorted(
        (row for row in csv.DictReader(stream) if row["pair_id"] == pair_id),
        key=lambda row: int(row["pair_position"]),
    )
if len(pair) != 2 or any(row["status"] != "rejected_needs_retry" for row in pair):
    raise SystemExit("pair retry requires exactly two rejected_needs_retry rows")
for row in pair:
    print("\t".join((
        row["episode_id"], row["first_color"], row["second_color"],
        row["base_micro_layout_id"], row["pink_x_offset_mm"],
        row["pink_y_offset_mm"], row["cyan_x_offset_mm"],
        row["cyan_y_offset_mm"],
    )))
PY
)
if ((${#rows[@]} != 2)); then
  echo "Pair retry preflight failed: expected two rejected rows for ${pair_id}." >&2
  exit 2
fi

if [[ "$mode" != "--execute" ]]; then
  echo "DRY RUN ONLY: whole pair ${pair_id} would be appended in both raw datasets:"
  printf '  %s\n' "${rows[@]}"
  echo "No camera, serial port, robot, dataset, upload, or training process was opened."
  exit 0
fi
if [[ "${SO101_VLA_RECOVERY_RETRY_APPROVED:-}" != "YES" ]]; then
  echo "Execution refused: SO101_VLA_RECOVERY_RETRY_APPROVED is not YES." >&2
  exit 2
fi

for row in "${rows[@]}"; do
  IFS=$'\t' read -r episode_id first_color second_color micro_layout \
    pink_x_mm pink_y_mm cyan_x_mm cyan_y_mm <<<"$row"
  echo "Retry ${pair_id}: ${episode_id}, ${first_color}->${second_color}"
  echo "Base marks: ${micro_layout}"
  echo "Pink: (${pink_x_mm}, ${pink_y_mm}) mm; Cyan: (${cyan_x_mm}, ${cyan_y_mm}) mm"
  echo "Origin=safe-workspace center; +x robot-right; +y farther from robot."
  echo "Required: ${first_color}->C1, release; ${second_color}->C2, release."
  read -r -p "恢复同一方块标记和同一机械臂起始姿态后按 Enter；q 退出：" answer
  if [[ "$answer" == "q" || "$answer" == "Q" ]]; then
    echo "Stopped before ${episode_id}; this pair remains rejected_needs_retry."
    exit 3
  fi
  SO101_VLA_RECORD_APPROVED=YES \
  SO101_VLA_EPISODE_ID="$episode_id" \
    commands/record_vla_recovery_episode.sh \
      --episode-id "$episode_id" \
      --retry-rejected \
      --operator-layout-confirmed \
      --execute
done

echo "Both replacement takes were appended. Review and accept/reject the pair again."
