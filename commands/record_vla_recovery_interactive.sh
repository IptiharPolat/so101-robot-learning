#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "${1:-}" != "--execute" ]]; then
  python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

rows = list(csv.DictReader(Path("manifests/vla_recovery_episode_schedule.csv").open()))
if len(rows) != 40:
    raise SystemExit(f"expected 40 manifest rows, got {len(rows)}")
for index in range(0, len(rows), 2):
    first, second = rows[index:index + 2]
    if first["pair_id"] != second["pair_id"]:
        raise SystemExit(f"rows {index + 1}-{index + 2} are not one adjacent pair")
    if {first["order_type"], second["order_type"]} != {
        "pink_then_cyan", "cyan_then_pink"
    }:
        raise SystemExit(f"{first['pair_id']} does not contain opposite orders")
print(f"DRY RUN ONLY: {len(rows)} planned episodes / {len(rows) // 2} adjacent pairs")
print("orders:", dict(Counter(row["order_type"] for row in rows)))
for index in range(0, len(rows), 2):
    first, second = rows[index:index + 2]
    print(
        f"{index // 2 + 1:02d}. {first['pair_id']}: "
        f"{first['first_color']}->{first['second_color']}, "
        f"{second['first_color']}->{second['second_color']}"
    )
print("No camera, serial port, robot, dataset, upload, or training process was opened.")
PY
  exit 0
fi

if [[ "${SO101_VLA_RECOVERY_BATCH_APPROVED:-}" != "YES" ]]; then
  echo "Execution refused: SO101_VLA_RECOVERY_BATCH_APPROVED is not YES." >&2
  exit 2
fi

mapfile -t pending_rows < <(
  python3 - <<'PY'
import csv
from pathlib import Path

with Path("manifests/vla_recovery_episode_schedule.csv").open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        if row["status"] == "planned":
            print("\t".join((
                row["episode_id"], row["pair_id"], row["base_micro_layout_id"],
                row["order_type"], row["first_color"], row["second_color"],
                row["pair_position"], row["planned_order"],
                row["pink_x_norm"], row["pink_y_norm"],
                row["cyan_x_norm"], row["cyan_y_norm"],
                row["pink_x_offset_mm"], row["pink_y_offset_mm"],
                row["cyan_x_offset_mm"], row["cyan_y_offset_mm"],
            )))
PY
)

if ((${#pending_rows[@]} == 0)); then
  echo "No planned recovery episodes remain. Run pair QC; do not train yet."
  exit 0
fi

echo "Pending recovery episodes: ${#pending_rows[@]}"
echo "Each adjacent pair must use identical cube marks and arm start pose."
echo "Uploads remain disabled. If either take is bad, stop and redo the whole pair."

for row in "${pending_rows[@]}"; do
  IFS=$'\t' read -r episode_id pair_id micro_layout order_type first_color second_color \
    pair_position planned_order pink_x pink_y cyan_x cyan_y \
    pink_x_mm pink_y_mm cyan_x_mm cyan_y_mm <<<"$row"
  echo "============================================================"
  echo "${planned_order}/40  ${episode_id}  ${pair_id}  ${first_color} -> ${second_color}"
  echo "Base marks: ${micro_layout}"
  echo "Pink: (${pink_x_mm}, ${pink_y_mm}) mm; Cyan: (${cyan_x_mm}, ${cyan_y_mm}) mm"
  echo "Coordinate origin: safe-workspace center; +x robot-right; +y farther from robot."
  echo "Required: ${first_color}->C1, release; ${second_color}->C2, release."
  read -r -p "双相机可用、两方块可见、中心区为空、两臂对齐后按 Enter；q 退出：" answer
  if [[ "$answer" == "q" || "$answer" == "Q" ]]; then
    echo "Stopped before ${episode_id}."
    exit 0
  fi

  SO101_VLA_RECORD_APPROVED=YES \
  SO101_VLA_EPISODE_ID="$episode_id" \
    commands/record_vla_recovery_episode.sh \
      --episode-id "$episode_id" \
      --operator-layout-confirmed \
      --execute

  echo "Saved ${episode_id} locally; no upload was performed."
  if [[ "$pair_position" == "1" ]]; then
    prompt="记住本条是否异常，但仍按 Enter 完成同对第二条；仅紧急情况输入 q："
  else
    prompt="同对两条都合格按 Enter；任一异常输入 q，随后整对判退和重录："
  fi
  read -r -p "$prompt" answer
  if [[ "$answer" == "q" || "$answer" == "Q" ]]; then
    echo "Stopped after ${episode_id}; do not continue until this whole pair is resolved."
    exit 0
  fi
done

echo "All planned recovery rows were invoked. Run whole-pair QC before derivation or training."
