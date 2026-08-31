#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "${SO101_VLA_BATCH_APPROVED:-}" != "YES" ]]; then
  echo "Execution refused: set SO101_VLA_BATCH_APPROVED=YES after reviewing the schedule."
  exit 2
fi

blocked_rows="$(python3 - <<'PY'
import csv
from pathlib import Path

with Path("manifests/vla_episode_schedule.csv").open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        if row["status"] == "rejected_needs_retry":
            print(f'{row["episode_id"]} ({row["micro_layout_id"]})')
PY
)"
if [[ -n "$blocked_rows" ]]; then
  echo "Batch blocked by rejected rows requiring correction:"
  printf '%s\n' "$blocked_rows"
  echo "Run the row-specific --retry-rejected command first; do not skip it."
  exit 2
fi

mapfile -t pending_rows < <(
  python3 - <<'PY'
import csv
from pathlib import Path

coverage = {}
with Path("manifests/vla_workspace_coverage.csv").open(newline="", encoding="utf-8") as stream:
    for item in csv.DictReader(stream):
        coverage[item["micro_layout_id"]] = item

with Path("manifests/vla_episode_schedule.csv").open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        if row["status"] == "planned":
            item = coverage[row["micro_layout_id"]]
            print("\t".join((
                row["episode_id"], row["micro_layout_id"], row["order_type"],
                row["first_color"], row["second_color"], row["planned_order"],
                item["pink_x_norm"], item["pink_y_norm"],
                item["cyan_x_norm"], item["cyan_y_norm"],
            )))
PY
)

if ((${#pending_rows[@]} == 0)); then
  echo "No planned VLA episodes remain."
  exit 0
fi

echo "Pending planned episodes: ${#pending_rows[@]}"
echo "Canonical schedule: manifests/vla_episode_schedule.csv"
echo "Each row requires physical placement and reset; upload remains disabled."
echo

for row in "${pending_rows[@]}"; do
  IFS=$'\t' read -r episode_id micro_layout order_type first_color second_color planned_order pink_x pink_y cyan_x cyan_y <<<"$row"
  echo "============================================================"
  echo "${planned_order}/100  ${episode_id}  ${micro_layout}  ${first_color} -> ${second_color}"
  echo "Normalized pickup points: Pink (${pink_x}, ${pink_y}), Cyan (${cyan_x}, ${cyan_y})"
  printf 'Approx. offsets from safe-workspace center (34x40 cm): Pink (%+.0f, %+.0f) mm, Cyan (%+.0f, %+.0f) mm\n' \
    "$(awk "BEGIN {print ${pink_x}*170}")" "$(awk "BEGIN {print ${pink_y}*200}")" \
    "$(awk "BEGIN {print ${cyan_x}*170}")" "$(awk "BEGIN {print ${cyan_y}*200}")"
  echo "Required: place ${first_color} at C1, release; place ${second_color} at C2, release."
  read -r -p "摆好该布局、中心区为空、两臂对齐后按 Enter 开始；输入 q 退出：" answer
  if [[ "$answer" == "q" || "$answer" == "Q" ]]; then
    echo "Stopped by operator before ${episode_id}."
    exit 0
  fi

  SO101_VLA_RECORD_APPROVED=YES \
  SO101_VLA_EPISODE_ID="$episode_id" \
    commands/record_vla_episode.sh \
      --episode-id "$episode_id" \
      --operator-layout-confirmed \
      --execute

  echo "Saved ${episode_id} locally; no upload was performed."
  read -r -p "检查该条画面和顺序；复位到下一布局后按 Enter 继续，输入 q 退出：" answer
  if [[ "$answer" == "q" || "$answer" == "Q" ]]; then
    echo "Stopped by operator after ${episode_id}."
    exit 0
  fi
done

echo "All currently planned VLA episodes have been invoked. Run pair QC before merging or training."
