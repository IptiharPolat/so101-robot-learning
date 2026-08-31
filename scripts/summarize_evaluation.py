#!/usr/bin/env python3
"""Summarize trial annotations and enforce the strict ACT pilot gate."""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path


BOOLEAN_FIELDS = [
    "sequence_complete",
    "pink_first",
    "pink_released_before_cyan",
    "cyan_grasped_after_pink",
    "both_inside_center",
    "c1_c2_clear",
    "front_video_usable",
    "side_video_usable",
    "colors_clear",
    "no_video_fault",
    "task_exact",
    "state_action_valid",
    "two_gripper_cycles",
    "no_excessive_pause",
    "strategy_consistent",
    "duration_sufficient",
    "no_collision_or_workspace_issue",
]
TRUE = {"1", "true", "yes", "y"}
FALSE = {"0", "false", "no", "n"}


def parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in TRUE:
        return True
    if normalized in FALSE:
        return False
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot-gate", action="store_true")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    missing_columns = [name for name in BOOLEAN_FIELDS if name not in (rows[0] if rows else {})]
    failures = collections.defaultdict(list)
    incomplete = []
    for row_number, row in enumerate(rows, start=2):
        episode = row.get("episode_id") or f"line_{row_number}"
        for field in BOOLEAN_FIELDS:
            value = parse_bool(row.get(field, ""))
            if value is False:
                failures[field].append(episode)
            elif value is None:
                incomplete.append(f"{episode}:{field}")

    ready = bool(rows) and not missing_columns and not incomplete
    passed = ready and not failures
    if args.pilot_gate:
        passed = passed and len(rows) == 12

    lines = [
        "# ACT pilot QC summary",
        "",
        f"- Episodes: {len(rows)}",
        f"- Annotation complete: {'yes' if ready else 'no'}",
        f"- Gate: {'PASS' if passed else 'FAIL/NOT READY'}",
        "",
        "## Failed checks",
        "",
    ]
    if failures:
        for field, episodes in sorted(failures.items()):
            lines.append(f"- `{field}`: {', '.join(episodes)}")
    else:
        lines.append("- None among completed annotations.")
    if missing_columns or incomplete:
        lines += ["", "## Missing annotations", ""]
        if missing_columns:
            lines.append("- Missing columns: " + ", ".join(missing_columns))
        if incomplete:
            lines.append("- Missing/invalid values: " + ", ".join(incomplete[:100]))
    lines += [
        "",
        "A PASS still requires human confirmation that layout reset, C1/C2 placement, "
        "camera quality, collisions, and task execution match the physical trial.",
        "",
    ]
    report = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
