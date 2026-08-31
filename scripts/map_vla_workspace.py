#!/usr/bin/env python3
"""Map normalized VLA micro-layouts to candidate millimeter offsets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from project_config import load_config, require


FIELDS = [
    "micro_layout_id",
    "layout_id",
    "repeat_id",
    "pink_x_offset_mm",
    "pink_y_offset_mm",
    "cyan_x_offset_mm",
    "cyan_y_offset_mm",
    "target_clearance_status",
    "camera_status",
    "reach_status",
    "physical_status",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rig.local.yaml")
    parser.add_argument(
        "--coverage", default="manifests/vla_workspace_coverage.csv", type=Path
    )
    parser.add_argument(
        "--output", default="manifests/vla_workspace_physical_map.csv", type=Path
    )
    args = parser.parse_args()

    rig = load_config(args.config)
    width = float(require(rig, "workspace_mapping.width_mm"))
    depth = float(require(rig, "workspace_mapping.depth_mm"))
    if width <= 0 or depth <= 0:
        raise SystemExit("Workspace width/depth must be positive")

    with args.coverage.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    output_rows = []
    separations = []
    for row in source_rows:
        px = float(row["pink_x_norm"]) * width / 2
        py = float(row["pink_y_norm"]) * depth / 2
        cx = float(row["cyan_x_norm"]) * width / 2
        cy = float(row["cyan_y_norm"]) * depth / 2
        separations.append(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)
        output_rows.append(
            {
                "micro_layout_id": row["micro_layout_id"],
                "layout_id": row["layout_id"],
                "repeat_id": row["repeat_id"],
                "pink_x_offset_mm": f"{px:.1f}",
                "pink_y_offset_mm": f"{py:.1f}",
                "cyan_x_offset_mm": f"{cx:.1f}",
                "cyan_y_offset_mm": f"{cy:.1f}",
                "target_clearance_status": "pending",
                "camera_status": "pending",
                "reach_status": "pending",
                "physical_status": "planned_unverified",
                "notes": "origin=safe_workspace_center",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"rows={len(output_rows)} width_mm={width:g} depth_mm={depth:g} "
        f"min_cube_separation_mm={min(separations):.1f}"
    )
    print(f"wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
