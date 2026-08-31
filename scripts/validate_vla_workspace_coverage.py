#!/usr/bin/env python3
"""Validate the fixed 50-micro-layout SmolVLA workspace coverage table."""

from __future__ import annotations

import argparse
import collections
import csv
import math
from pathlib import Path


EXPECTED_FOCUS = {
    "L1": "left_workspace",
    "L2": "right_workspace",
    "L3": "near_band",
    "L4": "far_band",
    "L5": "cross_workspace",
}
VALID_STATUSES = {"planned_unverified", "verified"}


def coordinates(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(
        float(row[key])
        for key in ("pink_x_norm", "pink_y_norm", "cyan_x_norm", "cyan_y_norm")
    )  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("manifests/vla_workspace_coverage.csv"),
    )
    parser.add_argument("--require-physical-verification", action="store_true")
    args = parser.parse_args()

    with args.coverage.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    errors: list[str] = []
    by_id: dict[str, dict[str, str]] = {}
    layout_counts = collections.Counter()
    geometry_counts = collections.Counter()

    for line, row in enumerate(rows, start=2):
        micro_id = row.get("micro_layout_id", "")
        if not micro_id or micro_id in by_id:
            errors.append(f"line {line}: missing or duplicate micro_layout_id {micro_id!r}")
        else:
            by_id[micro_id] = row
        layout = row.get("layout_id", "")
        layout_counts[layout] += 1
        geometry_counts[row.get("geometry_id", "")] += 1
        if row.get("coverage_focus") != EXPECTED_FOCUS.get(layout):
            errors.append(f"line {line}: coverage_focus does not match {layout}")
        try:
            repeat_id = int(row.get("repeat_id", ""))
            values = coordinates(row)
        except ValueError:
            errors.append(f"line {line}: invalid repeat or coordinate value")
            continue
        if repeat_id not in range(1, 11):
            errors.append(f"line {line}: repeat_id outside 1..10")
        if micro_id != f"{layout}_M{repeat_id:02d}":
            errors.append(f"line {line}: micro_layout_id does not match layout/repeat")
        if any(value < -1.0 or value > 1.0 for value in values):
            errors.append(f"line {line}: normalized coordinate outside [-1, 1]")
        px, py, cx, cy = values
        separation = math.hypot(px - cx, py - cy)
        if separation < 0.40:
            errors.append(f"line {line}: normalized cube separation {separation:.3f} < 0.40")
        status = row.get("physical_status", "")
        if status not in VALID_STATUSES:
            errors.append(f"line {line}: invalid physical_status {status!r}")
        if args.require_physical_verification and status != "verified":
            errors.append(f"line {line}: physical layout is not verified")

    if len(rows) != 50:
        errors.append(f"row count {len(rows)} != 50")
    for layout, focus in EXPECTED_FOCUS.items():
        if layout_counts[layout] != 10:
            errors.append(f"{layout} ({focus}) has {layout_counts[layout]} rows != 10")
        repeats = {
            int(row["repeat_id"])
            for row in rows
            if row.get("layout_id") == layout and row.get("repeat_id", "").isdigit()
        }
        if repeats != set(range(1, 11)):
            errors.append(f"{layout}: repeat IDs are not exactly 1..10")
    if set(layout_counts) != set(EXPECTED_FOCUS):
        errors.append(f"unexpected layout IDs: {sorted(set(layout_counts) - set(EXPECTED_FOCUS))}")
    if len(geometry_counts) != 25 or any(count != 2 for count in geometry_counts.values()):
        errors.append("expected 25 geometries with exactly two color-swapped rows each")

    for micro_id, row in by_id.items():
        partner_id = row.get("color_swap_of", "")
        partner = by_id.get(partner_id)
        if partner is None:
            errors.append(f"{micro_id}: missing color-swap partner {partner_id!r}")
            continue
        if partner.get("color_swap_of") != micro_id:
            errors.append(f"{micro_id}: color-swap link is not reciprocal")
        if partner.get("geometry_id") != row.get("geometry_id"):
            errors.append(f"{micro_id}: color-swap geometry mismatch")
        try:
            px, py, cx, cy = coordinates(row)
            qpx, qpy, qcx, qcy = coordinates(partner)
        except ValueError:
            continue
        if (px, py, cx, cy) != (qcx, qcy, qpx, qpy):
            errors.append(f"{micro_id}: partner coordinates are not an exact color swap")

    print(
        f"rows={len(rows)} geometries={len(geometry_counts)} "
        f"layout_counts={dict(sorted(layout_counts.items()))}"
    )
    if errors:
        print("FAIL")
        for error in errors:
            print("- " + error)
        return 1
    status_counts = collections.Counter(row["physical_status"] for row in rows)
    print(f"physical_status={dict(status_counts)}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
