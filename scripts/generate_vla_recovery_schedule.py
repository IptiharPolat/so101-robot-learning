#!/usr/bin/env python3
"""Generate the deterministic strict-pair SmolVLA recovery schedule."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from project_config import ACT_TASK, CYAN_THEN_PINK_TASK


DEFAULT_SEED = 20260829
SELECTED_LAYOUTS = ("L1_M09", "L2_M09", "L3_M09", "L4_M09", "L5_M09")
ORDERS = {
    "pink_then_cyan": ("pink", "cyan", ACT_TASK),
    "cyan_then_pink": ("cyan", "pink", CYAN_THEN_PINK_TASK),
}
FIELDS = (
    "episode_id",
    "pair_id",
    "base_micro_layout_id",
    "layout_id",
    "supplement_repeat_id",
    "order_type",
    "first_color",
    "second_color",
    "task",
    "pair_position",
    "planned_order",
    "pink_x_norm",
    "pink_y_norm",
    "cyan_x_norm",
    "cyan_y_norm",
    "pink_x_offset_mm",
    "pink_y_offset_mm",
    "cyan_x_offset_mm",
    "cyan_y_offset_mm",
    "status",
    "accepted",
    "initial_both_visible",
    "initial_pose_match",
    "smooth_motion_qc",
    "notes",
)


def read_index(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    index = {row[key]: row for row in rows}
    if len(index) != len(rows):
        raise SystemExit(f"duplicate {key} in {path}")
    return index


def shuffled_pairs(seed: int) -> tuple[list[tuple[str, int]], dict[tuple[str, int], str]]:
    rng = random.Random(seed)
    pairs = [(layout, repeat) for layout in SELECTED_LAYOUTS for repeat in range(1, 5)]

    for _ in range(10_000):
        candidate = pairs.copy()
        rng.shuffle(candidate)
        if all(a[0] != b[0] for a, b in zip(candidate, candidate[1:])):
            pairs = candidate
            break
    else:
        raise SystemExit("could not produce a schedule without adjacent repeated layouts")

    first_order: dict[tuple[str, int], str] = {}
    for layout in SELECTED_LAYOUTS:
        assignments = ["pink_then_cyan"] * 2 + ["cyan_then_pink"] * 2
        rng.shuffle(assignments)
        for repeat, order in zip(range(1, 5), assignments, strict=True):
            first_order[(layout, repeat)] = order
    return pairs, first_order


def validate(rows: list[dict[str, str]]) -> None:
    if len(rows) != 40:
        raise SystemExit(f"expected 40 rows, got {len(rows)}")
    if Counter(row["order_type"] for row in rows) != Counter(
        {"pink_then_cyan": 20, "cyan_then_pink": 20}
    ):
        raise SystemExit("instruction orders are not balanced 20/20")
    if Counter(row["layout_id"] for row in rows) != Counter(
        {f"L{i}": 8 for i in range(1, 6)}
    ):
        raise SystemExit("layout regions are not balanced at eight episodes each")

    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    if len(pairs) != 20:
        raise SystemExit(f"expected 20 pairs, got {len(pairs)}")
    for pair_id, pair_rows in pairs.items():
        if len(pair_rows) != 2:
            raise SystemExit(f"{pair_id} does not contain exactly two rows")
        if {row["order_type"] for row in pair_rows} != set(ORDERS):
            raise SystemExit(f"{pair_id} does not contain opposite instructions")
        planned = sorted(int(row["planned_order"]) for row in pair_rows)
        if planned[1] != planned[0] + 1:
            raise SystemExit(f"{pair_id} rows are not adjacent")

    pair_sequence = [rows[index]["base_micro_layout_id"] for index in range(0, len(rows), 2)]
    if any(a == b for a, b in zip(pair_sequence, pair_sequence[1:])):
        raise SystemExit("consecutive pairs reuse the same layout")

    for layout in SELECTED_LAYOUTS:
        layout_pairs = [pair for pair in pairs.values() if pair[0]["base_micro_layout_id"] == layout]
        first_counts = Counter(
            min(pair, key=lambda row: int(row["planned_order"]))["order_type"] for pair in layout_pairs
        )
        if first_counts != Counter({"pink_then_cyan": 2, "cyan_then_pink": 2}):
            raise SystemExit(f"pair-first order is not balanced within {layout}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("manifests/vla_workspace_coverage.csv"),
    )
    parser.add_argument(
        "--physical-map",
        type=Path,
        default=Path("manifests/vla_workspace_physical_map.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/vla_recovery_episode_schedule.csv"),
    )
    args = parser.parse_args()

    coverage = read_index(args.coverage, "micro_layout_id")
    physical = read_index(args.physical_map, "micro_layout_id")
    missing = [layout for layout in SELECTED_LAYOUTS if layout not in coverage or layout not in physical]
    if missing:
        raise SystemExit(f"selected layouts missing from source maps: {missing}")

    pair_order, first_orders = shuffled_pairs(args.seed)
    rows: list[dict[str, str]] = []
    episode_index = 0
    for layout, repeat in pair_order:
        first_order = first_orders[(layout, repeat)]
        second_order = next(order for order in ORDERS if order != first_order)
        coverage_row = coverage[layout]
        physical_row = physical[layout]
        pair_id = f"recovery_{layout}_R{repeat:02d}"
        for pair_position, order in enumerate((first_order, second_order), start=1):
            first_color, second_color, task = ORDERS[order]
            rows.append(
                {
                    "episode_id": f"svla_fix_{episode_index:03d}",
                    "pair_id": pair_id,
                    "base_micro_layout_id": layout,
                    "layout_id": coverage_row["layout_id"],
                    "supplement_repeat_id": str(repeat),
                    "order_type": order,
                    "first_color": first_color,
                    "second_color": second_color,
                    "task": task,
                    "pair_position": str(pair_position),
                    "planned_order": str(episode_index + 1),
                    "pink_x_norm": coverage_row["pink_x_norm"],
                    "pink_y_norm": coverage_row["pink_y_norm"],
                    "cyan_x_norm": coverage_row["cyan_x_norm"],
                    "cyan_y_norm": coverage_row["cyan_y_norm"],
                    "pink_x_offset_mm": physical_row["pink_x_offset_mm"],
                    "pink_y_offset_mm": physical_row["pink_y_offset_mm"],
                    "cyan_x_offset_mm": physical_row["cyan_x_offset_mm"],
                    "cyan_y_offset_mm": physical_row["cyan_y_offset_mm"],
                    "status": "planned",
                    "accepted": "",
                    "initial_both_visible": "",
                    "initial_pose_match": "",
                    "smooth_motion_qc": "",
                    "notes": "redo_whole_pair_if_either_take_is_rejected",
                }
            )
            episode_index += 1

    validate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={args.output}")
    print(f"seed={args.seed}")
    print("episodes=40 pairs=20 orders=20/20 layouts=8_each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
