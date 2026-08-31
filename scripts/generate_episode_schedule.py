#!/usr/bin/env python3
"""Generate deterministic ACT pilot/formal or paired VLA schedules."""

from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path

from project_config import ACT_TASK, CYAN_THEN_PINK_TASK


ACT_FIELDS = [
    "episode_id",
    "layout_id",
    "repeat_id",
    "first_color",
    "second_color",
    "task",
    "planned_order",
    "status",
    "accepted",
    "notes",
]
VLA_FIELDS = [
    "episode_id",
    "layout_id",
    "micro_layout_id",
    "repeat_id",
    "order_type",
    "first_color",
    "second_color",
    "task",
    "pair_id",
    "planned_order",
    "status",
    "accepted",
    "notes",
]
COVERAGE_FIELDS = [
    "micro_layout_id",
    "layout_id",
    "repeat_id",
    "geometry_id",
    "coverage_focus",
    "pink_x_norm",
    "pink_y_norm",
    "cyan_x_norm",
    "cyan_y_norm",
    "color_swap_of",
    "physical_status",
    "notes",
]


# Coordinates are normalized within the bounding rectangle of the
# operator-confirmed safe pickup area:
# x=-1/+1 is robot-left/right and y=-1/+1 is near/far from the robot.
# Physical mapping must reject points that intersect the gray center target area.
REGION_GEOMETRIES = {
    "L1": (
        "left_workspace",
        [
            ((-0.82, -0.62), (-0.38, -0.48)),
            ((-0.78, -0.18), (-0.34, 0.08)),
            ((-0.80, 0.30), (-0.36, 0.58)),
            ((-0.68, -0.68), (-0.30, 0.62)),
            ((-0.88, 0.62), (-0.42, -0.62)),
        ],
    ),
    "L2": (
        "right_workspace",
        [
            ((0.82, -0.62), (0.38, -0.48)),
            ((0.78, -0.18), (0.34, 0.08)),
            ((0.80, 0.30), (0.36, 0.58)),
            ((0.68, -0.68), (0.30, 0.62)),
            ((0.88, 0.62), (0.42, -0.62)),
        ],
    ),
    "L3": (
        "near_band",
        [
            ((-0.72, -0.72), (-0.28, -0.42)),
            ((-0.28, -0.68), (0.20, -0.38)),
            ((0.18, -0.70), (0.64, -0.44)),
            ((-0.62, -0.30), (0.02, -0.62)),
            ((-0.10, -0.34), (0.58, -0.72)),
        ],
    ),
    "L4": (
        "far_band",
        [
            ((-0.72, 0.72), (-0.28, 0.42)),
            ((-0.28, 0.68), (0.20, 0.38)),
            ((0.18, 0.70), (0.64, 0.44)),
            ((-0.62, 0.30), (0.02, 0.62)),
            ((-0.10, 0.34), (0.58, 0.72)),
        ],
    ),
    "L5": (
        "cross_workspace",
        [
            ((-0.72, -0.55), (0.68, -0.35)),
            ((-0.72, 0.55), (0.68, 0.35)),
            ((-0.62, -0.62), (0.58, 0.62)),
            ((-0.62, 0.62), (0.58, -0.62)),
            ((-0.82, 0.05), (0.78, -0.05)),
        ],
    ),
}


def max_layout_run(rows: list[dict[str, object]]) -> int:
    best = current = 0
    previous = None
    for row in rows:
        layout = row["layout_id"]
        current = current + 1 if layout == previous else 1
        best = max(best, current)
        previous = layout
    return best


def act_pilot(seed: int) -> list[dict[str, object]]:
    base = [
        {"layout_id": layout, "repeat_id": repeat_id}
        for layout in ("L1", "L2", "L3")
        for repeat_id in range(1, 5)
    ]
    rng = random.Random(seed)
    for _ in range(10_000):
        rows = [dict(row) for row in base]
        rng.shuffle(rows)
        if max_layout_run(rows) <= 2:
            break
    else:
        raise RuntimeError("Could not generate a balanced ACT ordering")

    result = []
    for index, row in enumerate(rows):
        result.append(
            {
                "episode_id": f"act_pilot_{index:03d}",
                **row,
                "first_color": "pink",
                "second_color": "cyan",
                "task": ACT_TASK,
                "planned_order": index + 1,
                "status": "planned",
                "accepted": "",
                "notes": "",
            }
        )
    return result


def act_formal(seed: int) -> list[dict[str, object]]:
    base = [
        {"layout_id": layout, "repeat_id": repeat_id}
        for layout in ("L1", "L2", "L3", "L4", "L5")
        for repeat_id in range(1, 11)
    ]
    rng = random.Random(seed)
    for _ in range(10_000):
        rows = [dict(row) for row in base]
        rng.shuffle(rows)
        if max_layout_run(rows) <= 2:
            break
    else:
        raise RuntimeError("Could not generate a balanced formal ACT ordering")

    result = []
    for index, row in enumerate(rows):
        result.append(
            {
                "episode_id": f"act_formal_{index:03d}",
                **row,
                "first_color": "pink",
                "second_color": "cyan",
                "task": ACT_TASK,
                "planned_order": index + 1,
                "status": "planned",
                "accepted": "",
                "notes": "",
            }
        )
    return result


def vla_workspace_coverage() -> list[dict[str, object]]:
    result = []
    for layout_id, (focus, geometries) in REGION_GEOMETRIES.items():
        for geometry_index, (point_a, point_b) in enumerate(geometries, start=1):
            geometry_id = f"{layout_id}_G{geometry_index:02d}"
            first_repeat = geometry_index * 2 - 1
            ids = (
                f"{layout_id}_M{first_repeat:02d}",
                f"{layout_id}_M{first_repeat + 1:02d}",
            )
            for offset, (pink, cyan) in enumerate(
                ((point_a, point_b), (point_b, point_a))
            ):
                repeat_id = first_repeat + offset
                result.append(
                    {
                        "micro_layout_id": ids[offset],
                        "layout_id": layout_id,
                        "repeat_id": repeat_id,
                        "geometry_id": geometry_id,
                        "coverage_focus": focus,
                        "pink_x_norm": f"{pink[0]:.2f}",
                        "pink_y_norm": f"{pink[1]:.2f}",
                        "cyan_x_norm": f"{cyan[0]:.2f}",
                        "cyan_y_norm": f"{cyan[1]:.2f}",
                        "color_swap_of": ids[1 - offset],
                        "physical_status": "planned_unverified",
                        "notes": "",
                    }
                )
    return result


def load_coverage(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def vla_formal(
    seed: int, coverage: list[dict[str, str]]
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    base_pairs = [dict(row) for row in coverage]
    for _ in range(10_000):
        pairs = [dict(row) for row in base_pairs]
        rng.shuffle(pairs)
        if all(
            pairs[index]["layout_id"] != pairs[index - 1]["layout_id"]
            for index in range(1, len(pairs))
        ):
            break
    else:
        raise RuntimeError("Could not avoid consecutive VLA pairs from one region")

    first_orders = ["pink_then_cyan"] * 25 + ["cyan_then_pink"] * 25
    rng.shuffle(first_orders)
    order_specs = {
        "pink_then_cyan": ("pink", "cyan", ACT_TASK),
        "cyan_then_pink": ("cyan", "pink", CYAN_THEN_PINK_TASK),
    }
    result = []
    for micro_layout, first_order in zip(pairs, first_orders, strict=True):
        layout = micro_layout["layout_id"]
        repeat_id = int(micro_layout["repeat_id"])
        micro_layout_id = micro_layout["micro_layout_id"]
        second_order = (
            "cyan_then_pink" if first_order == "pink_then_cyan" else "pink_then_cyan"
        )
        orders = [
            (first_order, *order_specs[first_order]),
            (second_order, *order_specs[second_order]),
        ]
        pair_id = micro_layout_id
        for order_type, first, second, task in orders:
            index = len(result)
            result.append(
                {
                    "episode_id": f"vla_{index:03d}",
                    "layout_id": layout,
                    "micro_layout_id": micro_layout_id,
                    "repeat_id": repeat_id,
                    "order_type": order_type,
                    "first_color": first,
                    "second_color": second,
                    "task": task,
                    "pair_id": pair_id,
                    "planned_order": index + 1,
                    "status": "planned",
                    "accepted": "",
                    "notes": "",
                }
            )
    return result


def render(rows: list[dict[str, object]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("act-pilot", "act-formal", "vla-coverage", "vla-formal"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("manifests/vla_workspace_coverage.csv"),
    )
    args = parser.parse_args()

    if args.mode == "act-pilot":
        content = render(act_pilot(args.seed), ACT_FIELDS)
    elif args.mode == "act-formal":
        content = render(act_formal(args.seed), ACT_FIELDS)
    elif args.mode == "vla-coverage":
        content = render(vla_workspace_coverage(), COVERAGE_FIELDS)
    else:
        content = render(vla_formal(args.seed, load_coverage(args.coverage)), VLA_FIELDS)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
