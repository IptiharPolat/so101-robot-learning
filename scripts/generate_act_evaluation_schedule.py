#!/usr/bin/env python3
"""Generate the fixed 32-trial ACT evaluation schedule."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


LAYOUTS = ["L1", "L2", "L3", "L4", "L5", "U1", "U2", "U3"]


def generate(seed: int) -> list[dict[str, object]]:
    base = [
        {
            "layout_id": layout,
            "repeat_id": repeat,
            "seen_layout": str(layout.startswith("L")).lower(),
        }
        for layout in LAYOUTS
        for repeat in range(1, 5)
    ]
    rng = random.Random(seed)
    for _ in range(10000):
        rows = base.copy()
        rng.shuffle(rows)
        if all(
            not (
                rows[index]["layout_id"] == rows[index - 1]["layout_id"]
                == rows[index - 2]["layout_id"]
            )
            for index in range(2, len(rows))
        ):
            break
    else:
        raise RuntimeError("could not generate a schedule without triples")

    output = []
    for index, row in enumerate(rows, start=1):
        output.append(
            {
                "evaluation_id": f"ACT30K-FORMAL-{index:03d}",
                "planned_order": index,
                **row,
                "checkpoint_step": 30000,
                "control_fps": 25,
                "n_action_steps": 1,
                "temporal_ensemble_coeff": 0.01,
                "status": "planned",
                "accepted": "",
                "notes": "",
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output", type=Path, default=Path("manifests/act_evaluation_schedule.csv")
    )
    args = parser.parse_args()
    rows = generate(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output} with seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
