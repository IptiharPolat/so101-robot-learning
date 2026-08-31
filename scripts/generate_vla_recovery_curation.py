#!/usr/bin/env python3
"""Freeze the old clean-v1 pair selection used to build recovery v2."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXCLUDE = {
    "L2_M05": "confirmed_color_swap_across_instructions",
    "L2_M08": "confirmed_color_swap_across_instructions",
    "L4_M08": "confirmed_pink_material_shift_across_instructions",
    "L3_M04": "cyan_not_identifiable_in_initial_front_view",
    "L1_M02": "cyan_not_identifiable_in_initial_front_view",
    "L3_M02": "cyan_not_identifiable_in_initial_front_view",
    "L5_M02": "cyan_not_identifiable_in_initial_front_view",
    "L5_M01": "pink_not_identifiable_in_initial_front_view",
    "L5_M06": "cyan_not_identifiable_in_initial_front_view",
}

PHYSICAL_MISMATCHES = {"L2_M05", "L2_M08", "L4_M08"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("manifests/vla_clean_selection.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/vla_recovery_existing_curation.csv"),
    )
    args = parser.parse_args()

    with args.selection.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pairs: dict[str, dict[str, str]] = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["order_type"]] = row["merged_episode"]
    if len(pairs) != 50:
        raise SystemExit(f"expected 50 source pairs, got {len(pairs)}")

    output_rows = []
    for pair_id, episodes in pairs.items():
        if set(episodes) != {"pink_then_cyan", "cyan_then_pink"}:
            raise SystemExit(f"incomplete source pair: {pair_id}")
        excluded = pair_id in EXCLUDE
        if pair_id in PHYSICAL_MISMATCHES:
            status = "exclude_confirmed_pair_mismatch"
        elif excluded:
            status = "exclude_initial_front_visibility_failure"
        else:
            status = "retain"
        output_rows.append(
            {
                "pair_id": pair_id,
                "pink_then_cyan_merged_episode": episodes["pink_then_cyan"],
                "cyan_then_pink_merged_episode": episodes["cyan_then_pink"],
                "include_in_recovery_v2": str(not excluded).lower(),
                "status": status,
                "reason": EXCLUDE.get(
                    pair_id,
                    "operator_order_review_passed; not_excluded_by_frozen_visibility_review",
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    included = sum(row["include_in_recovery_v2"] == "true" for row in output_rows)
    print(f"output={args.output}")
    print(f"pairs=50 retained={included} excluded={len(output_rows) - included}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
