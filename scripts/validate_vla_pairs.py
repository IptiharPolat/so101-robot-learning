#!/usr/bin/env python3
"""Validate strict paired and balanced VLA schedule/collection manifests."""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

from project_config import ACT_TASK, CYAN_THEN_PINK_TASK


EXPECTED = {
    "pink_then_cyan": ("pink", "cyan", ACT_TASK),
    "cyan_then_pink": ("cyan", "pink", CYAN_THEN_PINK_TASK),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("manifests/vla_workspace_coverage.csv"),
    )
    parser.add_argument("--require-accepted", action="store_true")
    parser.add_argument("--expected-per-order", type=int, default=50)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    with args.coverage.open(newline="", encoding="utf-8") as stream:
        coverage_rows = list(csv.DictReader(stream))
    coverage = {row.get("micro_layout_id", ""): row for row in coverage_rows}
    supplement_schema = {
        "base_micro_layout_id",
        "supplement_repeat_id",
    }.issubset(fieldnames)
    errors = []
    counts = collections.Counter(row.get("order_type") for row in rows)
    pairs: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    layouts = collections.Counter()
    episode_ids = collections.Counter()
    micro_layout_counts = collections.Counter()
    planned_orders = []
    for line, row in enumerate(rows, start=2):
        episode_ids[row.get("episode_id", "")] += 1
        try:
            planned_orders.append(int(row.get("planned_order", "")))
        except ValueError:
            errors.append(f"line {line}: invalid planned_order")
        order = row.get("order_type", "")
        if order not in EXPECTED:
            errors.append(f"line {line}: invalid order_type {order!r}")
            continue
        first, second, task = EXPECTED[order]
        if (row.get("first_color"), row.get("second_color"), row.get("task")) != (
            first,
            second,
            task,
        ):
            errors.append(f"line {line}: colors/task do not match {order}")
        pair_id = row.get("pair_id", "")
        if not pair_id:
            errors.append(f"line {line}: missing pair_id")
        else:
            pairs[pair_id].append(row)
        micro_layout_id = (
            row.get("base_micro_layout_id", "")
            if supplement_schema
            else row.get("micro_layout_id", "")
        )
        repeat_value = (
            row.get("supplement_repeat_id", "")
            if supplement_schema
            else row.get("repeat_id", "")
        )
        micro_layout_counts[micro_layout_id] += 1
        coverage_row = coverage.get(micro_layout_id)
        if coverage_row is None:
            errors.append(f"line {line}: unknown micro_layout_id {micro_layout_id!r}")
        else:
            if row.get("layout_id") != coverage_row.get("layout_id"):
                errors.append(f"line {line}: layout differs from coverage table")
            if not supplement_schema and repeat_value != coverage_row.get("repeat_id"):
                errors.append(f"line {line}: repeat differs from coverage table")
            expected_pair_id = (
                f"recovery_{micro_layout_id}_R{int(repeat_value):02d}"
                if supplement_schema and repeat_value.isdigit()
                else micro_layout_id
            )
            if pair_id != expected_pair_id:
                errors.append(
                    f"line {line}: pair_id {pair_id!r} != {expected_pair_id!r}"
                )
        layouts[(row.get("layout_id"), order)] += 1
        try:
            repeat_id = int(repeat_value)
        except ValueError:
            errors.append(f"line {line}: invalid repeat_id")
        else:
            max_repeat = args.expected_per_order // 5
            if repeat_id not in range(1, max_repeat + 1):
                errors.append(
                    f"line {line}: repeat_id outside 1..{max_repeat}"
                )
        if args.require_accepted and row.get("accepted", "").strip().lower() not in {
            "true",
            "1",
            "yes",
        }:
            errors.append(f"line {line}: episode is not accepted")

    for order in EXPECTED:
        if counts[order] != args.expected_per_order:
            errors.append(
                f"{order}: {counts[order]} rows != {args.expected_per_order}"
            )
    if len(rows) != 2 * args.expected_per_order:
        errors.append(f"total rows {len(rows)} != {2 * args.expected_per_order}")
    duplicates = sorted(key for key, count in episode_ids.items() if not key or count != 1)
    if duplicates:
        errors.append(f"missing or duplicate episode_id values: {duplicates}")
    if planned_orders != list(range(1, len(rows) + 1)):
        errors.append("planned_order must be exactly 1..N in file order")
    pair_first_orders = collections.Counter(
        rows[index].get("order_type") for index in range(0, len(rows), 2)
    )
    expected_pair_first = args.expected_per_order // 2
    if pair_first_orders != collections.Counter(
        {
            "pink_then_cyan": expected_pair_first,
            "cyan_then_pink": expected_pair_first,
        }
    ):
        errors.append(
            "pair first-order balance must be "
            f"{expected_pair_first}/{expected_pair_first}, got {dict(pair_first_orders)}"
        )
    pair_layouts = [rows[index].get("layout_id") for index in range(0, len(rows), 2)]
    if any(
        pair_layouts[index] == pair_layouts[index - 1]
        for index in range(1, len(pair_layouts))
    ):
        errors.append("consecutive instruction pairs must use different regions")
    if not supplement_schema and set(micro_layout_counts) != set(coverage):
        errors.append("schedule micro-layout IDs do not exactly match coverage table")
    if supplement_schema and not set(micro_layout_counts) <= set(coverage):
        errors.append("supplement micro-layout IDs are not a subset of coverage table")
    expected_per_micro_layout = len(rows) // len(micro_layout_counts)
    for micro_layout_id, count in sorted(micro_layout_counts.items()):
        if count != expected_per_micro_layout:
            errors.append(
                f"{micro_layout_id}: schedule count {count} != {expected_per_micro_layout}"
            )
    for pair_id, pair_rows in sorted(pairs.items()):
        pair_orders = {row["order_type"] for row in pair_rows}
        if len(pair_rows) != 2 or pair_orders != set(EXPECTED):
            errors.append(f"{pair_id}: expected exactly two opposite-order rows")
        if len({row.get("layout_id") for row in pair_rows}) != 1:
            errors.append(f"{pair_id}: layout mismatch")
        repeat_key = "supplement_repeat_id" if supplement_schema else "repeat_id"
        if len({row.get(repeat_key) for row in pair_rows}) != 1:
            errors.append(f"{pair_id}: repeat mismatch")
        indices = sorted(rows.index(row) for row in pair_rows)
        if len(indices) == 2 and indices[1] != indices[0] + 1:
            errors.append(f"{pair_id}: paired rows are not adjacent")

    expected_layouts = {"L1", "L2", "L3", "L4", "L5"}
    observed_layouts = {layout for layout, _ in layouts}
    if observed_layouts != expected_layouts:
        errors.append(
            f"layouts {sorted(observed_layouts)} != {sorted(expected_layouts)}"
        )
    expected_per_layout_order = args.expected_per_order // len(expected_layouts)
    for layout in sorted(expected_layouts):
        for order in EXPECTED:
            count = layouts[(layout, order)]
            if count != expected_per_layout_order:
                errors.append(
                    f"{layout} {order}: {count} rows != {expected_per_layout_order}"
                )

    print(f"rows={len(rows)} pairs={len(pairs)} counts={dict(counts)}")
    for key in sorted(layouts):
        print(f"layout_order_count {key[0]} {key[1]} {layouts[key]}")
    if errors:
        print("FAIL")
        for error in errors:
            print("- " + error)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
