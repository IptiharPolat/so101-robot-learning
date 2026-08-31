#!/usr/bin/env python3
"""Validate that a dataset uses exactly the configured canonical task label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_config import ACT_TASK, CYAN_THEN_PINK_TASK, load_config, require


def read_tasks(root: Path) -> list[str]:
    parquet = root / "meta" / "tasks.parquet"
    if parquet.is_file():
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("pyarrow is required to read tasks.parquet") from exc
        table = pq.read_table(parquet)
        frame = table.to_pandas()
        for column in ("task", "tasks"):
            if column in frame.columns:
                return [str(value) for value in frame[column].tolist()]
        # LeRobot v3 may store task strings as the pandas index and task_index
        # as the only visible data column.
        if frame.index.name is not None or not hasattr(frame.index, "start"):
            values = [str(value) for value in frame.index.tolist()]
            if values:
                return values
        raise SystemExit(f"No task strings in {parquet}: {table.column_names}")

    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.is_file():
        values = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            values.append(str(row.get("task", row.get("tasks", ""))))
        return values
    raise SystemExit(f"Neither tasks.parquet nor tasks.jsonl exists under {root / 'meta'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-file", default="configs/act_experiment.yaml")
    parser.add_argument(
        "--task-key",
        default="task",
        help="Dotted config key for the one expected task label.",
    )
    args = parser.parse_args()

    expected = require(load_config(args.task_file), args.task_key)
    if expected not in {ACT_TASK, CYAN_THEN_PINK_TASK}:
        raise SystemExit("Configured task is not one of the two canonical labels")
    observed = sorted(set(read_tasks(args.root.expanduser().resolve())))
    print(json.dumps({"expected": expected, "observed": observed}, indent=2))
    if observed != [expected]:
        print("FAIL: task labels do not exactly match")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
