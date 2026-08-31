#!/usr/bin/env python3
"""Validate LeRobot state/action shapes, finite values, cameras, and episode count."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def flatten(value: Any):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def shape_of(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(v) for v in value.shape)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child = shape_of(value[0])
        if any(shape_of(item) != child for item in value[1:]):
            return (len(value), -1)
        return (len(value),) + child
    return ()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--camera", action="append", default=[])
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required") from exc

    root = args.root.expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if args.expected_episodes is not None and info.get("total_episodes") != args.expected_episodes:
        errors.append(
            f"episodes {info.get('total_episodes')} != {args.expected_episodes}"
        )

    features = info.get("features", {})
    for camera in args.camera:
        key = f"observation.images.{camera}"
        feature = features.get(key)
        if not feature:
            errors.append(f"missing camera feature {key}")
            continue
        if feature.get("shape") != [480, 640, 3]:
            errors.append(f"{key} shape {feature.get('shape')} != [480, 640, 3]")
        camera_dir = root / "videos"
        if not any(camera in str(path) for path in camera_dir.glob("**/*") if path.is_file()):
            errors.append(f"no video file found for {camera}")

    expected_shapes = {
        "observation.state": tuple(features.get("observation.state", {}).get("shape", [])),
        "action": tuple(features.get("action", {}).get("shape", [])),
    }
    if expected_shapes["observation.state"] != (6,):
        errors.append(f"metadata observation.state shape is {expected_shapes['observation.state']}")
    if expected_shapes["action"] != (6,):
        errors.append(f"metadata action shape is {expected_shapes['action']}")

    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        errors.append("no parquet data files")
    row_count = 0
    seen_shapes = {key: set() for key in expected_shapes}
    nonfinite = {key: 0 for key in expected_shapes}
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        missing = set(expected_shapes) - available
        if missing:
            errors.append(f"{path}: missing columns {sorted(missing)}")
            continue
        for batch in parquet.iter_batches(columns=list(expected_shapes), batch_size=2048):
            row_count += batch.num_rows
            rows = batch.to_pydict()
            for key, values in rows.items():
                for value in values:
                    seen_shapes[key].add(shape_of(value))
                    for scalar in flatten(value):
                        try:
                            if not math.isfinite(float(scalar)):
                                nonfinite[key] += 1
                        except (TypeError, ValueError):
                            nonfinite[key] += 1

    for key, expected in expected_shapes.items():
        if seen_shapes[key] and seen_shapes[key] != {expected}:
            errors.append(f"{key} observed shapes {sorted(seen_shapes[key])} != {expected}")
        if nonfinite[key]:
            errors.append(f"{key} has {nonfinite[key]} NaN/Inf/non-numeric values")
    if info.get("total_frames") is not None and row_count != info["total_frames"]:
        errors.append(f"parquet rows {row_count} != metadata frames {info['total_frames']}")

    report = {
        "root": str(root),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "parquet_rows": row_count,
        "expected_shapes": {k: list(v) for k, v in expected_shapes.items()},
        "observed_shapes": {k: [list(v) for v in sorted(s)] for k, s in seen_shapes.items()},
        "nonfinite": nonfinite,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PASS" if not errors else "FAIL")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
