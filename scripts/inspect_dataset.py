#!/usr/bin/env python3
"""Inspect LeRobot v3 dataset metadata without controlling hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"FAIL: missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    camera_features = sorted(
        name for name in features if name.startswith("observation.images.")
    )
    summary = {
        "root": str(root),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "camera_features": camera_features,
        "state": features.get("observation.state"),
        "action": features.get("action"),
        "parquet_files": len(list((root / "data").glob("**/*.parquet"))),
        "video_files": len(list((root / "videos").glob("**/*"))),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    errors = []
    if args.expected_episodes is not None and info.get("total_episodes") != args.expected_episodes:
        errors.append(
            f"expected {args.expected_episodes} episodes, found {info.get('total_episodes')}"
        )
    if sorted(camera_features) != [
        "observation.images.front",
        "observation.images.side",
    ]:
        errors.append(f"expected front+side camera features, found {camera_features}")
    if not summary["parquet_files"]:
        errors.append("no data parquet files")
    if errors:
        print("FAIL: " + "; ".join(errors))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
