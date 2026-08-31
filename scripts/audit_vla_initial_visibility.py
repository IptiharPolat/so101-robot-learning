#!/usr/bin/env python3
"""Quantify early front-camera pink/cyan visibility for every merged VLA pair."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]

# Contact-sheet review found that the small red arm/cable component passes the
# broad pink HSV mask in this pair even though the pink cube itself is hidden.
MANUAL_REVIEW_EXCLUSIONS = {
    "L5_M01": "pink_cube_hidden; compact_red_arm_or_cable_false_positive",
}


def largest_component(mask: np.ndarray) -> tuple[int, float | None, float | None]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0, None, None
    candidates = []
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if width < 5 or height < 5:
            continue
        aspect = width / height
        fill = area / (width * height)
        # Cube projections are compact; this rejects thin red cables and noise.
        if not 0.25 <= aspect <= 4.0 or fill < 0.20:
            continue
        candidates.append(component)
    if not candidates:
        return 0, None, None
    component = max(candidates, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    x, y = centroids[component]
    return area, float(x), float(y)


def color_components(frame: np.ndarray) -> dict[str, tuple[int, float | None, float | None]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    workspace = np.zeros((height, width), np.uint8)
    # Keep edge/near-arm layouts; compact-component filtering rejects cables.
    workspace[int(height * 0.15) :, int(width * 0.03) :] = 255
    masks = {
        "pink": cv2.inRange(hsv, (0, 40, 140), (12, 255, 255)),
        "cyan": cv2.inRange(hsv, (82, 65, 80), (112, 255, 255)),
    }
    kernel = np.ones((3, 3), np.uint8)
    return {
        name: largest_component(
            cv2.morphologyEx(cv2.bitwise_and(mask, workspace), cv2.MORPH_OPEN, kernel)
        )
        for name, mask in masks.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs/datasets/vla_pink_cyan_order_clean_v1",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "manifests/vla_clean_selection.csv",
    )
    parser.add_argument("--camera", default="front")
    parser.add_argument("--offset-s", type=float, action="append", default=[])
    parser.add_argument(
        "--visible-area-threshold",
        type=int,
        default=120,
        help="Largest HSV component area in native 640x480 pixels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/vla_initial_visibility_audit.json",
    )
    args = parser.parse_args()
    offsets = args.offset_s or [0.2, 0.5, 0.8]
    if any(value < 0 for value in offsets):
        raise SystemExit("offsets must be non-negative")

    dataset_root = args.dataset_root.resolve()
    episode_files = sorted((dataset_root / "meta/episodes").glob("**/*.parquet"))
    if not episode_files:
        raise SystemExit(f"no episode metadata under {dataset_root}")
    episode_rows = []
    for path in episode_files:
        episode_rows.extend(pq.read_table(path).to_pylist())
    episodes = {int(row["episode_index"]): row for row in episode_rows}

    with args.selection.open(newline="", encoding="utf-8") as stream:
        selection = list(csv.DictReader(stream))
    if len(selection) != 100:
        raise SystemExit(f"expected 100 selection rows, got {len(selection)}")

    captures: dict[Path, cv2.VideoCapture] = {}
    rows = []
    try:
        for selected in selection:
            episode_index = int(selected["merged_episode"])
            episode = episodes[episode_index]
            prefix = f"videos/observation.images.{args.camera}"
            chunk_index = int(episode[f"{prefix}/chunk_index"])
            file_index = int(episode[f"{prefix}/file_index"])
            video_path = (
                dataset_root
                / "videos"
                / f"observation.images.{args.camera}"
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            capture = captures.get(video_path)
            if capture is None:
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"cannot open {video_path}")
                captures[video_path] = capture
            start = float(episode[f"{prefix}/from_timestamp"])
            end = float(episode[f"{prefix}/to_timestamp"])

            samples = []
            for offset in offsets:
                timestamp = min(start + offset, max(start, end - 1 / 30))
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"cannot decode episode {episode_index} at {timestamp:.3f}s"
                    )
                components = color_components(frame)
                samples.append(
                    {
                        "offset_s": offset,
                        "pink_area": components["pink"][0],
                        "pink_x": components["pink"][1],
                        "pink_y": components["pink"][2],
                        "cyan_area": components["cyan"][0],
                        "cyan_x": components["cyan"][1],
                        "cyan_y": components["cyan"][2],
                    }
                )
            max_area = {
                color: max(sample[f"{color}_area"] for sample in samples)
                for color in ("pink", "cyan")
            }
            visible = {
                color: max_area[color] >= args.visible_area_threshold
                for color in ("pink", "cyan")
            }
            rows.append(
                {
                    "pair_id": selected["pair_id"],
                    "order_type": selected["order_type"],
                    "merged_episode": episode_index,
                    "max_early_area": max_area,
                    "visible": visible,
                    "both_visible": all(visible.values()),
                    "samples": samples,
                }
            )
    finally:
        for capture in captures.values():
            capture.release()

    pair_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pair_rows[row["pair_id"]].append(row)
    pairs = []
    for pair_id, members in pair_rows.items():
        if len(members) != 2:
            raise SystemExit(f"{pair_id}: expected two orders, got {len(members)}")
        pairs.append(
            {
                "pair_id": pair_id,
                "both_orders_both_colors_visible": all(row["both_visible"] for row in members),
                "orders": sorted(members, key=lambda row: row["order_type"]),
            }
        )

    flagged = [pair for pair in pairs if not pair["both_orders_both_colors_visible"]]
    frozen_exclusions = sorted(
        set(pair["pair_id"] for pair in flagged) | set(MANUAL_REVIEW_EXCLUSIONS)
    )
    payload = {
        "dataset_root": str(dataset_root),
        "camera": args.camera,
        "offsets_s": offsets,
        "visible_area_threshold": args.visible_area_threshold,
        "interpretation": (
            "HSV area is a screening signal, not an automatic deletion decision. "
            "The frozen exclusion set combines the screen with documented contact-sheet review."
        ),
        "episodes": len(rows),
        "pairs": len(pairs),
        "flagged_pair_count": len(flagged),
        "flagged_pair_ids": [pair["pair_id"] for pair in flagged],
        "manual_review_exclusions": MANUAL_REVIEW_EXCLUSIONS,
        "frozen_exclusion_pair_count": len(frozen_exclusions),
        "frozen_exclusion_pair_ids": frozen_exclusions,
        "pair_results": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"episodes={len(rows)} pairs={len(pairs)} flagged_pairs={len(flagged)}")
    print("flagged=" + ",".join(payload["flagged_pair_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
