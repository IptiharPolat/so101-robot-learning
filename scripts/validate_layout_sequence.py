#!/usr/bin/env python3
"""Compare scheduled ACT layouts with cube positions in episode start frames."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def largest_centroid(mask: np.ndarray) -> tuple[float, float, int]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise RuntimeError("no color component detected")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    x, y = centroids[component]
    return float(x), float(y), area


def detect_positions(frame: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    workspace = np.zeros((height, width), np.uint8)
    workspace[int(height * 0.25) :, int(width * 0.30) :] = 255
    pink = cv2.inRange(hsv, (0, 40, 140), (12, 255, 255))
    cyan = cv2.inRange(hsv, (82, 65, 80), (112, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    pink = cv2.morphologyEx(cv2.bitwise_and(pink, workspace), cv2.MORPH_OPEN, kernel)
    cyan = cv2.morphologyEx(cv2.bitwise_and(cyan, workspace), cv2.MORPH_OPEN, kernel)
    px, py, pink_area = largest_centroid(pink)
    cx, cy, cyan_area = largest_centroid(cyan)
    return np.array([px, py, cx, cy]), {"pink": pink_area, "cyan": cyan_area}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    episode_path = args.root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path).to_pylist()
    if len(manifest) != len(episodes):
        raise RuntimeError(f"manifest={len(manifest)} episodes={len(episodes)}")

    captures: dict[tuple[int, int], cv2.VideoCapture] = {}
    observations = []
    for planned, episode in zip(manifest, episodes, strict=True):
        prefix = f"videos/observation.images.{args.camera}"
        video_key = (
            int(episode[f"{prefix}/chunk_index"]),
            int(episode[f"{prefix}/file_index"]),
        )
        if video_key not in captures:
            chunk_index, file_index = video_key
            video = (
                args.root
                / "videos"
                / f"observation.images.{args.camera}"
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            captures[video_key] = cv2.VideoCapture(str(video))
            if not captures[video_key].isOpened():
                raise RuntimeError(f"cannot open {video}")
        capture = captures[video_key]
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        start = float(episode[f"{prefix}/from_timestamp"])
        end = float(episode[f"{prefix}/to_timestamp"])
        timestamp = start + args.sample_fraction * (end - start)
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot decode episode {episode['episode_index']}")
        vector, areas = detect_positions(frame)
        observations.append(
            {
                "episode_index": int(episode["episode_index"]),
                "episode_id": planned["episode_id"],
                "planned_layout": planned["layout_id"],
                "vector": vector,
                "areas": areas,
            }
        )
    for capture in captures.values():
        capture.release()

    matrix = np.stack([item["vector"] for item in observations])
    scale = np.maximum(np.std(matrix, axis=0), 1.0)
    labels = sorted({item["planned_layout"] for item in observations})
    centers = {
        label: np.median(
            np.stack(
                [item["vector"] for item in observations if item["planned_layout"] == label]
            ),
            axis=0,
        )
        for label in labels
    }
    rows = []
    for item in observations:
        distances = {
            label: float(np.linalg.norm((item["vector"] - center) / scale))
            for label, center in centers.items()
        }
        observed = min(distances, key=distances.get)
        vector = item["vector"]
        rows.append(
            {
                "episode_index": item["episode_index"],
                "episode_id": item["episode_id"],
                "planned_layout": item["planned_layout"],
                "observed_layout": observed,
                "match": observed == item["planned_layout"],
                "pink_x": round(float(vector[0]), 1),
                "pink_y": round(float(vector[1]), 1),
                "cyan_x": round(float(vector[2]), 1),
                "cyan_y": round(float(vector[3]), 1),
                "pink_area": item["areas"]["pink"],
                "cyan_area": item["areas"]["cyan"],
                "nearest_distance": round(distances[observed], 3),
            }
        )

    payload = {
        "episodes": len(rows),
        "planned_counts": dict(Counter(row["planned_layout"] for row in rows)),
        "observed_counts": dict(Counter(row["observed_layout"] for row in rows)),
        "mismatches": [row for row in rows if not row["match"]],
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not payload["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
