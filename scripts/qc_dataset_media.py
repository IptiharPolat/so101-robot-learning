#!/usr/bin/env python3
"""Decode LeRobot v3 videos and build per-episode visual QC contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


SAMPLE_FRACTIONS = (0.05, 0.25, 0.50, 0.75, 0.95)


def load_episode_ranges(root: Path, camera: str) -> list[dict[str, float | int]]:
    columns = [
        "episode_index",
        "length",
        f"videos/observation.images.{camera}/chunk_index",
        f"videos/observation.images.{camera}/file_index",
        f"videos/observation.images.{camera}/from_timestamp",
        f"videos/observation.images.{camera}/to_timestamp",
    ]
    paths = sorted((root / "meta" / "episodes").glob("**/file-*.parquet"))
    if not paths:
        raise RuntimeError(f"no episode metadata parquet files under {root / 'meta/episodes'}")
    # LeRobot v3 may shard episode metadata into one parquet per episode or
    # consolidate several episodes into a shard. Read all shards so media QC
    # covers the complete dataset rather than only file-000.
    tables = [pq.read_table(path, columns=columns) for path in paths]
    import pyarrow as pa

    table = pa.concat_tables(tables, promote_options="default").to_pylist()
    return [
        {
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "chunk_index": int(
                row[f"videos/observation.images.{camera}/chunk_index"]
            ),
            "file_index": int(row[f"videos/observation.images.{camera}/file_index"]),
            "from_timestamp": float(
                row[f"videos/observation.images.{camera}/from_timestamp"]
            ),
            "to_timestamp": float(
                row[f"videos/observation.images.{camera}/to_timestamp"]
            ),
        }
        for row in table
    ]


def make_sheet(samples: list[tuple[int, np.ndarray]], fraction: float) -> np.ndarray:
    tile_width, tile_height = 256, 192
    columns = 10
    rows = (len(samples) + columns - 1) // columns
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 24, np.uint8)
    for position, (episode_index, frame) in enumerate(samples):
        tile = cv2.resize(frame, (tile_width, tile_height))
        cv2.rectangle(tile, (0, 0), (tile_width, 25), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"ep {episode_index:02d} @ {fraction:.0%}",
            (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(position, columns)
        y0, x0 = row * tile_height, column * tile_width
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
    return sheet


def inspect_camera(root: Path, camera: str, output_dir: Path) -> dict:
    episodes = load_episode_ranges(root, camera)
    samples: dict[float, list[tuple[int, np.ndarray]]] = {
        fraction: [] for fraction in SAMPLE_FRACTIONS
    }
    episode_groups: dict[tuple[int, int], list[dict]] = {}
    for episode in episodes:
        key = (int(episode["chunk_index"]), int(episode["file_index"]))
        episode_groups.setdefault(key, []).append(episode)

    decoded_frames = 0
    expected_frames = 0
    black_frames = 0
    overbright_frames = 0
    exact_duplicate_frames = 0
    means: list[float] = []
    contrast: list[float] = []
    blur_scores: list[float] = []
    fps = width = height = None
    video_paths = []
    for (chunk_index, file_index), file_episodes in sorted(episode_groups.items()):
        video = (
            root
            / "videos"
            / f"observation.images.{camera}"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        video_paths.append(str(video))
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {video}")
        file_fps = float(capture.get(cv2.CAP_PROP_FPS))
        file_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        file_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps is None:
            fps, width, height = file_fps, file_width, file_height
        elif (file_fps, file_width, file_height) != (fps, width, height):
            raise RuntimeError(f"inconsistent video properties: {video}")
        expected_frames += int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_targets: dict[int, tuple[int, float]] = {}
        for episode in file_episodes:
            start = float(episode["from_timestamp"])
            end = float(episode["to_timestamp"])
            for fraction in SAMPLE_FRACTIONS:
                timestamp = start + fraction * (end - start)
                sample_targets[round(timestamp * file_fps)] = (
                    int(episode["episode_index"]),
                    fraction,
                )

        file_frame_index = 0
        previous: np.ndarray | None = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean())
            std = float(gray.std())
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            means.append(mean)
            contrast.append(std)
            blur_scores.append(blur)
            black_frames += mean < 5.0
            overbright_frames += mean > 250.0
            if previous is not None and np.array_equal(frame, previous):
                exact_duplicate_frames += 1
            previous = frame
            if file_frame_index in sample_targets:
                episode_index, fraction = sample_targets[file_frame_index]
                samples[fraction].append((episode_index, frame.copy()))
            file_frame_index += 1
        decoded_frames += file_frame_index
        capture.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths = []
    for fraction, phase_samples in samples.items():
        phase_samples.sort(key=lambda item: item[0])
        path = output_dir / f"{camera}_{round(fraction * 100):02d}pct.jpg"
        if not cv2.imwrite(str(path), make_sheet(phase_samples, fraction)):
            raise RuntimeError(f"failed to write {path}")
        sheet_paths.append(str(path))

    return {
        "camera": camera,
        "videos": video_paths,
        "fps": fps,
        "width": width,
        "height": height,
        "metadata_frame_count": expected_frames,
        "decoded_frame_count": decoded_frames,
        "black_frames": black_frames,
        "overbright_frames": overbright_frames,
        "exact_consecutive_duplicate_frames": exact_duplicate_frames,
        "luma_mean_min": min(means),
        "luma_mean_max": max(means),
        "luma_mean_p01": float(np.percentile(means, 1)),
        "luma_mean_p99": float(np.percentile(means, 99)),
        "contrast_p01": float(np.percentile(contrast, 1)),
        "laplacian_variance_p01": float(np.percentile(blur_scores, 1)),
        "sample_count_by_fraction": {
            str(fraction): len(phase_samples)
            for fraction, phase_samples in samples.items()
        },
        "contact_sheets": sheet_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    results = [inspect_camera(args.root, camera, args.output_dir) for camera in args.camera]
    payload = {"root": str(args.root.resolve()), "cameras": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
