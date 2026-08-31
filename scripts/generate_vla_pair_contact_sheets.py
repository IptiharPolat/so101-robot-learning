#!/usr/bin/env python3
"""Generate labeled initial-frame contact sheets for manual VLA pair review."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import pyarrow.parquet as pq


ORDER_LEFT = "pink_then_cyan"
ORDER_RIGHT = "cyan_then_pink"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/datasets/vla_pink_cyan_order_clean_v1"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("manifests/vla_clean_selection.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/vla_pair_initial_contact_sheets"),
    )
    parser.add_argument("--pairs-per-sheet", type=int, default=5)
    parser.add_argument("--offset-s", type=float, default=0.2)
    parser.add_argument("--camera", choices=("front", "side"), default="front")
    return parser.parse_args()


def read_frame(video_path: Path, timestamp_s: float) -> "cv2.typing.MatLike":
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode {video_path} at {timestamp_s:.3f}s")
    return frame


def labeled_frame(frame: "cv2.typing.MatLike", label: str) -> "cv2.typing.MatLike":
    width = 480
    height = 360
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    bar_height = 34
    canvas = cv2.copyMakeBorder(
        resized,
        bar_height,
        2,
        2,
        2,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    cv2.putText(
        canvas,
        label,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    selection_path = args.selection.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episode_meta = pq.read_table(episode_meta_path).to_pandas().set_index("episode_index")

    with selection_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pairs: dict[str, dict[str, int]] = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["order_type"]] = int(row["merged_episode"])

    ordered_pairs = list(pairs.items())
    for sheet_index in range(math.ceil(len(ordered_pairs) / args.pairs_per_sheet)):
        start = sheet_index * args.pairs_per_sheet
        group = ordered_pairs[start : start + args.pairs_per_sheet]
        pair_rows = []
        for pair_id, episode_by_order in group:
            frames = []
            for order, short_label in ((ORDER_LEFT, "PINK->CYAN"), (ORDER_RIGHT, "CYAN->PINK")):
                episode_index = episode_by_order[order]
                meta = episode_meta.loc[episode_index]
                video_key = f"videos/observation.images.{args.camera}"
                file_index = int(meta[f"{video_key}/file_index"])
                timestamp = float(meta[f"{video_key}/from_timestamp"]) + args.offset_s
                video_path = (
                    root
                    / f"videos/observation.images.{args.camera}/chunk-000"
                    / f"file-{file_index:03d}.mp4"
                )
                frame = read_frame(video_path, timestamp)
                frames.append(
                    labeled_frame(frame, f"{pair_id} | {short_label} | merged_ep={episode_index}")
                )
            pair_rows.append(cv2.hconcat(frames))
        sheet = cv2.vconcat(pair_rows)
        output_path = output_dir / f"pairs_{start + 1:02d}_{start + len(group):02d}.jpg"
        if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"Failed to write {output_path}")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
