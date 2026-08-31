#!/usr/bin/env python3
"""Validate every LeRobot video with FFmpeg when OpenCV lacks AV1 decode."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq


def load_episode_rows(root: Path) -> list[dict]:
    paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if not paths:
        raise SystemExit("no episode metadata parquet files")
    return pa.concat_tables(
        [pq.read_table(path) for path in paths], promote_options="default"
    ).to_pylist()


def probe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}")
    return streams[0]


def decode(path: Path) -> str:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return result.stderr.strip() or f"ffmpeg exit {result.returncode}"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    rows = load_episode_rows(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    cameras: dict[str, dict] = {}

    for camera in args.camera:
        prefix = f"videos/observation.images.{camera}"
        by_file: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in rows:
            key = (int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"]))
            by_file[key].append(row)

        file_results = []
        total_container_frames = 0
        for (chunk, file), file_rows in sorted(by_file.items()):
            path = (
                root
                / "videos"
                / f"observation.images.{camera}"
                / f"chunk-{chunk:03d}"
                / f"file-{file:03d}.mp4"
            )
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            try:
                stream = probe(path)
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            decode_error = decode(path)
            if decode_error:
                errors.append(f"decode failed for {path}: {decode_error}")
            frames = int(stream["nb_frames"])
            duration = float(stream["duration"])
            total_container_frames += frames
            if (int(stream["width"]), int(stream["height"])) != (640, 480):
                errors.append(f"resolution mismatch for {path}")
            if stream["r_frame_rate"] != "30/1":
                errors.append(f"FPS mismatch for {path}: {stream['r_frame_rate']}")
            max_to = max(float(row[f"{prefix}/to_timestamp"]) for row in file_rows)
            if max_to > duration + 1 / 30:
                errors.append(
                    f"episode range exceeds video duration for {path}: {max_to} > {duration}"
                )
            file_results.append(
                {
                    "path": str(path),
                    "codec": stream["codec_name"],
                    "frames": frames,
                    "duration_s": duration,
                    "episodes": len(file_rows),
                    "decode_pass": not decode_error,
                }
            )
        cameras[camera] = {
            "files": len(file_results),
            "container_frames": total_container_frames,
            "dataset_frames": int(info["total_frames"]),
            "unused_container_tail_frames": total_container_frames - int(info["total_frames"]),
            "codecs": sorted({item["codec"] for item in file_results}),
            "all_decode_pass": all(item["decode_pass"] for item in file_results),
            "file_results": file_results,
        }

    payload = {
        "root": str(root),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "cameras": cameras,
        "errors": errors,
        "pass": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "cameras": {
        key: {k: v for k, v in value.items() if k != "file_results"}
        for key, value in cameras.items()
    }}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
