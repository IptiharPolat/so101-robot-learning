#!/usr/bin/env python3
"""Offline structural, media, and motion audit for recovery-v2 20K rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_frames(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet data under {root}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def stack(series: pd.Series) -> np.ndarray:
    values = np.stack([np.asarray(value, dtype=np.float64) for value in series])
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"expected [N,6], got {values.shape}")
    return values


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def first_sustained(values: np.ndarray, threshold: float, frames: int = 10) -> float | None:
    above = values > threshold
    if len(above) < frames:
        return None
    hits = np.flatnonzero(
        np.convolve(above.astype(np.int64), np.ones(frames, dtype=np.int64), mode="valid")
        == frames
    )
    return None if not len(hits) else float(hits[0] / 30.0)


def active_segments(values: np.ndarray, threshold: float = 10.0, min_frames: int = 5) -> list[list[float]]:
    segments: list[list[float]] = []
    start: int | None = None
    for index, active in enumerate(values > threshold):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(values) - 1):
            end = index if not active else index + 1
            if end - start >= min_frames:
                segments.append([start / 30.0, end / 30.0])
            start = None
    return segments


def decode_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = black = overbright = duplicates = 0
    means: list[float] = []
    previous: np.ndarray | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        means.append(mean)
        black += int(mean < 5.0)
        overbright += int(mean > 250.0)
        duplicates += int(previous is not None and np.array_equal(previous, frame))
        previous = frame
        decoded += 1
    capture.release()
    return {
        "path": str(path),
        "fps": fps,
        "width": width,
        "height": height,
        "expected_frames": expected,
        "decoded_frames": decoded,
        "black_frames": black,
        "overbright_frames": overbright,
        "exact_consecutive_duplicate_frames": duplicates,
        "luma_mean": float(np.mean(means)),
        "luma_p01": percentile(np.asarray(means), 1),
        "luma_p99": percentile(np.asarray(means), 99),
    }


def summarize(root: Path) -> dict:
    frames = load_frames(root)
    action = stack(frames["action"])
    state = stack(frames["observation.state"])
    body_displacement = np.max(np.abs(state[:, :5] - state[0, :5]), axis=1)
    body_step = np.max(np.abs(np.diff(action[:, :5], axis=0)), axis=1)
    target_frames = np.arange(1, len(action))
    boundary = target_frames % 50 == 0
    gripper_step = np.abs(np.diff(action[:, 5]))
    videos = {}
    for camera in ("front", "side"):
        paths = sorted((root / "videos" / f"observation.images.{camera}").glob("**/*.mp4"))
        if len(paths) != 1:
            raise RuntimeError(f"{root}: expected one {camera} video, got {len(paths)}")
        videos[camera] = decode_video(paths[0])
    return {
        "name": root.name,
        "kind": "night" if "_night_" in root.name else "formal",
        "frames": int(len(frames)),
        "duration_s": float(len(frames) / 30.0),
        "task_index_values": sorted(int(value) for value in frames["task_index"].unique()),
        "finite_action": bool(np.isfinite(action).all()),
        "finite_state": bool(np.isfinite(state).all()),
        "initial_state": state[0].tolist(),
        "first_sustained_body_displacement_s": {
            str(threshold): first_sustained(body_displacement, threshold)
            for threshold in (1.0, 2.0, 5.0)
        },
        "body_action_step": {
            "p95": percentile(body_step, 95),
            "max": percentile(body_step, 100),
            "boundary_p95": percentile(body_step[boundary], 95),
            "non_boundary_p95": percentile(body_step[~boundary], 95),
        },
        "gripper_action_step": {
            "p95": percentile(gripper_step, 95),
            "max": percentile(gripper_step, 100),
        },
        "gripper_high_segments": active_segments(state[:, 5]),
        "videos": videos,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/evaluations/smolvla_recovery_smooth_screen"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/smolvla_recovery20k_rollout_audit.json"),
    )
    args = parser.parse_args()
    roots = sorted(path for path in args.root.resolve().glob("recovery20k_*") if path.is_dir())
    results = [summarize(root) for root in roots]
    formal = [item for item in results if item["kind"] == "formal"]
    pairs = []
    for layout in range(1, 6):
        prefix = f"recovery20k_l{layout}_m09_"
        pink = [item for item in formal if item["name"].startswith(prefix + "p2c_")]
        cyan = [item for item in formal if item["name"].startswith(prefix + "c2p_")]
        if not pink or not cyan:
            continue
        cyan_initial = np.asarray(cyan[0]["initial_state"])
        comparisons = []
        for item in pink:
            delta = np.abs(np.asarray(item["initial_state"]) - cyan_initial)
            comparisons.append(
                {
                    "pink_trial": item["name"],
                    "cyan_trial": cyan[0]["name"],
                    "initial_state_linf_all": float(delta.max()),
                    "initial_state_linf_body": float(delta[:5].max()),
                    "initial_gripper_difference": float(delta[5]),
                    "passes_training_pair_pose_tolerance_le_5": bool(delta.max() <= 5.0),
                }
            )
        pairs.append({"layout": f"L{layout}_M09", "comparisons": comparisons})
    payload = {
        "scope": "offline only; no robot, recording, upload, or training",
        "interpretation_limits": [
            "Joint-space statistics do not directly measure Cartesian gripper height.",
            "Operator task outcomes must be joined separately; this report does not infer success from motion alone.",
            "A 50-frame modulo boundary is a proxy for synchronous action chunk refresh points.",
        ],
        "rollouts": results,
        "pair_initial_state_checks": pairs,
        "aggregate": {
            "datasets": len(results),
            "formal_datasets": len(formal),
            "night_datasets": len(results) - len(formal),
            "all_finite": all(item["finite_action"] and item["finite_state"] for item in results),
            "all_videos_fully_decoded": all(
                camera["decoded_frames"] == camera["expected_frames"]
                for item in results
                for camera in item["videos"].values()
            ),
            "formal_duration_median_s": float(np.median([item["duration_s"] for item in formal])),
            "formal_duration_max_s": float(max(item["duration_s"] for item in formal)),
            "formal_boundary_body_step_p95_median": float(
                np.median([item["body_action_step"]["boundary_p95"] for item in formal])
            ),
            "formal_non_boundary_body_step_p95_median": float(
                np.median([item["body_action_step"]["non_boundary_p95"] for item in formal])
            ),
            "formal_gripper_high_segment_count_median": float(
                np.median([len(item["gripper_high_segments"]) for item in formal])
            ),
            "formal_gripper_high_segment_count_max": int(
                max(len(item["gripper_high_segments"]) for item in formal)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
