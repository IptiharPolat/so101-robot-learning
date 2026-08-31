#!/usr/bin/env python3
"""Quantify action continuity in SmolVLA demonstrations and saved rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ("p50", "p95", "p99", "max")}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def parquet_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").glob("**/*.parquet"))


def load_frames(dataset_root: Path) -> pd.DataFrame:
    files = parquet_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"no data parquet files under {dataset_root}")
    frames = [
        pd.read_parquet(
            path,
            columns=["episode_index", "action", "observation.state"],
        )
        for path in files
    ]
    return pd.concat(frames, ignore_index=True)


def arrays(series: pd.Series) -> np.ndarray:
    result = np.stack([np.asarray(value, dtype=np.float64) for value in series])
    if result.ndim != 2 or result.shape[1] != 6:
        raise ValueError(f"expected [N,6], got {result.shape}")
    return result


def summarize(dataset_root: Path, chunk_size: int | None = None) -> dict:
    data = load_frames(dataset_root)
    deltas: list[np.ndarray] = []
    second_deltas: list[np.ndarray] = []
    target_gaps: list[np.ndarray] = []
    boundary_body_steps: list[np.ndarray] = []
    non_boundary_body_steps: list[np.ndarray] = []
    max_step_modulo_chunk: list[int] = []
    lengths: list[int] = []
    for _, episode in data.groupby("episode_index", sort=False):
        action = arrays(episode["action"])
        state = arrays(episode["observation.state"])
        lengths.append(len(action))
        if len(action) > 1:
            episode_delta = np.diff(action, axis=0)
            deltas.append(episode_delta)
            if chunk_size is not None:
                episode_body_linf = np.max(np.abs(episode_delta[:, :5]), axis=1)
                target_frame = np.arange(1, len(action))
                boundary = target_frame % chunk_size == 0
                boundary_body_steps.append(episode_body_linf[boundary])
                non_boundary_body_steps.append(episode_body_linf[~boundary])
                max_step_modulo_chunk.append(
                    int(target_frame[int(np.argmax(episode_body_linf))] % chunk_size)
                )
        if len(action) > 2:
            second_deltas.append(np.diff(action, n=2, axis=0))
        target_gaps.append(action - state)

    delta = np.concatenate(deltas) if deltas else np.empty((0, 6))
    second_delta = np.concatenate(second_deltas) if second_deltas else np.empty((0, 6))
    target_gap = np.concatenate(target_gaps) if target_gaps else np.empty((0, 6))
    body_linf = np.max(np.abs(delta[:, :5]), axis=1) if len(delta) else np.empty(0)
    target_body_linf = (
        np.max(np.abs(target_gap[:, :5]), axis=1) if len(target_gap) else np.empty(0)
    )

    result = {
        "root": str(dataset_root),
        "episodes": int(data["episode_index"].nunique()),
        "frames": int(len(data)),
        "episode_duration_s_at_30fps": percentiles(np.asarray(lengths, dtype=float) / 30.0),
        "per_frame_absolute_action_delta": {
            name: percentiles(np.abs(delta[:, index]))
            for index, name in enumerate(ACTION_NAMES)
        },
        "body_step_linf": percentiles(body_linf),
        "body_step_gt_5_fraction": float(np.mean(body_linf > 5.0)) if len(body_linf) else 0.0,
        "body_step_gt_10_fraction": float(np.mean(body_linf > 10.0)) if len(body_linf) else 0.0,
        "gripper_step_abs": percentiles(np.abs(delta[:, 5])) if len(delta) else percentiles(np.empty(0)),
        "second_difference_body_linf": percentiles(
            np.max(np.abs(second_delta[:, :5]), axis=1) if len(second_delta) else np.empty(0)
        ),
        "command_minus_state_body_linf": percentiles(target_body_linf),
        "command_minus_state_gripper_abs": percentiles(np.abs(target_gap[:, 5])),
    }
    if chunk_size is not None:
        result["chunk_boundary_analysis"] = {
            "chunk_size": chunk_size,
            "boundary_body_step_linf": percentiles(
                np.concatenate(boundary_body_steps) if boundary_body_steps else np.empty(0)
            ),
            "non_boundary_body_step_linf": percentiles(
                np.concatenate(non_boundary_body_steps) if non_boundary_body_steps else np.empty(0)
            ),
            "episode_max_exactly_at_boundary_fraction": (
                float(np.mean(np.asarray(max_step_modulo_chunk) == 0))
                if max_step_modulo_chunk
                else 0.0
            ),
            "episode_max_modulo_chunk": max_step_modulo_chunk,
        }
    return result


def recent_eval_roots(base: Path) -> Iterable[Path]:
    for path in sorted(base.glob("smolvla_30k_*")):
        if (path / "meta/info.json").is_file() and parquet_files(path):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-root",
        type=Path,
        default=ROOT / "outputs/datasets/vla_pink_cyan_order_clean_v1",
    )
    parser.add_argument(
        "--evaluations-root",
        type=Path,
        default=ROOT / "outputs/evaluations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/smolvla_motion_audit.json",
    )
    args = parser.parse_args()

    report = {
        "units": (
            "SO101 normalized position units: body joints use [-100,100] and gripper uses [0,100] "
            "because robot.use_degrees was not enabled"
        ),
        "interpretation_warning": (
            "These metrics describe recorded target continuity, not Cartesian speed. Future rollouts using "
            "max_relative_target need a recorder fix to persist the clipped action actually sent."
        ),
        "training": summarize(args.training_root.resolve()),
        "evaluations": [],
    }
    for path in recent_eval_roots(args.evaluations_root.resolve()):
        report["evaluations"].append(summarize(path, chunk_size=50))

    if report["evaluations"]:
        evaluations = report["evaluations"]
        report["evaluation_aggregate"] = {
            "datasets": len(evaluations),
            "median_dataset_body_step_p95": float(
                np.median([item["body_step_linf"]["p95"] for item in evaluations])
            ),
            "maximum_saved_body_step": float(
                max(item["body_step_linf"]["max"] for item in evaluations)
            ),
            "datasets_whose_max_step_is_exactly_at_50_step_boundary": int(
                sum(
                    item["chunk_boundary_analysis"][
                        "episode_max_exactly_at_boundary_fraction"
                    ]
                    == 1.0
                    for item in evaluations
                )
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(
        "training_body_step_p95="
        f"{report['training']['body_step_linf']['p95']:.4f} "
        "training_body_step_max="
        f"{report['training']['body_step_linf']['max']:.4f}"
    )
    print(f"saved_eval_datasets={len(report['evaluations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
