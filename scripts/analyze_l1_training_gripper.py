#!/usr/bin/env python3
"""Analyze gripper cycles and release edges in balanced L1 ACT demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_L1_BALANCED = [5, 23, 26, 32, 38, 41, 43, 47, 48, 49]


def thresholds(values: np.ndarray) -> tuple[float, float, float, float]:
    low = float(np.quantile(values, 0.10))
    high = float(np.quantile(values, 0.90))
    span = high - low
    if span < 1e-6:
        raise ValueError("gripper command has no usable range")
    return low, high, low + 0.35 * span, low + 0.65 * span


def edge_metrics(values: np.ndarray, fps: int) -> dict:
    low, high, lower, upper = thresholds(values)
    state = "low" if values[0] <= lower else "high" if values[0] >= upper else "middle"
    low_to_high = 0
    high_to_low = 0
    close_durations = []
    release_durations = []
    rising_start = None
    falling_start = None

    for index, value in enumerate(values[1:], start=1):
        if state == "low":
            if value <= lower:
                rising_start = None
            elif rising_start is None:
                rising_start = index
            if value >= upper:
                low_to_high += 1
                start = rising_start if rising_start is not None else index
                close_durations.append((index - start) / fps)
                rising_start = None
                state = "high"
        elif state == "high":
            if value >= upper:
                falling_start = None
            elif falling_start is None:
                falling_start = index
            if value <= lower:
                high_to_low += 1
                start = falling_start if falling_start is not None else index
                release_durations.append((index - start) / fps)
                falling_start = None
                state = "low"
        else:
            if value >= upper:
                state = "high"
            elif value <= lower:
                state = "low"

    delta = np.diff(values)
    direction = np.sign(delta)
    direction[np.abs(delta) < 0.25] = 0
    nonzero = direction[direction != 0]
    reversals = int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
    transition_fraction = float(np.mean((values > lower) & (values < upper)))
    return {
        "gripper_min": round(float(values.min()), 4),
        "gripper_max": round(float(values.max()), 4),
        "lower_threshold": round(lower, 4),
        "upper_threshold": round(upper, 4),
        "close_phase_count": low_to_high,
        "release_phase_count": high_to_low,
        "close_edge_durations_s": [round(value, 4) for value in close_durations],
        "release_edge_durations_s": [round(value, 4) for value in release_durations],
        "transition_fraction": round(transition_fraction, 4),
        "direction_reversals_threshold_0_25": reversals,
        "mean_abs_step": round(float(np.mean(np.abs(delta))), 5),
        "max_abs_step": round(float(np.max(np.abs(delta))), 5),
    }


def episode_actions(table: pd.DataFrame, episode_index: int) -> np.ndarray:
    rows = table[table["episode_index"] == episode_index]
    if rows.empty:
        raise ValueError(f"episode {episode_index} has no rows")
    return np.stack(rows["action"].to_numpy()).astype(np.float32)


def aggregate(rows: list[dict]) -> dict:
    releases = [value for row in rows for value in row["release_edge_durations_s"]]
    loaded_releases = [
        value for row in rows for value in row["release_edge_durations_s"][:2]
    ]
    return {
        "episodes": len(rows),
        "episodes_matching_three_close_phase_training_signature": sum(
            row["close_phase_count"] == 3 for row in rows
        ),
        "episodes_with_more_than_three_close_phases": sum(
            row["close_phase_count"] > 3 for row in rows
        ),
        "close_phase_counts": [row["close_phase_count"] for row in rows],
        "all_release_edges_measured_including_final_empty_cycle": len(releases),
        "loaded_release_edges_measured": len(loaded_releases),
        "loaded_release_duration_median_s": round(
            statistics.median(loaded_releases), 4
        ),
        "loaded_release_duration_max_s": round(max(loaded_releases), 4),
        "loaded_release_duration_p90_s": round(
            float(np.quantile(loaded_releases, 0.90)), 4
        ),
        "loaded_release_edges_over_0_5_s": sum(
            value > 0.5 for value in loaded_releases
        ),
        "transition_fraction_median": round(
            statistics.median(row["transition_fraction"] for row in rows), 4
        ),
        "direction_reversals_median": round(
            statistics.median(
                row["direction_reversals_threshold_0_25"] for row in rows
            ),
            2,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--balanced-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/datasets/act_formal_balanced_v1",
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/datasets/act_l1_correction_v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests/act_formal_balanced_episode_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluations/l1_training_gripper_analysis",
    )
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    l1_rows = [row for row in manifest if row["layout_id"] == "L1"]
    balanced_indices = [int(row["episode_id"].rsplit("_", 1)[1]) for row in l1_rows]
    if balanced_indices != EXPECTED_L1_BALANCED:
        raise SystemExit(
            f"FAIL: unexpected balanced L1 mapping: {balanced_indices}"
        )
    if [row["source_dataset"] for row in l1_rows[-3:]] != ["l1_correction"] * 3:
        raise SystemExit("FAIL: balanced episodes 47-49 are not correction-derived")

    balanced = pd.read_parquet(args.balanced_root / "data/chunk-000/file-000.parquet")
    correction = pd.read_parquet(args.correction_root / "data/chunk-000/file-000.parquet")
    fps = 30
    analysis_rows = []
    correction_duplicate_checks = []

    for manifest_row, balanced_index in zip(l1_rows, balanced_indices, strict=True):
        actions = episode_actions(balanced, balanced_index)
        metrics = edge_metrics(actions[:, 5], fps)
        analysis_rows.append(
            {
                "balanced_episode_index": balanced_index,
                "repeat_id": int(manifest_row["repeat_id"]),
                "source_dataset": manifest_row["source_dataset"],
                "source_episode_index": int(manifest_row["source_episode_index"]),
                "frames": len(actions),
                "duration_s": round(len(actions) / fps, 4),
                **metrics,
                "loaded_release_edge_durations_s": metrics[
                    "release_edge_durations_s"
                ][:2],
                "final_empty_release_edge_duration_s": (
                    metrics["release_edge_durations_s"][2]
                    if len(metrics["release_edge_durations_s"]) >= 3
                    else None
                ),
            }
        )

    for offset, balanced_index in enumerate((47, 48, 49)):
        balanced_actions = episode_actions(balanced, balanced_index)
        correction_actions = episode_actions(correction, offset)
        same_shape = balanced_actions.shape == correction_actions.shape
        max_abs_difference = (
            float(np.max(np.abs(balanced_actions - correction_actions)))
            if same_shape
            else None
        )
        correction_duplicate_checks.append(
            {
                "balanced_episode_index": balanced_index,
                "correction_episode_index": offset,
                "same_shape": same_shape,
                "max_abs_action_difference": max_abs_difference,
                "exact_action_duplicate": bool(same_shape and max_abs_difference == 0.0),
            }
        )
    if not all(row["exact_action_duplicate"] for row in correction_duplicate_checks):
        raise SystemExit("FAIL: correction actions do not exactly match balanced copies")

    correction_rows = [row for row in analysis_rows if row["source_dataset"] == "l1_correction"]
    raw_rows = [row for row in analysis_rows if row["source_dataset"] == "raw50"]
    summary = {
        "status": "PASS",
        "hardware_opened": False,
        "datasets_modified": False,
        "fps": fps,
        "balanced_l1_episode_indices": balanced_indices,
        "unique_l1_demonstrations": 10,
        "raw_l1_demonstrations": 7,
        "correction_l1_demonstrations": 3,
        "correction_rows_are_already_in_balanced_10": True,
        "correction_duplicate_checks": correction_duplicate_checks,
        "all_l1": aggregate(analysis_rows),
        "raw_l1": aggregate(raw_rows),
        "correction_l1": aggregate(correction_rows),
        "episodes": analysis_rows,
        "definitions": {
            "low_high_levels": "per-episode gripper action p10 and p90",
            "hysteresis_thresholds": "35% and 65% of the p10-p90 span",
            "release_edge_duration": "time from falling below upper threshold to reaching lower threshold",
            "task_semantic_loaded_grasps": 2,
            "observed_training_signature": "three close phases: Pink loaded grasp, Cyan loaded grasp, then a brief empty-air close/open before final retreat",
        },
        "limitation": "Action traces measure commanded release edges, not the exact video frame when a cube physically detached from the gripper.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    flat_rows = []
    for row in analysis_rows:
        flat_rows.append(
            {
                **{key: value for key, value in row.items() if not isinstance(value, list)},
                "close_edge_durations_s": json.dumps(row["close_edge_durations_s"]),
                "release_edge_durations_s": json.dumps(row["release_edge_durations_s"]),
            }
        )
    pd.DataFrame(flat_rows).to_csv(args.output_dir / "per_episode.csv", index=False)

    fig, axes = plt.subplots(5, 2, figsize=(13, 14), sharey=True)
    for axis, row in zip(axes.flat, analysis_rows, strict=True):
        actions = episode_actions(balanced, row["balanced_episode_index"])
        time_s = np.arange(len(actions)) / fps
        color = "#8c564b" if row["source_dataset"] == "l1_correction" else "#2878b5"
        axis.plot(time_s, actions[:, 5], linewidth=1.0, color=color)
        axis.set_title(
            f"balanced {row['balanced_episode_index']} | {row['source_dataset']} {row['source_episode_index']} | phases={row['close_phase_count']}"
        )
        axis.grid(alpha=0.25)
        axis.set_xlabel("s")
        axis.set_ylabel("gripper")
    fig.suptitle("Balanced L1 training demonstrations: gripper action")
    fig.tight_layout()
    fig.savefig(args.output_dir / "l1_gripper_traces.png", dpi=150)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
