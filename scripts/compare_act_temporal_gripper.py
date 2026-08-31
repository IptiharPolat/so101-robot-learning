#!/usr/bin/env python3
"""Compare ACT gripper commands with temporal ensembling on/off offline."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import copy
import hashlib
import json
from pathlib import Path
import statistics

import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

from project_config import ACT_TASK, load_config, require


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    "ACT30K-L1-TEMPORAL-001": PROJECT_ROOT
    / "outputs/evaluations/act_30k_l1_temporal_diag_001",
    "ACT30K-L1-TEMPORAL-002": PROJECT_ROOT
    / "outputs/evaluations/act_30k_l1_temporal_diag_002",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray, fraction: float) -> float:
    return float(np.quantile(values, fraction))


def active_frame_count(actions: np.ndarray, fps: int) -> int:
    """Keep through one second after the last meaningful recorded action change."""
    if len(actions) < 2:
        return len(actions)
    changing = np.flatnonzero(np.max(np.abs(np.diff(actions, axis=0)), axis=1) > 0.05)
    if len(changing) == 0:
        return min(len(actions), fps)
    return min(len(actions), int(changing[-1]) + 2 + fps)


def hysteresis_phases(values: np.ndarray) -> dict:
    low = percentile(values, 0.10)
    high = percentile(values, 0.90)
    span = high - low
    if span < 1e-6:
        return {
            "low": low,
            "high": high,
            "low_to_high_phases": 0,
            "high_to_low_phases": 0,
            "transition_fraction": 0.0,
        }
    lower = low + 0.35 * span
    upper = low + 0.65 * span
    state = "low" if values[0] <= lower else "high" if values[0] >= upper else "middle"
    low_to_high = 0
    high_to_low = 0
    for value in values[1:]:
        if value >= upper and state != "high":
            if state == "low":
                low_to_high += 1
            state = "high"
        elif value <= lower and state != "low":
            if state == "high":
                high_to_low += 1
            state = "low"
    transition_fraction = float(np.mean((values > lower) & (values < upper)))
    return {
        "low": low,
        "high": high,
        "lower_threshold": lower,
        "upper_threshold": upper,
        "low_to_high_phases": low_to_high,
        "high_to_low_phases": high_to_low,
        "transition_fraction": transition_fraction,
    }


def signal_metrics(values: np.ndarray) -> dict:
    phases = hysteresis_phases(values)
    delta = np.diff(values)
    direction = np.sign(delta)
    direction[np.abs(delta) < 0.05] = 0
    nonzero_direction = direction[direction != 0]
    reversals = (
        int(np.sum(nonzero_direction[1:] != nonzero_direction[:-1]))
        if len(nonzero_direction) > 1
        else 0
    )
    return {
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "p10": round(percentile(values, 0.10), 4),
        "p90": round(percentile(values, 0.90), 4),
        "mean_abs_step": round(float(np.mean(np.abs(delta))), 5),
        "max_abs_step": round(float(np.max(np.abs(delta))), 5),
        "direction_reversals": reversals,
        "low_to_high_phases": phases["low_to_high_phases"],
        "high_to_low_phases": phases["high_to_low_phases"],
        "transition_fraction": round(float(phases["transition_fraction"]), 4),
        "lower_threshold": round(float(phases.get("lower_threshold", phases["low"])), 4),
        "upper_threshold": round(float(phases.get("upper_threshold", phases["high"])), 4),
    }


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    return capture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/act_eval_temporal_diagnostic.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluations/act_gripper_temporal_offline_compare",
    )
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is unavailable")
    config = load_config(args.config)
    policy_source = require(config, "policy")
    snapshot = Path(
        snapshot_download(
            repo_id=str(require(policy_source, "repo_id")),
            revision=str(require(policy_source, "revision")),
            local_files_only=True,
        )
    )
    actual_sha = sha256(snapshot / "model.safetensors")
    if actual_sha != str(require(policy_source, "weight_sha256")):
        raise SystemExit(f"FAIL: model SHA mismatch: {actual_sha}")

    policy_config = PreTrainedConfig.from_pretrained(
        snapshot,
        local_files_only=True,
        cli_overrides=[
            "--device=cuda",
            "--use_amp=true",
            "--n_action_steps=1",
        ],
    )
    if not isinstance(policy_config, ACTConfig):
        raise SystemExit(f"FAIL: expected ACTConfig, got {type(policy_config).__name__}")
    policy_config.pretrained_path = str(snapshot)
    policy = ACTPolicy.from_pretrained(
        snapshot,
        config=policy_config,
        local_files_only=True,
    ).to("cuda")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(snapshot),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_summaries = []

    for evaluation_id, root in TRIALS.items():
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        fps = int(info["fps"])
        frame_table = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
        recorded_actions = np.stack(frame_table["action"].to_numpy()).astype(np.float32)
        states = np.stack(frame_table["observation.state"].to_numpy()).astype(np.float32)
        frame_count = active_frame_count(recorded_actions, fps)
        if args.max_frames is not None:
            frame_count = min(frame_count, args.max_frames)

        front_capture = open_video(
            root / "videos/observation.images.front/chunk-000/file-000.mp4"
        )
        side_capture = open_video(
            root / "videos/observation.images.side/chunk-000/file-000.mp4"
        )
        ensembler = ACTTemporalEnsembler(0.01, policy_config.chunk_size)
        raw_actions = []
        ensemble_actions = []

        try:
            for frame_index in range(frame_count):
                front_ok, front_bgr = front_capture.read()
                side_ok, side_bgr = side_capture.read()
                if not front_ok or not side_ok:
                    raise RuntimeError(f"video ended early at frame {frame_index}")
                raw_observation = {
                    "observation.state": states[frame_index].copy(),
                    "observation.images.front": cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB),
                    "observation.images.side": cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB),
                }
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda")
                    if policy_config.use_amp
                    else nullcontext(),
                ):
                    observation = prepare_observation_for_inference(
                        copy(raw_observation),
                        device,
                        task=ACT_TASK,
                        robot_type="so_follower",
                    )
                    observation = preprocessor(observation)
                    chunk = policy.predict_action_chunk(observation)
                    raw = postprocessor(chunk[:, 0].clone())
                    ensembled = postprocessor(ensembler.update(chunk).clone())
                raw_actions.append(raw.squeeze(0).numpy())
                ensemble_actions.append(ensembled.squeeze(0).numpy())
                if (frame_index + 1) % 500 == 0:
                    print(f"{evaluation_id}: processed {frame_index + 1}/{frame_count}", flush=True)
        finally:
            front_capture.release()
            side_capture.release()

        raw_array = np.stack(raw_actions)
        ensemble_array = np.stack(ensemble_actions)
        recorded_active = recorded_actions[:frame_count]
        output = pd.DataFrame(
            {
                "frame_index": np.arange(frame_count),
                "timestamp_s": np.arange(frame_count) / fps,
                "recorded_gripper": recorded_active[:, 5],
                "offline_no_ensemble_gripper": raw_array[:, 5],
                "offline_temporal_gripper": ensemble_array[:, 5],
            }
        )
        output.to_csv(args.output_dir / f"{evaluation_id.lower()}_gripper.csv", index=False)

        temporal_reproduction_mae = float(
            np.mean(np.abs(ensemble_array - recorded_active))
        )
        temporal_reproduction_gripper_mae = float(
            np.mean(np.abs(ensemble_array[:, 5] - recorded_active[:, 5]))
        )
        arm_raw_step = np.mean(np.abs(np.diff(raw_array[:, :5], axis=0)))
        arm_temporal_step = np.mean(np.abs(np.diff(ensemble_array[:, :5], axis=0)))
        trial_summaries.append(
            {
                "evaluation_id": evaluation_id,
                "source_frames": len(frame_table),
                "active_frames_compared": frame_count,
                "active_duration_s": round(frame_count / fps, 3),
                "recorded_temporal": signal_metrics(recorded_active[:, 5]),
                "offline_no_ensemble": signal_metrics(raw_array[:, 5]),
                "offline_temporal_0_01": signal_metrics(ensemble_array[:, 5]),
                "temporal_vs_recorded_all_action_mae": round(temporal_reproduction_mae, 5),
                "temporal_vs_recorded_gripper_mae": round(
                    temporal_reproduction_gripper_mae, 5
                ),
                "arm_mean_abs_step_no_ensemble": round(float(arm_raw_step), 5),
                "arm_mean_abs_step_temporal": round(float(arm_temporal_step), 5),
                "arm_step_reduction_fraction": (
                    round(float(1 - arm_temporal_step / arm_raw_step), 4)
                    if arm_raw_step > 0
                    else 0.0
                ),
                "gripper_no_ensemble_vs_temporal_mae": round(
                    float(np.mean(np.abs(raw_array[:, 5] - ensemble_array[:, 5]))), 5
                ),
            }
        )

    summary = {
        "status": "PASS",
        "hardware_opened": False,
        "actions_sent": 0,
        "datasets_modified": False,
        "policy_revision": require(policy_source, "revision"),
        "weight_sha256": actual_sha,
        "device": torch.cuda.get_device_name(0),
        "use_amp": True,
        "n_action_steps": 1,
        "compared_temporal_coefficients": [None, 0.01],
        "method": "one predicted chunk per recorded frame; compare chunk[0] with ACTTemporalEnsembler(0.01)",
        "trials": trial_summaries,
        "limitation": "Counterfactual actions are evaluated on observations generated by the temporal policy; this is not a no-ensemble robot rollout.",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
