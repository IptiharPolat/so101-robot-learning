#!/usr/bin/env python3
"""Measure ACT observation-plus-inference timing without sending robot actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.utils.control_utils import predict_action

from project_config import ACT_TASK, load_config, require


MOTOR_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(np.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def camera_config(source: dict) -> OpenCVCameraConfig:
    return OpenCVCameraConfig(
        index_or_path=Path(require(source, "index_or_path")),
        width=int(require(source, "width")),
        height=int(require(source, "height")),
        fps=int(require(source, "fps")),
        fourcc=str(require(source, "fourcc")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig", default="configs/rig.local.yaml")
    parser.add_argument("--config", default="configs/act_eval_temporal_diagnostic.yaml")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--target-fps", type=int, default=30, choices=(25, 30))
    args = parser.parse_args()

    if os.environ.get("SO101_TIMING_APPROVED") != "YES":
        raise SystemExit("FAIL: set SO101_TIMING_APPROVED=YES only after explicit approval")
    if args.warmup < 1 or args.iterations < 30:
        raise SystemExit("FAIL: use at least 1 warmup and 30 measured iterations")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is unavailable")

    rig = load_config(args.rig)
    config = load_config(args.config)
    policy_source = require(config, "policy")
    if rig.get("audit_status") != "VERIFIED":
        raise SystemExit("FAIL: rig audit_status is not VERIFIED")
    if require(policy_source, "n_action_steps") != 1:
        raise SystemExit("FAIL: n_action_steps must be 1")
    if float(require(policy_source, "temporal_ensemble_coeff")) != 0.01:
        raise SystemExit("FAIL: temporal_ensemble_coeff must be 0.01")
    if not bool(require(policy_source, "use_amp")):
        raise SystemExit("FAIL: this timing check requires the preflighted CUDA AMP mode")

    robot_source = require(rig, "robot")
    calibration_file = Path(require(robot_source, "calibration_file"))
    required_paths = [
        Path(require(robot_source, "port")),
        Path(require(rig, "cameras.front.index_or_path")),
        Path(require(rig, "cameras.side.index_or_path")),
        calibration_file,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: missing hardware/calibration paths: {missing}")

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
            "--temporal_ensemble_coeff=0.01",
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

    cameras = {
        name: camera_config(require(rig, f"cameras.{name}"))
        for name in ("front", "side")
    }
    robot_config = SOFollowerRobotConfig(
        id=str(require(robot_source, "id")),
        calibration_dir=calibration_file.parent,
        port=str(require(robot_source, "port")),
        disable_torque_on_disconnect=True,
        max_relative_target=robot_source.get("max_relative_target"),
        cameras=cameras,
    )
    robot = SOFollower(robot_config)
    device = torch.device("cuda")
    observation_ms: list[float] = []
    inference_ms: list[float] = []
    total_ms: list[float] = []
    first_state = None
    last_state = None
    last_action = None
    frame_shapes = None

    def run_once() -> tuple[float, float, float, np.ndarray, torch.Tensor, dict]:
        cycle_start = time.perf_counter()
        observation_start = cycle_start
        raw = robot.get_observation()
        observation_elapsed = (time.perf_counter() - observation_start) * 1000
        state = np.asarray([raw[key] for key in MOTOR_KEYS], dtype=np.float32)
        policy_observation = {
            "observation.state": state,
            "observation.images.front": raw["front"],
            "observation.images.side": raw["side"],
        }
        inference_start = time.perf_counter()
        action = predict_action(
            policy_observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            use_amp=True,
            task=ACT_TASK,
            robot_type="so_follower",
        )
        inference_elapsed = (time.perf_counter() - inference_start) * 1000
        cycle_elapsed = (time.perf_counter() - cycle_start) * 1000
        return observation_elapsed, inference_elapsed, cycle_elapsed, state, action, raw

    try:
        # Intentionally bypass robot.connect(): this avoids configure(), calibration,
        # torque enable, controller-gain writes, and all Goal_Position writes.
        robot.bus.connect()
        for camera in robot.cameras.values():
            camera.connect()
        if not robot.is_connected:
            raise RuntimeError("robot bus/camera read path did not fully connect")

        policy.reset()
        period_s = 1 / args.target_fps
        for _ in range(args.warmup):
            started = time.perf_counter()
            run_once()
            remaining = period_s - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)

        for _ in range(args.iterations):
            started = time.perf_counter()
            obs_t, infer_t, cycle_t, state, action, raw = run_once()
            if not torch.isfinite(action).all():
                raise RuntimeError("policy produced NaN or Inf")
            if first_state is None:
                first_state = state.copy()
                frame_shapes = {
                    "front": list(raw["front"].shape),
                    "side": list(raw["side"].shape),
                }
            last_state = state.copy()
            last_action = action
            observation_ms.append(obs_t)
            inference_ms.append(infer_t)
            total_ms.append(cycle_t)
            remaining = period_s - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        for camera in robot.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)

    period_ms = 1000 / args.target_fps
    deadline_misses = sum(value > period_ms for value in total_ms)
    result = {
        "status": "PASS" if deadline_misses == 0 and percentile(total_ms, 0.95) <= period_ms else "FAIL",
        "actions_sent": 0,
        "goal_position_writes": 0,
        "dataset_created": False,
        "upload_attempted": False,
        "connection_mode": "bus and cameras only; robot.connect/configure bypassed",
        "disconnect_torque_disabled": True,
        "gpu": torch.cuda.get_device_name(0),
        "policy_revision": require(policy_source, "revision"),
        "weight_sha256": actual_sha,
        "n_action_steps": policy_config.n_action_steps,
        "temporal_ensemble_coeff": policy_config.temporal_ensemble_coeff,
        "use_amp": policy_config.use_amp,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "target_fps": args.target_fps,
        "period_ms": round(period_ms, 3),
        "observation_mean_ms": round(statistics.fmean(observation_ms), 3),
        "observation_p95_ms": round(percentile(observation_ms, 0.95), 3),
        "inference_mean_ms": round(statistics.fmean(inference_ms), 3),
        "inference_p95_ms": round(percentile(inference_ms, 0.95), 3),
        "total_mean_ms": round(statistics.fmean(total_ms), 3),
        "total_median_ms": round(statistics.median(total_ms), 3),
        "total_p95_ms": round(percentile(total_ms, 0.95), 3),
        "total_max_ms": round(max(total_ms), 3),
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": round(deadline_misses / len(total_ms), 4),
        "frame_shapes": frame_shapes,
        "first_state": [round(float(value), 4) for value in first_state],
        "last_state": [round(float(value), 4) for value in last_state],
        "max_abs_state_delta": round(float(np.max(np.abs(last_state - first_state))), 4),
        "last_action_discarded": [round(float(value), 4) for value in last_action.squeeze(0)],
        "gate": f"zero {args.target_fps} Hz deadline misses and total P95 <= {period_ms:.3f} ms",
    }
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
