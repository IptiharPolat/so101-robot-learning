#!/usr/bin/env python3
"""Render or explicitly execute one guarded ACT real-robot evaluation trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex

from project_config import ACT_TASK, bool_text, load_config, require


POLICY_PLACEHOLDER = "<CACHED_30K_POLICY_SNAPSHOT>"


def camera_arg(rig: dict) -> str:
    cameras = {}
    for name in ("front", "side"):
        source = require(rig, f"cameras.{name}")
        cameras[name] = {
            "type": require(source, "type"),
            "index_or_path": require(source, "index_or_path"),
            "width": require(source, "width"),
            "height": require(source, "height"),
            "fps": require(source, "fps"),
            "fourcc": require(source, "fourcc"),
        }
    return json.dumps(cameras, separators=(",", ":"))


def resolve_snapshot(config: dict, allow_missing: bool) -> str:
    if allow_missing:
        return POLICY_PLACEHOLDER
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=str(require(config, "policy.repo_id")),
        revision=str(require(config, "policy.revision")),
        local_files_only=True,
    )


def build_command(rig: dict, config: dict, policy_path: str) -> list[str]:
    trial = require(config, "single_trial")
    record = require(rig, "recording")
    robot_calibration = Path(require(rig, "robot.calibration_file"))
    dataset_root = Path(require(trial, "dataset_root")).expanduser().resolve()
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(require(rig, "conda_env")),
        "lerobot-record",
        f"--robot.type={require(rig, 'robot.type')}",
        f"--robot.port={require(rig, 'robot.port')}",
        f"--robot.id={require(rig, 'robot.id')}",
        f"--robot.calibration_dir={robot_calibration.parent}",
        f"--robot.cameras={camera_arg(rig)}",
        f"--display_data={bool_text(bool(require(record, 'display_data')))}",
        f"--dataset.repo_id={require(trial, 'dataset_repo_id')}",
        f"--dataset.root={dataset_root}",
        f"--dataset.fps={trial.get('control_fps', require(record, 'fps'))}",
        f"--dataset.num_episodes={require(trial, 'num_episodes')}",
        f"--dataset.single_task={ACT_TASK}",
        f"--dataset.episode_time_s={require(trial, 'episode_time_s')}",
        f"--dataset.reset_time_s={require(trial, 'reset_time_s')}",
        "--dataset.video=true",
        f"--dataset.vcodec={require(record, 'vcodec')}",
        f"--dataset.push_to_hub={bool_text(bool(require(trial, 'push_to_hub')))}",
        f"--dataset.private={bool_text(bool(require(trial, 'private')))}",
        f"--policy.path={policy_path}",
        f"--policy.device={require(config, 'policy.device')}",
    ]
    if "n_action_steps" in config["policy"]:
        command.append(f"--policy.n_action_steps={config['policy']['n_action_steps']}")
    if "use_amp" in config["policy"]:
        command.append(f"--policy.use_amp={bool_text(bool(config['policy']['use_amp']))}")
    if "temporal_ensemble_coeff" in config["policy"]:
        command.append(
            f"--policy.temporal_ensemble_coeff={config['policy']['temporal_ensemble_coeff']}"
        )
    max_target = rig["robot"].get("max_relative_target")
    if max_target is not None:
        command.insert(10, f"--robot.max_relative_target={max_target}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig", default="configs/rig.local.yaml")
    parser.add_argument("--config", default="configs/act_eval.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-missing-model", action="store_true")
    args = parser.parse_args()
    if args.execute and args.allow_missing_model:
        raise SystemExit("FAIL: --allow-missing-model is forbidden in execute mode")

    rig = load_config(args.rig)
    config = load_config(args.config)
    trial = require(config, "single_trial")
    if require(trial, "num_episodes") != 1:
        raise SystemExit("FAIL: the guarded evaluation must contain exactly one episode")
    layout_id = str(require(trial, "layout_id"))
    seen_layout = bool(require(trial, "seen_layout"))
    valid_seen = {"L1", "L2", "L3", "L4", "L5"}
    valid_unseen = {"U1", "U2", "U3"}
    if (seen_layout and layout_id not in valid_seen) or (
        not seen_layout and layout_id not in valid_unseen
    ):
        raise SystemExit(
            "FAIL: layout_id/seen_layout mismatch; use L1-L5 with true or "
            "U1-U3 with false"
        )
    if bool(require(trial, "push_to_hub")):
        raise SystemExit("FAIL: the guarded first evaluation must remain local")

    policy_path = resolve_snapshot(config, args.allow_missing_model)
    command = build_command(rig, config, policy_path)
    print(shlex.join(command))
    if not args.execute:
        print("\nDRY RUN: no camera, serial port, or robot was opened")
        return 0

    approval_env = trial.get("execution_authorization_env", "SO101_HARDWARE_APPROVED")
    if os.environ.get(approval_env) != "YES":
        raise SystemExit(f"FAIL: set {approval_env}=YES only for the approved trial")
    if trial.get("hardware_gate") not in (None, "PASS"):
        raise SystemExit(
            f"FAIL: diagnostic hardware gate is {trial['hardware_gate']!r}, not 'PASS'"
        )
    if rig.get("audit_status") != "VERIFIED":
        raise SystemExit("FAIL: rig audit_status is not VERIFIED")
    required_paths = [
        Path(require(rig, "robot.port")),
        Path(require(rig, "cameras.front.index_or_path")),
        Path(require(rig, "cameras.side.index_or_path")),
        Path(require(rig, "robot.calibration_file")),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: required hardware/calibration paths are missing: {missing}")
    dataset_root = Path(require(trial, "dataset_root")).expanduser().resolve()
    if dataset_root.exists():
        raise SystemExit(f"FAIL: evaluation output already exists: {dataset_root}")
    if not Path(policy_path).is_dir():
        raise SystemExit(f"FAIL: cached policy snapshot not found: {policy_path}")
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
