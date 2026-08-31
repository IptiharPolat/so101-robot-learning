#!/usr/bin/env python3
"""Render or explicitly execute a config-driven ACT record command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex

from project_config import ACT_TASK, bool_text, load_config, require


PLACEHOLDERS = {
    "robot.port": "<FOLLOWER_PORT>",
    "teleop.port": "<LEADER_PORT>",
    "cameras.front.index_or_path": "<FRONT_CAMERA_PATH>",
    "cameras.side.index_or_path": "<SIDE_CAMERA_PATH>",
}


def resolved(config: dict, key: str, allow_placeholders: bool) -> object:
    value: object = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            value = None
            break
        value = value[part]
    if value is None or value == "":
        if allow_placeholders and key in PLACEHOLDERS:
            return PLACEHOLDERS[key]
        raise SystemExit(f"Unresolved required config value: {key}")
    return value


def camera_arg(config: dict, allow_placeholders: bool) -> str:
    cameras = {}
    for name in ("front", "side"):
        source = require(config, f"cameras.{name}")
        cameras[name] = {
            "type": require(source, "type"),
            "index_or_path": resolved(
                config, f"cameras.{name}.index_or_path", allow_placeholders
            ),
            "width": require(source, "width"),
            "height": require(source, "height"),
            "fps": require(source, "fps"),
            "fourcc": require(source, "fourcc"),
        }
    return json.dumps(cameras, separators=(",", ":"))


def build_command(
    rig: dict, experiment: dict, phase: str, allow_placeholders: bool
) -> list[str]:
    env_name = require(rig, "conda_env")
    robot_id = require(rig, "robot.id")
    teleop_id = require(rig, "teleop.id")
    robot_calibration = Path(require(rig, "robot.calibration_file"))
    teleop_calibration = Path(require(rig, "teleop.calibration_file"))
    record = require(rig, "recording")
    phase_config = require(experiment, phase)
    dataset_root = require(record, f"dataset_roots.{phase}")
    task = require(experiment, "task")
    if task != ACT_TASK:
        raise SystemExit("ACT experiment task is not the canonical task string")

    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(env_name),
        "lerobot-record",
        f"--robot.type={require(rig, 'robot.type')}",
        f"--robot.port={resolved(rig, 'robot.port', allow_placeholders)}",
        f"--robot.id={robot_id}",
        f"--robot.calibration_dir={robot_calibration.parent}",
        f"--robot.cameras={camera_arg(rig, allow_placeholders)}",
        f"--teleop.type={require(rig, 'teleop.type')}",
        f"--teleop.port={resolved(rig, 'teleop.port', allow_placeholders)}",
        f"--teleop.id={teleop_id}",
        f"--teleop.calibration_dir={teleop_calibration.parent}",
        f"--display_data={bool_text(bool(require(record, 'display_data')))}",
        f"--dataset.repo_id={require(phase_config, 'dataset_repo_id')}",
        f"--dataset.root={dataset_root}",
        f"--dataset.fps={require(record, 'fps')}",
        f"--dataset.num_episodes={require(phase_config, 'num_episodes')}",
        f"--dataset.single_task={task}",
        f"--dataset.episode_time_s={require(record, 'episode_time_s')}",
        f"--dataset.reset_time_s={require(record, 'reset_time_s')}",
        f"--dataset.video=true",
        f"--dataset.vcodec={require(record, 'vcodec')}",
        f"--dataset.push_to_hub={bool_text(bool(require(record, 'push_to_hub')))}",
        f"--dataset.private={bool_text(bool(require(record, 'private')))}",
    ]
    max_target = rig["robot"].get("max_relative_target")
    if max_target is not None:
        command.insert(10, f"--robot.max_relative_target={max_target}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rig.local.yaml")
    parser.add_argument("--experiment", default="configs/act_experiment.yaml")
    parser.add_argument(
        "--phase", choices=("pilot", "formal", "correction"), default="pilot"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Control hardware and start recording. Never use without explicit approval.",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Printing only: render unresolved hardware fields as placeholders.",
    )
    args = parser.parse_args()

    if args.execute and args.allow_placeholders:
        raise SystemExit("Placeholders are forbidden in execute mode")
    rig = load_config(args.config)
    experiment = load_config(args.experiment)
    command = build_command(rig, experiment, args.phase, args.allow_placeholders)

    print(shlex.join(command))
    if not args.execute:
        print(
            "\nDRY DRAFT ONLY: no hardware was opened. Pass --execute only after "
            "explicit approval and verified rig.local.yaml."
        )
        return 0

    if os.environ.get("SO101_HARDWARE_APPROVED") != "YES":
        raise SystemExit(
            "Execution refused: set SO101_HARDWARE_APPROVED=YES only after the "
            "operator approves this hardware session."
        )
    if rig.get("audit_status") != "VERIFIED":
        raise SystemExit("Execution refused: rig.local.yaml audit_status is not VERIFIED")
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
