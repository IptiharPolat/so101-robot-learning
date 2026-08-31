#!/usr/bin/env python3
"""Render or execute a guarded, non-recording SO-101 layout reach check."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex

from project_config import load_config, require


def build_command(rig: dict) -> list[str]:
    robot_calibration = Path(require(rig, "robot.calibration_file"))
    teleop_calibration = Path(require(rig, "teleop.calibration_file"))
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(require(rig, "conda_env")),
        "lerobot-teleoperate",
        f"--robot.type={require(rig, 'robot.type')}",
        f"--robot.port={require(rig, 'robot.port')}",
        f"--robot.id={require(rig, 'robot.id')}",
        f"--robot.calibration_dir={robot_calibration.parent}",
        f"--teleop.type={require(rig, 'teleop.type')}",
        f"--teleop.port={require(rig, 'teleop.port')}",
        f"--teleop.id={require(rig, 'teleop.id')}",
        f"--teleop.calibration_dir={teleop_calibration.parent}",
        f"--fps={require(rig, 'teleoperation.fps')}",
        "--display_data=false",
    ]
    max_target = rig["teleoperation"].get("max_relative_target")
    if max_target is not None:
        command.insert(10, f"--robot.max_relative_target={max_target}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig", default="configs/rig.local.yaml")
    parser.add_argument("--layout", required=True, choices=("U1", "U2", "U3"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    rig = load_config(args.rig)
    command = build_command(rig)
    print(shlex.join(command))
    if not args.execute:
        print("\nDRY RUN: no camera, serial port, or robot was opened")
        return 0

    approval_env = f"SO101_{args.layout}_REACH_APPROVED"
    if os.environ.get(approval_env) != "YES":
        raise SystemExit(f"FAIL: set {approval_env}=YES only for the approved check")
    if rig.get("audit_status") != "VERIFIED":
        raise SystemExit("FAIL: rig audit_status is not VERIFIED")
    if not require(rig, "teleoperation.enabled"):
        raise SystemExit("FAIL: teleoperation is disabled in rig.local.yaml")
    if rig["teleoperation"].get("max_relative_target") is not None:
        raise SystemExit("FAIL: reach checks require the verified no-clamp configuration")

    required_paths = [
        Path(require(rig, "robot.port")),
        Path(require(rig, "teleop.port")),
        Path(require(rig, "robot.calibration_file")),
        Path(require(rig, "teleop.calibration_file")),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: required hardware/calibration paths are missing: {missing}")

    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
