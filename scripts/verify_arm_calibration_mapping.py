#!/usr/bin/env python3
"""Read-only verification of SO-101 port-to-calibration mapping.

This intentionally does not call Robot.connect() or Teleoperator.connect(),
because those paths configure motors and write registers. It opens the motor
bus, performs the normal read-only handshake, reads calibration registers, and
disconnects with disable_torque=False to avoid any motor write.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_config import load_config, require


def check_follower(rig: dict, port: str | None = None) -> bool:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    calibration_file = Path(require(rig, "robot.calibration_file"))
    config = SO101FollowerConfig(
        port=port or require(rig, "robot.port"),
        id=require(rig, "robot.id"),
        calibration_dir=calibration_file.parent,
        cameras={},
    )
    device = SO101Follower(config)
    try:
        device.bus.connect()
        matched = device.bus.is_calibrated
        print(
            f"follower port={config.port} id={config.id} "
            f"calibration_match={str(matched).lower()}"
        )
        return matched
    finally:
        if device.bus.is_connected:
            device.bus.disconnect(disable_torque=False)


def check_leader(rig: dict, port: str | None = None) -> bool:
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    calibration_file = Path(require(rig, "teleop.calibration_file"))
    config = SO101LeaderConfig(
        port=port or require(rig, "teleop.port"),
        id=require(rig, "teleop.id"),
        calibration_dir=calibration_file.parent,
    )
    device = SO101Leader(config)
    try:
        device.bus.connect()
        matched = device.bus.is_calibrated
        print(
            f"leader port={config.port} id={config.id} "
            f"calibration_match={str(matched).lower()}"
        )
        return matched
    finally:
        if device.bus.is_connected:
            device.bus.disconnect(disable_torque=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rig.local.yaml")
    parser.add_argument(
        "--acknowledge-read-only-motor-bus",
        action="store_true",
        help="Required: open serial buses and read calibration registers.",
    )
    args = parser.parse_args()
    if not args.acknowledge_read_only_motor_bus:
        raise SystemExit(
            "Refusing to open motor buses without "
            "--acknowledge-read-only-motor-bus"
        )
    rig = load_config(args.config)
    follower_port = require(rig, "robot.port")
    leader_port = require(rig, "teleop.port")
    follower_ok = check_follower(rig, follower_port)
    leader_ok = check_leader(rig, leader_port)
    if follower_ok and leader_ok:
        print("PASS: both stable serial paths match their existing calibration files")
        return 0
    print("Configured mapping failed; checking the swapped mapping read-only")
    swapped_follower_ok = check_follower(rig, leader_port)
    swapped_leader_ok = check_leader(rig, follower_port)
    if swapped_follower_ok and swapped_leader_ok:
        print("SWAPPED_MATCH: exchange robot.port and teleop.port in the rig config")
        return 2
    print("FAIL: neither direct nor swapped port/calibration mapping matches")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
