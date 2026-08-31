#!/usr/bin/env python3
"""Run LeRobot record with a project-local fix for policy-only reset timing."""

from __future__ import annotations

import logging
import time

from lerobot.scripts import lerobot_record
from lerobot.utils.robot_utils import precise_sleep


_original_record_loop = lerobot_record.record_loop


def reset_safe_record_loop(*args, **kwargs):
    """Unlock the follower and require operator confirmation between episodes."""
    policy = kwargs.get("policy")
    teleop = kwargs.get("teleop")
    dataset = kwargs.get("dataset")
    if policy is not None or teleop is not None or dataset is not None:
        return _original_record_loop(*args, **kwargs)

    robot = kwargs["robot"]
    events = kwargs["events"]
    fps = float(kwargs["fps"])
    bus = robot.bus
    bus.disable_torque(num_retry=5)
    logging.info(
        "Manual reset window started with follower torque disabled; align the arms, "
        "reset the cubes, then press right arrow",
    )
    confirmed = False
    while True:
        if events["stop_recording"]:
            break
        if events["exit_early"]:
            events["exit_early"] = False
            confirmed = True
            break
        loop_started = time.perf_counter()
        robot.get_observation()
        precise_sleep(max(1 / fps - (time.perf_counter() - loop_started), 0.0))
    if confirmed:
        present = bus.sync_read("Present_Position", num_retry=5)
        bus.sync_write("Goal_Position", present, num_retry=5)
        bus.enable_torque(num_retry=5)
        logging.info("Manual reset confirmed; current pose latched and follower torque enabled")
    else:
        logging.info("Manual reset interrupted; follower torque remains disabled")


def main() -> None:
    lerobot_record.record_loop = reset_safe_record_loop
    lerobot_record.main()


if __name__ == "__main__":
    main()
