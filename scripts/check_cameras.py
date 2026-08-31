#!/usr/bin/env python3
"""List cameras or explicitly probe the configured streams."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import json
from pathlib import Path
import subprocess
import time

from project_config import load_config, require


def list_devices() -> None:
    print("video_devices=" + json.dumps(sorted(glob.glob("/dev/video*"))))
    result = subprocess.run(
        ["v4l2-ctl", "--list-devices"], text=True, capture_output=True, check=False
    )
    print(result.stdout or result.stderr)


def _probe_one(name: str, camera: dict, frames: int):
    import cv2

    device = require(camera, "index_or_path")
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        return name, {"device": device, "opened": False}, None
    capture.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*str(require(camera, "fourcc"))),
    )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(require(camera, "width")))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(require(camera, "height")))
    capture.set(cv2.CAP_PROP_FPS, int(require(camera, "fps")))
    ok_count = 0
    last_frame = None
    started = time.monotonic()
    for _ in range(frames):
        ok, frame = capture.read()
        if ok and frame is not None:
            ok_count += 1
            last_frame = frame
    elapsed = time.monotonic() - started
    actual = {
        "device": device,
        "opened": True,
        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps_reported": capture.get(cv2.CAP_PROP_FPS),
        "frames_requested": frames,
        "frames_ok": ok_count,
        "elapsed_s": round(elapsed, 3),
        "effective_read_fps": round(ok_count / elapsed, 2) if elapsed else 0,
    }
    capture.release()
    return name, actual, last_frame


def probe(config_path: str, frames: int, output: Path | None) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required for --probe") from exc

    config = load_config(config_path)
    cameras = {
        name: require(config, f"cameras.{name}") for name in ("front", "side")
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_probe_one, name, camera, frames)
            for name, camera in cameras.items()
        ]
        results = [future.result() for future in futures]

    failures = []
    if output:
        output.mkdir(parents=True, exist_ok=True)
    for name, actual, last_frame in results:
        print(name + "=" + json.dumps(actual, sort_keys=True))
        if not actual["opened"]:
            failures.append(f"{name}: failed to open {actual['device']}")
        elif actual["frames_ok"] != frames:
            failures.append(
                f"{name}: only {actual['frames_ok']}/{frames} readable frames"
            )
        elif actual["effective_read_fps"] < 27:
            failures.append(
                f"{name}: effective read rate {actual['effective_read_fps']} fps < 27"
            )
        if output and last_frame is not None:
            cv2.imwrite(str(output / f"{name}.jpg"), last_frame)
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rig.local.yaml")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acknowledge-hardware-access", action="store_true")
    args = parser.parse_args()

    if not args.list and not args.probe:
        parser.error("choose --list and/or --probe")
    if args.list:
        list_devices()
    if args.probe:
        if not args.acknowledge_hardware_access:
            raise SystemExit(
                "--probe opens cameras; pass --acknowledge-hardware-access only "
                "after explicit operator approval"
            )
        return probe(args.config, args.frames, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
