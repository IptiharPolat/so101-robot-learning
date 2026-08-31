#!/usr/bin/env python3
"""Capture a redacted, reproducible command/environment snapshot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess


SAFE_ENV_PREFIXES = ("CUDA_", "WANDB_MODE", "HF_HOME", "PYTORCH_")
SENSITIVE_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")


def output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout or result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conda-env", default="so101-ordering")
    args = parser.parse_args()

    safe_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(SAFE_ENV_PREFIXES)
        and not any(word in key.upper() for word in SENSITIVE_WORDS)
    }
    data = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": args.command,
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "conda_env": args.conda_env,
        "git_sha": output(["git", "rev-parse", "HEAD"]),
        "git_status_short": output(["git", "status", "--short"]),
        "python_version": output(
            ["conda", "run", "-n", args.conda_env, "python", "--version"]
        ),
        "pip_freeze": output(
            ["conda", "run", "-n", args.conda_env, "python", "-m", "pip", "freeze"]
        ).splitlines(),
        "nvidia_smi": output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "safe_environment": safe_environment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
