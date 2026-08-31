#!/usr/bin/env python3
"""Build and optionally execute config-driven ACT training commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "act_experiment.yaml"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_command(config: dict, phase: str, output_dir: Path) -> list[str]:
    formal = config["formal"]
    train = config["training"]
    steps = int(train[f"{phase}_steps"])
    save_freq = steps if phase == "smoke" else int(train["save_steps"][0])
    log_freq = 20 if phase == "smoke" else 200
    job_name = f"act_two_cube_{phase}_{steps}"

    command = [
        "lerobot-train",
        f"--dataset.repo_id={formal['dataset_repo_id']}",
        f"--dataset.revision={train['dataset_revision']}",
        f"--dataset.video_backend={train['video_backend']}",
        "--policy.type=act",
        "--policy.device=cuda",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={steps}",
        f"--batch_size={int(train['batch_size_start'])}",
        f"--num_workers={int(train['num_workers'])}",
        f"--seed={int(train['seed'])}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        "--save_checkpoint=true",
        f"--wandb.enable={str(train['wandb_mode'] != 'disabled').lower()}",
        f"--wandb.project={train['wandb_project']}",
        f"--wandb.mode={train['wandb_mode']}",
        f"--wandb.disable_artifact={str(bool(train['wandb_disable_artifact'])).lower()}",
        f"--policy.push_to_hub={str(bool(train['policy_push_to_hub'])).lower()}",
        f"--policy.repo_id={train['policy_repo_id']}",
        "--policy.private=true",
    ]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm-full",
        action="store_true",
        help="Second guard required for the 30K run.",
    )
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    relative_output = Path(config["training"][f"{args.phase}_output_dir"])
    train_root = Path(os.environ.get("SO101_TRAIN_ROOT", PROJECT_ROOT))
    output_dir = relative_output if relative_output.is_absolute() else train_root / relative_output
    command = build_command(config, args.phase, output_dir.resolve())

    print(shlex.join(command))
    if not args.execute:
        print("DRY RUN: add --execute to start this phase.")
        return 0
    if args.phase == "full" and not args.confirm_full:
        raise SystemExit("REFUSED: full training also requires --confirm-full")
    if output_dir.exists():
        raise SystemExit(f"REFUSED: output directory already exists: {output_dir}")

    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
