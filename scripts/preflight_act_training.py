#!/usr/bin/env python3
"""Cloud-side ACT dataset/model preflight without optimizer updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "act_experiment.yaml")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = config["training"]
    expected_task = config["task"]
    repo_id = config["formal"]["dataset_repo_id"]

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: torch.cuda.is_available() is false")

    kwargs = {
        "repo_id": repo_id,
        "revision": train["dataset_revision"],
        "video_backend": train["video_backend"],
    }
    if args.root:
        kwargs["root"] = args.root.expanduser().resolve()
    dataset = LeRobotDataset(**kwargs)

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    camera_keys = sorted(key for key in batch if key.startswith("observation.images."))
    if camera_keys != ["observation.images.front", "observation.images.side"]:
        raise SystemExit(f"FAIL: unexpected cameras: {camera_keys}")
    if tuple(batch["observation.state"].shape) != (2, 6):
        raise SystemExit(f"FAIL: state shape {tuple(batch['observation.state'].shape)}")
    if tuple(batch["action"].shape) != (2, 6):
        raise SystemExit(f"FAIL: action shape {tuple(batch['action'].shape)}")

    tasks = pd.read_parquet(Path(dataset.root) / "meta" / "tasks.parquet")
    task_values = list(tasks.index.astype(str))
    if task_values != [expected_task]:
        raise SystemExit(f"FAIL: unexpected task labels: {task_values}")

    policy_cfg = ACTConfig(device="cuda")
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    total_params = sum(parameter.numel() for parameter in policy.parameters())
    trainable_params = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)

    result = {
        "status": "PASS",
        "repo_id": repo_id,
        "revision": train["dataset_revision"],
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "batch_size_checked": 2,
        "camera_shapes": {key: list(batch[key].shape) for key in camera_keys},
        "state_shape": list(batch["observation.state"].shape),
        "action_shape": list(batch["action"].shape),
        "task": task_values[0],
        "gpu": torch.cuda.get_device_name(0),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
