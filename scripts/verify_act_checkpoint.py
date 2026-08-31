#!/usr/bin/env python3
"""Reload an ACT checkpoint and report its parameter counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = args.output_dir.expanduser().resolve() / "checkpoints" / "last" / "pretrained_model"
    if not checkpoint.is_dir():
        raise SystemExit(f"FAIL: checkpoint not found: {checkpoint}")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, ACTConfig):
        raise SystemExit(f"FAIL: expected ACTConfig, found {type(config).__name__}")
    config.device = args.device
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True)
    total = sum(parameter.numel() for parameter in policy.parameters())
    trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    print(
        json.dumps(
            {
                "status": "PASS",
                "checkpoint": str(checkpoint),
                "device": args.device,
                "total_parameters": total,
                "trainable_parameters": trainable,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
