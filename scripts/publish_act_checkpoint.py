#!/usr/bin/env python3
"""Validate and publish one trained ACT checkpoint to a private HF model repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline


EXPECTED_DATASET = "iptihar/so101_pink_cyan_sequence_act_v1"
EXPECTED_REPO = "iptihar/act_so101_pink_cyan_sequence_v1"
EXPECTED_ACCOUNT = "iptihar"
REQUIRED_FILES = {
    "README.md",
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
}


def checkpoint_dir(output_dir: Path, step: int) -> Path:
    checkpoint = output_dir.expanduser().resolve() / "checkpoints" / f"{step:06d}" / "pretrained_model"
    if not checkpoint.is_dir():
        raise SystemExit(f"FAIL: checkpoint not found: {checkpoint}")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=30000)
    parser.add_argument("--repo-id", default=EXPECTED_REPO)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-repo", default="")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow adding commits to an already existing target repository.",
    )
    args = parser.parse_args()

    if args.step != 30000:
        raise SystemExit("FAIL: this project publishes only the QC-approved 30000-step checkpoint")
    if args.repo_id != EXPECTED_REPO:
        raise SystemExit(f"FAIL: expected repo {EXPECTED_REPO!r}, got {args.repo_id!r}")

    checkpoint = checkpoint_dir(args.output_dir, args.step)
    policy_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(policy_config, ACTConfig):
        raise SystemExit(f"FAIL: expected ACTConfig, found {type(policy_config).__name__}")

    train_config = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True)
    if train_config.dataset.repo_id != EXPECTED_DATASET:
        raise SystemExit(
            f"FAIL: expected dataset {EXPECTED_DATASET!r}, got {train_config.dataset.repo_id!r}"
        )

    policy_config.device = "cpu"
    policy_config.repo_id = args.repo_id
    policy_config.private = True
    policy_config.push_to_hub = False
    train_config.policy = policy_config

    policy = ACTPolicy.from_pretrained(checkpoint, config=policy_config, local_files_only=True)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
    )
    total = sum(parameter.numel() for parameter in policy.parameters())

    api = HfApi()
    account = api.whoami()["name"]
    if account != EXPECTED_ACCOUNT:
        raise SystemExit(f"FAIL: expected HF account {EXPECTED_ACCOUNT!r}, got {account!r}")

    repo_exists = True
    try:
        existing = api.repo_info(args.repo_id, repo_type="model")
        existing_private = existing.private
    except RepositoryNotFoundError:
        repo_exists = False
        existing_private = None

    summary = {
        "status": "READY_TO_PUBLISH" if not args.execute else "PUBLISH_REQUESTED",
        "checkpoint": str(checkpoint),
        "step": args.step,
        "dataset_repo_id": train_config.dataset.repo_id,
        "target_repo_id": args.repo_id,
        "target_private": True,
        "target_repo_exists": repo_exists,
        "target_existing_private": existing_private,
        "hf_account": account,
        "total_parameters": total,
        "preprocessor_steps": len(preprocessor.steps),
        "postprocessor_steps": len(postprocessor.steps),
    }
    print(json.dumps(summary, indent=2))

    if not args.execute:
        print("DRY RUN: no files were uploaded")
        return 0
    if args.confirm_repo != args.repo_id:
        raise SystemExit("FAIL: --confirm-repo must exactly match --repo-id")
    if repo_exists and not args.allow_existing:
        raise SystemExit("FAIL: target repo already exists; inspect it before using --allow-existing")
    if repo_exists and existing_private is not True:
        raise SystemExit("FAIL: existing target repository is not private")

    policy.push_model_to_hub(train_config)
    preprocessor.push_to_hub(args.repo_id, private=True, commit_message="Upload ACT preprocessor")
    postprocessor.push_to_hub(args.repo_id, private=True, commit_message="Upload ACT postprocessor")

    info = api.repo_info(args.repo_id, repo_type="model")
    files = set(api.list_repo_files(args.repo_id, repo_type="model", revision=info.sha))
    missing = sorted(REQUIRED_FILES - files)
    if missing:
        raise SystemExit(f"FAIL: upload completed but required files are missing: {missing}")
    if info.private is not True:
        raise SystemExit("FAIL: uploaded repository is not private")

    print(
        json.dumps(
            {
                "status": "PASS",
                "repo_id": args.repo_id,
                "revision": info.sha,
                "private": info.private,
                "required_files_present": sorted(REQUIRED_FILES),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
