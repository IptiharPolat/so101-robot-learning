#!/usr/bin/env python3
"""Cache and reload the fixed ACT 30K Hub revision before hardware access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline

from project_config import load_config, require


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/act_eval.yaml")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    repo_id = str(require(config, "policy.repo_id"))
    revision = str(require(config, "policy.revision"))
    expected_sha = str(require(config, "policy.weight_sha256"))

    if not args.download:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "repo_id": repo_id,
                    "revision": revision,
                    "expected_weight_sha256": expected_sha,
                },
                indent=2,
            )
        )
        print("No network access or model download was attempted")
        return 0

    account = HfApi().whoami()["name"]
    if account != "iptihar":
        raise SystemExit(f"FAIL: expected HF account 'iptihar', got {account!r}")
    snapshot = Path(snapshot_download(repo_id=repo_id, revision=revision))
    weight_path = snapshot / "model.safetensors"
    actual_sha = sha256(weight_path)
    if actual_sha != expected_sha:
        raise SystemExit(
            f"FAIL: model SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    policy_config = PreTrainedConfig.from_pretrained(snapshot, local_files_only=True)
    if not isinstance(policy_config, ACTConfig):
        raise SystemExit(f"FAIL: expected ACTConfig, got {type(policy_config).__name__}")
    policy_config.device = "cpu"
    policy = ACTPolicy.from_pretrained(snapshot, config=policy_config, local_files_only=True)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        snapshot,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        snapshot,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "hf_account": account,
                "repo_id": repo_id,
                "revision": revision,
                "snapshot": str(snapshot),
                "weight_sha256": actual_sha,
                "total_parameters": sum(p.numel() for p in policy.parameters()),
                "preprocessor_steps": len(preprocessor.steps),
                "postprocessor_steps": len(postprocessor.steps),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
