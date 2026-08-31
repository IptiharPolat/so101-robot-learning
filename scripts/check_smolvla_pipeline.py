#!/usr/bin/env python3
"""Static and metadata audit of the exact completed SmolVLA training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from project_config import ACT_TASK, CYAN_THEN_PINK_TASK


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/cloud_runs/smolvla_full_30k_b8_wandb_v1"
DEFAULT_DATASET = ROOT / "outputs/datasets/vla_pink_cyan_order_clean_v1"
LEROBOT = Path(
    os.environ.get("LEROBOT_ROOT", ROOT / ".runtime/lerobot-smolvla-eval")
).expanduser().resolve()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=LEROBOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/smolvla_pipeline_audit.json",
    )
    args = parser.parse_args()

    checkpoint = args.run / "checkpoints/030000/pretrained_model"
    policy = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    train = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
    info = json.loads((args.dataset / "meta/info.json").read_text(encoding="utf-8"))
    episodes = pd.read_parquet(args.dataset / "meta/episodes/chunk-000/file-000.parquet")
    task_counts = Counter(item[0] for item in episodes["tasks"])

    recorder_source = (LEROBOT / "src/lerobot/scripts/lerobot_record.py").read_text(
        encoding="utf-8"
    )
    changed_paths = git("status", "--short").splitlines()
    changed_smolvla_core = [
        line for line in changed_paths if "src/lerobot/policies/smolvla/" in line
    ]
    expected_rename = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.side": "observation.images.camera2",
    }

    checks = {
        "dataset_100_episodes": info.get("total_episodes") == 100,
        "dataset_65691_frames": info.get("total_frames") == 65691,
        "tasks_exact_and_balanced": task_counts
        == Counter({ACT_TASK: 50, CYAN_THEN_PINK_TASK: 50}),
        "two_expected_camera_features": all(
            key in info.get("features", {})
            for key in ("observation.images.front", "observation.images.side")
        ),
        "state_action_are_6d": info["features"]["action"]["shape"] == [6]
        and info["features"]["observation.state"]["shape"] == [6],
        "training_started_from_smolvla_base": policy.get("pretrained_path")
        == "lerobot/smolvla_base",
        "training_camera_rename_correct": train.get("rename_map") == expected_rename,
        "training_steps_and_scheduler_match": train.get("steps") == 30000
        and train.get("scheduler", {}).get("num_decay_steps") == 30000,
        "training_batch_seed_correct": train.get("batch_size") == 8
        and train.get("seed") == 20260824,
        "hub_model_upload_disabled": policy.get("push_to_hub") is False,
        "checkpoint_action_shape_6d": policy.get("output_features", {})
        .get("action", {})
        .get("shape")
        == [6],
        "smolvla_core_has_no_local_diff": not changed_smolvla_core,
        "recorder_forwards_camera_rename": "rename_map=cfg.dataset.rename_map" in recorder_source,
        "recorder_persists_actual_clipped_action": (
            "build_dataset_frame(dataset.features, _sent_action, prefix=ACTION)" in recorder_source
        ),
    }
    report = {
        "lerobot_commit": git("rev-parse", "HEAD"),
        "lerobot_status_short": changed_paths,
        "project_root_is_git_repository": (ROOT / ".git").exists(),
        "dataset": {
            "root": str(args.dataset),
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "task_counts": dict(task_counts),
        },
        "training": {
            "run": str(args.run),
            "steps": train.get("steps"),
            "batch_size": train.get("batch_size"),
            "seed": train.get("seed"),
            "rename_map": train.get("rename_map"),
            "train_expert_only": policy.get("train_expert_only"),
            "freeze_vision_encoder": policy.get("freeze_vision_encoder"),
            "n_action_steps": policy.get("n_action_steps"),
            "chunk_size": policy.get("chunk_size"),
            "wandb_run_id": train.get("wandb", {}).get("run_id"),
        },
        "checks": checks,
        "structural_training_pipeline_pass": all(
            value
            for key, value in checks.items()
            if key != "recorder_persists_actual_clipped_action"
        ),
        "known_behavioral_gate": "FAIL_instruction_switch_0_of_4",
        "known_data_content_exclusions": {
            "confirmed_physical_mismatch_pair_ids": [
                "L2_M05",
                "L2_M08",
                "L4_M08",
            ],
            "initial_front_visibility_exclusion_pair_ids": [
                "L3_M04",
                "L1_M02",
                "L3_M02",
                "L5_M02",
                "L5_M01",
                "L5_M06",
            ],
            "all_pair_ids": [
                "L2_M05",
                "L2_M08",
                "L4_M08",
                "L3_M04",
                "L1_M02",
                "L3_M02",
                "L5_M02",
                "L5_M01",
                "L5_M06",
            ],
            "all_merged_episode_indices": [
                16,
                21,
                22,
                29,
                31,
                43,
                45,
                47,
                49,
                66,
                71,
                72,
                79,
                81,
                93,
                95,
                97,
                99,
            ],
            "pair_level_policy": "exclude_both_orders_from_recovery_v2",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    for name, passed in checks.items():
        print(f"{name}={'PASS' if passed else 'WARN'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
