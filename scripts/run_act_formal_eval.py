#!/usr/bin/env python3
"""Render or execute one row of the fixed ACT 32-trial evaluation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex

from project_config import load_config, require
from run_act_eval import build_command, resolve_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    "repo_id": "iptihar/act_so101_pink_cyan_sequence_v1",
    "revision": "d5d6fa1afa56b928808091306b2edcc7d01c200b",
    "weight_sha256": "dc2900eed6e179c143f817cf051efe41843f82d87b5ecd418e2fca8fef558d01",
    "device": "cuda",
    "use_amp": True,
    "n_action_steps": 1,
    "temporal_ensemble_coeff": 0.01,
}


def find_row(path: Path, evaluation_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row["evaluation_id"] == evaluation_id]
    if len(matches) != 1:
        raise SystemExit(f"FAIL: expected one manifest row for {evaluation_id}, found {len(matches)}")
    return matches[0]


def make_config(row: dict[str, str]) -> dict:
    evaluation_id = row["evaluation_id"]
    index = int(row["planned_order"])
    return {
        "policy": POLICY,
        "single_trial": {
            "evaluation_id": evaluation_id,
            "layout_id": row["layout_id"],
            "seen_layout": row["seen_layout"].lower() == "true",
            "dataset_repo_id": f"iptihar/eval_act_30k_formal_{index:03d}",
            "dataset_root": f"outputs/evaluations/act_formal_32/{evaluation_id.lower()}",
            "num_episodes": 1,
            "control_fps": int(row["control_fps"]),
            "episode_time_s": 120,
            "reset_time_s": 10,
            "push_to_hub": False,
            "private": True,
        },
    }


def make_continuous_config(start_index: int) -> dict:
    return {
        "policy": POLICY,
        "single_trial": {
            "evaluation_id": "ACT30K-FORMAL-32-CONTINUOUS",
            "dataset_repo_id": "iptihar/eval_act_30k_formal_32_v2",
            "dataset_root": "outputs/evaluations/act_formal_32_continuous_v2",
            "num_episodes": 32 - start_index,
            "control_fps": 25,
            "episode_time_s": 120,
            "reset_time_s": 60,
            "push_to_hub": False,
            "private": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_id", nargs="?")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/act_evaluation_schedule.csv"))
    parser.add_argument("--rig", default="configs/rig.local.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--allow-missing-model", action="store_true")
    args = parser.parse_args()
    if args.execute and args.allow_missing_model:
        raise SystemExit("FAIL: --allow-missing-model is forbidden in execute mode")

    if args.continuous:
        if args.evaluation_id is not None:
            raise SystemExit("FAIL: omit evaluation_id when using --continuous")
        if not 0 <= args.start_index < 32:
            raise SystemExit("FAIL: --start-index must be in [0, 31]")
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        prior_ok = {"recorded_pending_annotation", "completed"}
        if (
            len(rows) != 32
            or any(row["status"] not in prior_ok for row in rows[: args.start_index])
            or any(row["status"] != "planned" for row in rows[args.start_index :])
        ):
            raise SystemExit("FAIL: manifest statuses do not match --start-index")
        config = make_continuous_config(args.start_index)
    else:
        if args.evaluation_id is None:
            raise SystemExit("FAIL: evaluation_id is required without --continuous")
        row = find_row(args.manifest, args.evaluation_id)
        if row["status"] != "planned":
            raise SystemExit(f"FAIL: manifest status is {row['status']!r}, not 'planned'")
        config = make_config(row)
    rig = load_config(args.rig)
    policy_path = resolve_snapshot(config, args.allow_missing_model)
    command = build_command(rig, config, policy_path)
    if args.continuous:
        command[5:6] = ["python", str(PROJECT_ROOT / "scripts/lerobot_record_reset_safe.py")]
        if args.start_index:
            command.append("--resume=true")
    print(shlex.join(command))
    if not args.execute:
        print("\nDRY RUN: no camera, serial port, or robot was opened")
        return 0

    if os.environ.get("SO101_ACT_FORMAL_32_APPROVED") != "YES":
        raise SystemExit("FAIL: set SO101_ACT_FORMAL_32_APPROVED=YES only for the approved phase")
    if rig.get("audit_status") != "VERIFIED":
        raise SystemExit("FAIL: rig audit_status is not VERIFIED")
    trial = config["single_trial"]
    required_paths = [
        Path(require(rig, "robot.port")),
        Path(require(rig, "cameras.front.index_or_path")),
        Path(require(rig, "cameras.side.index_or_path")),
        Path(require(rig, "robot.calibration_file")),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: required hardware/calibration paths are missing: {missing}")
    output = PROJECT_ROOT / trial["dataset_root"]
    if args.continuous and args.start_index:
        info_path = output / "meta/info.json"
        if not info_path.is_file():
            raise SystemExit(f"FAIL: resume metadata missing: {info_path}")
        import json

        total_episodes = int(json.loads(info_path.read_text())["total_episodes"])
        if total_episodes != args.start_index:
            raise SystemExit(
                f"FAIL: resume dataset has {total_episodes} episodes, expected {args.start_index}"
            )
    elif output.exists():
        raise SystemExit(f"FAIL: evaluation output already exists: {output}")
    if not Path(policy_path).is_dir():
        raise SystemExit(f"FAIL: cached policy snapshot not found: {policy_path}")
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
