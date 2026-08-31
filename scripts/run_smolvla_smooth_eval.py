#!/usr/bin/env python3
"""Render or explicitly execute one guarded Stage-1 SmolVLA smoothness trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from project_config import bool_text, load_config, require
from run_act_eval import camera_arg


ROOT = Path(__file__).resolve().parents[1]
POLICY_PLACEHOLDER = "<SELECTED_RECOVERY_CHECKPOINT>"
EXPECTED_LEROBOT_COMMIT = "0f392484458cb5ebca0310c0c4c47390a31c80ed"
RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.side": "observation.images.camera2",
}
TRIAL_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig", default="configs/rig.local.yaml")
    parser.add_argument("--experiment", default="configs/smolvla_recovery_experiment.yaml")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--layout-id", choices=[f"L{i}_M09" for i in range(1, 6)], required=True)
    parser.add_argument(
        "--order",
        choices=("pink_then_cyan", "cyan_then_pink"),
        required=True,
    )
    parser.add_argument("--mode", choices=("chunk50", "chunk20"), default="chunk50")
    parser.add_argument("--policy-path", default=POLICY_PLACEHOLDER)
    parser.add_argument(
        "--lerobot-checkout",
        default=None,
        help="Optional dedicated pinned LeRobot checkout used instead of rig.lerobot_checkout.",
    )
    parser.add_argument("--episode-time-s", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not TRIAL_RE.fullmatch(args.trial_id):
        raise SystemExit("trial-id must be 3-64 lowercase letters, digits, '_' or '-'")
    if not 10 <= args.episode_time_s <= 180:
        raise SystemExit("episode-time-s must be between 10 and 180")

    rig = load_config(ROOT / args.rig)
    experiment = load_config(ROOT / args.experiment)
    task = require(experiment, f"tasks.{args.order}")
    motion_key = (
        "inference_smoothness.synchronous_guard_candidate"
        if args.mode == "chunk50"
        else "inference_smoothness.synchronous_reactive_diagnostic_candidate"
    )
    n_action_steps = int(require(experiment, f"{motion_key}.n_action_steps"))
    max_relative_target = require(experiment, f"{motion_key}.max_relative_target")
    use_amp = bool(require(experiment, f"{motion_key}.use_amp"))
    if set(max_relative_target) != {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    }:
        raise SystemExit("per-motor safety envelope is incomplete")

    output = ROOT / "outputs/evaluations/smolvla_recovery_smooth_screen" / args.trial_id
    # LeRobot requires policy-generated rollout dataset names to start with
    # `eval_`; this remains local because push_to_hub is always disabled here.
    repo_id = f"iptihar/eval_so101_{args.trial_id}"
    robot_calibration = Path(require(rig, "robot.calibration_file"))
    record = require(rig, "recording")
    policy_path = args.policy_path
    if policy_path != POLICY_PLACEHOLDER:
        candidate = Path(policy_path).expanduser()
        policy_path = str((ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())

    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(require(rig, "conda_env")),
        "lerobot-record",
        f"--robot.type={require(rig, 'robot.type')}",
        f"--robot.port={require(rig, 'robot.port')}",
        f"--robot.id={require(rig, 'robot.id')}",
        f"--robot.calibration_dir={robot_calibration.parent}",
        "--robot.max_relative_target=" + json.dumps(max_relative_target, separators=(",", ":")),
        f"--robot.cameras={camera_arg(rig)}",
        f"--display_data={bool_text(bool(require(record, 'display_data')))}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={output}",
        "--dataset.fps=30",
        "--dataset.num_episodes=1",
        f"--dataset.single_task={task}",
        f"--dataset.episode_time_s={args.episode_time_s}",
        "--dataset.reset_time_s=0",
        "--dataset.video=true",
        f"--dataset.vcodec={require(record, 'vcodec')}",
        "--dataset.push_to_hub=false",
        f"--dataset.private={bool_text(bool(require(record, 'private')))}",
        "--dataset.rename_map=" + json.dumps(RENAME_MAP, separators=(",", ":")),
        f"--policy.path={policy_path}",
        "--policy.device=cuda",
        f"--policy.n_action_steps={n_action_steps}",
        f"--policy.use_amp={bool_text(use_amp)}",
    ]

    print(f"trial={args.trial_id} layout={args.layout_id} order={args.order}")
    print(f"runtime={args.mode} n_action_steps={n_action_steps} fps=30 amp={str(use_amp).lower()}")
    print("max_relative_target=" + json.dumps(max_relative_target, separators=(",", ":")))
    print("task=" + task)
    print(shlex.join(command))
    if not args.execute:
        print("\nDRY RUN ONLY: no camera, serial port, robot, dataset, upload, or policy process was opened.")
        return 0

    if policy_path == POLICY_PLACEHOLDER:
        raise SystemExit("execution refused: provide a selected local checkpoint with --policy-path")
    if os.environ.get("SO101_SMOLVLA_SMOOTH_EVAL_APPROVED") != "YES":
        raise SystemExit("execution refused: SO101_SMOLVLA_SMOOTH_EVAL_APPROVED is not YES")
    if os.environ.get("SO101_SMOLVLA_TRIAL_ID") != args.trial_id:
        raise SystemExit("execution refused: SO101_SMOLVLA_TRIAL_ID does not match")
    if rig.get("audit_status") != "VERIFIED":
        raise SystemExit("execution refused: rig audit_status is not VERIFIED")
    if output.exists():
        raise SystemExit(f"execution refused: output already exists: {output}")
    policy_dir = Path(policy_path)
    if not (policy_dir / "config.json").is_file():
        raise SystemExit(f"execution refused: invalid policy directory: {policy_dir}")
    policy_config = json.loads((policy_dir / "config.json").read_text(encoding="utf-8"))
    if policy_config.get("type") != "smolvla":
        raise SystemExit("execution refused: selected checkpoint is not SmolVLA")
    if policy_config.get("output_features", {}).get("action", {}).get("shape") != [6]:
        raise SystemExit("execution refused: selected checkpoint action shape is not 6D")

    lerobot_root = (
        Path(args.lerobot_checkout).expanduser().resolve()
        if args.lerobot_checkout
        else Path(require(rig, "lerobot_checkout")).expanduser().resolve()
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=lerobot_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if commit != EXPECTED_LEROBOT_COMMIT:
        raise SystemExit(f"execution refused: unexpected LeRobot commit {commit}")
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=lerobot_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    unrelated = [
        line
        for line in status
        if not line.strip().endswith("src/lerobot/scripts/lerobot_record.py")
    ]
    if unrelated:
        raise SystemExit(
            "execution refused: LeRobot checkout has unrelated local changes; "
            "use a clean pinned checkout"
        )
    runtime_env = os.environ.copy()
    runtime_env["PYTHONPATH"] = str(lerobot_root / "src")
    environment_probe = subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            str(require(rig, "conda_env")),
            "python",
            "-c",
            "import pathlib, lerobot; print(pathlib.Path(lerobot.__file__).resolve())",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=runtime_env,
    ).stdout.strip().splitlines()
    expected_import = (lerobot_root / "src/lerobot/__init__.py").resolve()
    if not environment_probe or Path(environment_probe[-1]).resolve() != expected_import:
        raise SystemExit(
            "execution refused: Conda LeRobot import does not point to the audited checkout"
        )
    recorder = lerobot_root / "src/lerobot/scripts/lerobot_record.py"
    source = recorder.read_text(encoding="utf-8")
    required_source = (
        "rename_map=cfg.dataset.rename_map",
        "build_dataset_frame(dataset.features, _sent_action, prefix=ACTION)",
    )
    missing_source = [snippet for snippet in required_source if snippet not in source]
    if missing_source:
        raise SystemExit(
            "execution refused: reviewed recorder runtime patch is not fully applied; "
            f"missing {missing_source}"
        )
    required_paths = [
        Path(require(rig, "robot.port")),
        Path(require(rig, "cameras.front.index_or_path")),
        Path(require(rig, "cameras.side.index_or_path")),
        robot_calibration,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise SystemExit(f"execution refused: missing hardware/calibration paths: {missing_paths}")
    subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            str(require(rig, "conda_env")),
            "python",
            "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.__version__)",
        ],
        check=True,
    )
    os.environ.update(runtime_env)
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
