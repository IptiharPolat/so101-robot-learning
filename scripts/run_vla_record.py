#!/usr/bin/env python3
"""Render or explicitly execute one schedule-locked SmolVLA demonstration."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from project_config import ACT_TASK, CYAN_THEN_PINK_TASK, bool_text, load_config, require
from run_record import camera_arg, resolved


EXPECTED = {
    "pink_then_cyan": ("pink", "cyan", ACT_TASK),
    "cyan_then_pink": ("cyan", "pink", CYAN_THEN_PINK_TASK),
}
COMPLETED_STATUSES = {"recorded_pending_qc", "accepted"}
RETRY_STATUS = "rejected_needs_retry"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {path}")
        return reader.fieldnames, list(reader)


def dataset_episode_count(root: Path) -> int:
    if not root.exists():
        return 0
    info = root / "meta" / "info.json"
    if not info.is_file():
        raise SystemExit(
            f"Dataset root exists without meta/info.json; refuse ambiguous resume: {root}"
        )
    value = json.loads(info.read_text(encoding="utf-8")).get("total_episodes")
    if not isinstance(value, int) or value < 0:
        raise SystemExit(f"Invalid total_episodes in {info}")
    return value


def validate_row(row: dict[str, str], experiment: dict) -> None:
    order = row.get("order_type", "")
    if order not in EXPECTED:
        raise SystemExit(f"Invalid order_type for selected row: {order!r}")
    first, second, task = EXPECTED[order]
    expected_task = require(experiment, f"tasks.{order}")
    if expected_task != task:
        raise SystemExit(f"Experiment task for {order} is not canonical")
    if (row.get("first_color"), row.get("second_color"), row.get("task")) != (
        first,
        second,
        task,
    ):
        raise SystemExit("Selected manifest row has inconsistent colors or task")


def micro_layout_id(row: dict[str, str]) -> str:
    value = row.get("micro_layout_id") or row.get("base_micro_layout_id")
    if not value:
        raise SystemExit("Selected manifest row has no micro-layout identifier")
    return value


def repeat_id(row: dict[str, str]) -> str:
    value = row.get("repeat_id") or row.get("supplement_repeat_id")
    if not value:
        raise SystemExit("Selected manifest row has no repeat identifier")
    return value


def recorded_source_count(rows: list[dict[str, str]], order: str) -> int:
    """Count every raw take retained for one order, including replacements."""
    prefixes = ("raw_dataset_episode=", "replacement_raw_dataset_episode=")
    return sum(
        1
        for row in rows
        if row.get("order_type") == order
        for note in row.get("notes", "").split("; ")
        if note.startswith(prefixes)
    )


def dataset_root_for_order(root: Path, rig: dict, experiment: dict, order: str) -> Path:
    configured = experiment.get("collection", {}).get("dataset_roots", {}).get(order)
    if configured is not None:
        path = Path(str(configured)).expanduser()
        return (root / path).resolve() if not path.is_absolute() else path.resolve()
    return Path(require(rig, f"recording.dataset_roots.vla.{order}")).resolve()


def build_command(
    rig: dict,
    experiment: dict,
    row: dict[str, str],
    dataset_root: Path,
    resume: bool,
    allow_placeholders: bool,
) -> list[str]:
    record = require(rig, "recording")
    if bool(require(record, "push_to_hub")):
        raise SystemExit("VLA collection requires recording.push_to_hub=false")
    if bool(require(experiment, "collection.upload_during_collection")):
        raise SystemExit("VLA collection config must keep upload_during_collection=false")
    if require(experiment, "collection.episodes_per_invocation") != 1:
        raise SystemExit("Exactly one episode per VLA invocation is required")
    camera_specs = {
        (
            require(rig, f"cameras.{name}.width"),
            require(rig, f"cameras.{name}.height"),
            require(rig, f"cameras.{name}.fps"),
            require(rig, f"cameras.{name}.fourcc"),
        )
        for name in ("front", "side")
    }
    if len(camera_specs) != 1:
        raise SystemExit("Front and side camera recording specs must be identical")
    if next(iter(camera_specs))[2] != require(record, "fps"):
        raise SystemExit("Camera FPS and dataset FPS must match")

    order = row["order_type"]
    robot_calibration = Path(require(rig, "robot.calibration_file"))
    teleop_calibration = Path(require(rig, "teleop.calibration_file"))
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(require(rig, "conda_env")),
        "lerobot-record",
        f"--robot.type={require(rig, 'robot.type')}",
        f"--robot.port={resolved(rig, 'robot.port', allow_placeholders)}",
        f"--robot.id={require(rig, 'robot.id')}",
        f"--robot.calibration_dir={robot_calibration.parent}",
        f"--robot.cameras={camera_arg(rig, allow_placeholders)}",
        f"--teleop.type={require(rig, 'teleop.type')}",
        f"--teleop.port={resolved(rig, 'teleop.port', allow_placeholders)}",
        f"--teleop.id={require(rig, 'teleop.id')}",
        f"--teleop.calibration_dir={teleop_calibration.parent}",
        f"--display_data={bool_text(bool(require(record, 'display_data')))}",
        f"--dataset.repo_id={require(experiment, f'datasets.{order}')}",
        f"--dataset.root={dataset_root}",
        f"--dataset.fps={require(record, 'fps')}",
        "--dataset.num_episodes=1",
        f"--dataset.single_task={row['task']}",
        f"--dataset.episode_time_s={require(record, 'episode_time_s')}",
        f"--dataset.reset_time_s={require(record, 'reset_time_s')}",
        "--dataset.video=true",
        f"--dataset.vcodec={require(record, 'vcodec')}",
        "--dataset.push_to_hub=false",
        f"--dataset.private={bool_text(bool(require(record, 'private')))}",
    ]
    if resume:
        command.append("--resume=true")
    max_target = rig["robot"].get("max_relative_target")
    if max_target is not None:
        command.insert(10, f"--robot.max_relative_target={max_target}")
    return command


def update_manifest(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    episode_id: str,
    source_episode: int,
    replacement: bool,
) -> None:
    for row in rows:
        if row["episode_id"] == episode_id:
            row["status"] = "recorded_pending_qc"
            row["accepted"] = ""
            prefix = "replacement_raw_dataset_episode" if replacement else "raw_dataset_episode"
            note = f"{prefix}={source_episode}"
            row["notes"] = "; ".join(filter(None, (row.get("notes", ""), note)))
            break
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--config", default="configs/rig.local.yaml")
    parser.add_argument("--experiment", default="configs/smolvla_experiment.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="Append a replacement for a row explicitly marked rejected_needs_retry.",
    )
    parser.add_argument("--allow-placeholders", action="store_true")
    parser.add_argument(
        "--operator-layout-confirmed",
        action="store_true",
        help=(
            "Permit only this selected micro-layout after the operator confirms "
            "its physical placement for the authorized recording session."
        ),
    )
    args = parser.parse_args()

    if args.execute and args.allow_placeholders:
        raise SystemExit("Placeholders are forbidden in execute mode")
    rig = load_config(args.config)
    experiment = load_config(args.experiment)
    manifest = Path(args.manifest or require(experiment, "collection.manifest")).resolve()
    fieldnames, rows = read_rows(manifest)
    coverage_path = Path(require(experiment, "collection.coverage_manifest")).resolve()
    _, coverage_rows = read_rows(coverage_path)
    selected = [row for row in rows if row.get("episode_id") == args.episode_id]
    if len(selected) != 1:
        raise SystemExit(f"Expected exactly one manifest row for {args.episode_id}")
    row = selected[0]
    validate_row(row, experiment)
    selected_micro_layout = micro_layout_id(row)
    selected_repeat = repeat_id(row)
    coverage_matches = [
        item
        for item in coverage_rows
        if item.get("micro_layout_id") == selected_micro_layout
    ]
    if len(coverage_matches) != 1:
        raise SystemExit(
            f"Expected one coverage row for {selected_micro_layout!r}"
        )
    coverage_row = coverage_matches[0]
    if row.get("base_micro_layout_id"):
        expected_pair_id = f"recovery_{selected_micro_layout}_R{int(selected_repeat):02d}"
        pair_matches = row.get("pair_id") == expected_pair_id
    else:
        pair_matches = (
            row.get("pair_id") == selected_micro_layout
            and coverage_row.get("repeat_id") == selected_repeat
        )
    if coverage_row.get("layout_id") != row.get("layout_id") or not pair_matches:
        raise SystemExit("Selected schedule row does not match its coverage row")
    row_index = rows.index(row)
    incomplete_prior = [
        prior["episode_id"]
        for prior in rows[:row_index]
        if prior.get("status") not in COMPLETED_STATUSES
    ]
    if incomplete_prior:
        raise SystemExit(
            "Schedule lock: earlier rows are incomplete: " + ", ".join(incomplete_prior[:5])
        )
    if row.get("status") in COMPLETED_STATUSES:
        raise SystemExit(f"Refuse to overwrite completed row {args.episode_id}")
    if args.retry_rejected and row.get("status") != RETRY_STATUS:
        raise SystemExit(
            f"Retry refused: {args.episode_id} status is {row.get('status')!r}, "
            f"expected {RETRY_STATUS!r}"
        )
    if not args.retry_rejected and row.get("status") == RETRY_STATUS:
        raise SystemExit(
            f"Retry refused: {args.episode_id} requires --retry-rejected"
        )

    order = row["order_type"]
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = dataset_root_for_order(project_root, rig, experiment, order)
    current_count = dataset_episode_count(dataset_root)
    # Every completed take is append-only, including rejected attempts and
    # replacements of historical schedule rows. Counting only rows before the
    # selected row is wrong once the full schedule has already been recorded.
    expected_current_count = recorded_source_count(rows, order)
    if current_count != expected_current_count:
        raise SystemExit(
            f"Dataset/manifest mismatch for {order}: dataset has {current_count} episodes, "
            f"expected {expected_current_count} before {args.episode_id}"
        )
    command = build_command(
        rig, experiment, row, dataset_root, dataset_root.exists(), args.allow_placeholders
    )

    print(
        f"schedule={row['planned_order']}/{len(rows)} episode_id={row['episode_id']} "
        f"pair_id={row['pair_id']} layout={row['layout_id']} "
        f"micro_layout={selected_micro_layout} repeat={selected_repeat}"
    )
    print(
        "normalized_start="
        f"pink({coverage_row['pink_x_norm']},{coverage_row['pink_y_norm']}) "
        f"cyan({coverage_row['cyan_x_norm']},{coverage_row['cyan_y_norm']}) "
        f"physical_status={coverage_row['physical_status']}"
    )
    print(
        f"order={order} source_dataset_episode={current_count} "
        f"replacement={str(args.retry_rejected).lower()}"
    )
    print(f"task={row['task']}")
    print(shlex.join(command))
    if not args.execute:
        print("\nDRY RUN ONLY: no serial port or camera was opened; no episode was created.")
        return 0

    if os.environ.get("SO101_VLA_RECORD_APPROVED") != "YES":
        raise SystemExit("Execution refused: SO101_VLA_RECORD_APPROVED is not YES")
    if os.environ.get("SO101_VLA_EPISODE_ID") != args.episode_id:
        raise SystemExit("Execution refused: SO101_VLA_EPISODE_ID does not match --episode-id")
    if rig.get("audit_status") != "VERIFIED":
        raise SystemExit("Execution refused: rig.local.yaml audit_status is not VERIFIED")
    if (
        coverage_row.get("physical_status") != "verified"
        and not args.operator_layout_confirmed
    ):
        raise SystemExit(
            f"Execution refused: {selected_micro_layout} physical_status is not verified"
        )
    if coverage_row.get("physical_status") != "verified":
        print(
            "OPERATOR LAYOUT OVERRIDE: selected micro-layout was confirmed for "
            "this session; global physical verification remains pending."
        )
    for key in ("robot.calibration_file", "teleop.calibration_file"):
        if not Path(require(rig, key)).is_file():
            raise SystemExit(f"Execution refused: missing {key}")
    for key in (
        "robot.port",
        "teleop.port",
        "cameras.front.index_or_path",
        "cameras.side.index_or_path",
    ):
        if not Path(str(require(rig, key))).exists():
            raise SystemExit(f"Execution refused: configured device path is absent: {key}")

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"lerobot-record exited with code {result.returncode}")
    new_count = dataset_episode_count(dataset_root)
    if new_count != current_count + 1:
        raise SystemExit(
            f"Recording returned successfully but episode count is {new_count}, expected {current_count + 1}"
        )
    update_manifest(
        manifest,
        fieldnames,
        rows,
        args.episode_id,
        current_count,
        args.retry_rejected,
    )
    print(f"Recorded {args.episode_id}; manifest status=recorded_pending_qc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
