#!/usr/bin/env python3
"""Build recovery-v2 non-destructively after all 40 supplement rows pass QC."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re

from project_config import ACT_TASK, CYAN_THEN_PINK_TASK


ROOT = Path(__file__).resolve().parents[1]
SOURCE_EPISODE_RE = re.compile(
    r"(?:replacement_raw_dataset_episode|raw_dataset_episode)=(\d+)"
)
EXPECTED_EXCLUDED_PAIRS = {
    "L2_M05",
    "L2_M08",
    "L4_M08",
    "L3_M04",
    "L1_M02",
    "L3_M02",
    "L5_M02",
    "L5_M01",
    "L5_M06",
}
EXPECTED_TASKS = {
    "pink_then_cyan": ACT_TASK,
    "cyan_then_pink": CYAN_THEN_PINK_TASK,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def source_episode(notes: str) -> int:
    matches = SOURCE_EPISODE_RE.findall(notes)
    if not matches:
        raise SystemExit(f"accepted supplement row has no raw source episode: {notes!r}")
    return int(matches[-1])


def load_dataset(repo_id: str, root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not (root / "meta/info.json").is_file():
        raise SystemExit(f"missing dataset metadata: {root}")
    return LeRobotDataset(repo_id, root=root)


def selected_copy(dataset, selected: set[int], output: Path, repo_id: str):
    total = dataset.meta.total_episodes
    invalid = selected - set(range(total))
    if invalid:
        raise SystemExit(f"selected source episodes out of range for {dataset.root}: {invalid}")
    if len(selected) != 20:
        raise SystemExit(f"expected 20 selected episodes for {dataset.root}, got {len(selected)}")
    delete = sorted(set(range(total)) - selected)
    if not delete:
        return dataset
    from lerobot.datasets.dataset_tools import delete_episodes

    if output.exists():
        raise SystemExit(f"refuse existing selected-data output: {output}")
    return delete_episodes(dataset, delete, output_dir=output, repo_id=repo_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--old-root",
        type=Path,
        default=ROOT / "outputs/datasets/vla_pink_cyan_order_clean_v1",
    )
    parser.add_argument(
        "--pink-root",
        type=Path,
        default=ROOT / "outputs/datasets/vla_recovery_pink_then_cyan_raw_v1",
    )
    parser.add_argument(
        "--cyan-root",
        type=Path,
        default=ROOT / "outputs/datasets/vla_recovery_cyan_then_pink_raw_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/datasets/vla_pink_cyan_order_recovery_v2",
    )
    parser.add_argument(
        "--intermediate",
        type=Path,
        default=ROOT / "outputs/datasets/vla_recovery_build_intermediate",
        help="Non-overwriting workspace for selected source subsets.",
    )
    parser.add_argument(
        "--reuse-complete-intermediate",
        action="store_true",
        help="Reuse already validated 82/20/20 selected subsets after a merge-only failure.",
    )
    args = parser.parse_args()

    curation_path = ROOT / "manifests/vla_recovery_existing_curation.csv"
    supplement_path = ROOT / "manifests/vla_recovery_episode_schedule.csv"
    clean_selection_path = ROOT / "manifests/vla_clean_selection.csv"
    provenance_path = ROOT / "manifests/vla_recovery_v2_provenance.csv"
    curation = read_csv(curation_path)
    supplement = read_csv(supplement_path)
    clean_selection = read_csv(clean_selection_path)

    excluded = {row["pair_id"] for row in curation if row["include_in_recovery_v2"] == "false"}
    retained = [row for row in curation if row["include_in_recovery_v2"] == "true"]
    if len(curation) != 50 or len(retained) != 41 or excluded != EXPECTED_EXCLUDED_PAIRS:
        raise SystemExit("frozen old-data curation is not the expected 41/9 pair split")
    if len(supplement) != 40 or Counter(row["order_type"] for row in supplement) != Counter(
        {"pink_then_cyan": 20, "cyan_then_pink": 20}
    ):
        raise SystemExit("supplement schedule is not 40 rows balanced 20/20")
    if len({row["pair_id"] for row in supplement}) != 20:
        raise SystemExit("supplement does not contain 20 strict pairs")

    print("old source: 100 episodes -> retain 82 episodes / 41 complete pairs")
    print("supplement target: 40 accepted episodes / 20 complete pairs")
    print("recovery-v2 target: 122 episodes / 61 per instruction")
    print(f"output: {args.output.resolve()}")
    if not args.execute:
        print("DRY RUN ONLY: no dataset was read, copied, deleted, merged, or uploaded.")
        return 0

    if os.environ.get("SMOLVLA_RECOVERY_BUILD_APPROVED") != "YES":
        raise SystemExit("execution refused: SMOLVLA_RECOVERY_BUILD_APPROVED is not YES")
    if args.output.exists() or provenance_path.exists():
        raise SystemExit("refuse to overwrite recovery-v2 output or provenance")
    required_qc = ("initial_both_visible", "initial_pose_match", "smooth_motion_qc")
    for row in supplement:
        if row["status"] != "accepted" or row["accepted"].lower() != "true":
            raise SystemExit(f"supplement row is not accepted: {row['episode_id']}")
        if any(row[field].lower() != "true" for field in required_qc):
            raise SystemExit(f"supplement row lacks complete QC flags: {row['episode_id']}")
        if row["task"] != EXPECTED_TASKS[row["order_type"]]:
            raise SystemExit(f"noncanonical task in {row['episode_id']}")

    old = load_dataset("iptihar/so101_vla_pink_cyan_order_clean_v1", args.old_root.resolve())
    pink = load_dataset(
        "iptihar/so101_vla_recovery_pink_then_cyan_raw_v1", args.pink_root.resolve()
    )
    cyan = load_dataset(
        "iptihar/so101_vla_recovery_cyan_then_pink_raw_v1", args.cyan_root.resolve()
    )
    if old.meta.total_episodes != 100:
        raise SystemExit(f"old clean-v1 count is {old.meta.total_episodes}, expected 100")

    # Some valid LeRobot v3 datasets consolidate all episode metadata rows in
    # one parquet while retaining the original per-episode file indices inside
    # those rows. LeRobot 0.4.4 delete_episodes trusts the embedded index and
    # then looks for nonexistent file-001, file-002, ... paths. Normalize only
    # the in-memory lookup metadata when exactly one physical parquet exists;
    # the source files themselves remain untouched.
    old_episode_files = sorted((old.root / "meta/episodes").glob("**/*.parquet"))
    if len(old_episode_files) == 1:
        import pyarrow.parquet as pq

        consolidated = pq.read_table(old_episode_files[0], columns=["episode_index"])
        if consolidated.num_rows != old.meta.total_episodes:
            raise SystemExit(
                "single old episode-metadata parquet does not contain all episodes"
            )
        chunk_index = int(old_episode_files[0].parent.name.removeprefix("chunk-"))
        file_index = int(old_episode_files[0].stem.removeprefix("file-"))
        from lerobot.datasets import dataset_tools

        original_loader = dataset_tools._load_episode_with_stats
        consolidated_rows = pq.read_table(old_episode_files[0]).to_pandas()

        def load_episode_with_stats_compat(src_dataset, episode_idx):
            if src_dataset.root.resolve() == old.root.resolve():
                matches = consolidated_rows[
                    consolidated_rows["episode_index"] == episode_idx
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one consolidated metadata row for episode {episode_idx}"
                    )
                return matches.iloc[0].to_dict()
            return original_loader(src_dataset, episode_idx)

        dataset_tools._load_episode_with_stats = load_episode_with_stats_compat
        print(
            "compatibility: normalized consolidated old episode metadata "
            f"lookup to chunk={chunk_index} file={file_index} in memory only"
        )

    excluded_episodes = sorted(
        int(row[key])
        for row in curation
        if row["include_in_recovery_v2"] == "false"
        for key in ("pink_then_cyan_merged_episode", "cyan_then_pink_merged_episode")
    )
    if len(excluded_episodes) != 18:
        raise SystemExit("old exclusion list must contain 18 episodes")

    from lerobot.datasets.dataset_tools import delete_episodes

    intermediate = args.intermediate.resolve()
    if intermediate.exists() and not args.reuse_complete_intermediate:
        raise SystemExit(f"refuse existing intermediate output: {intermediate}")
    if args.reuse_complete_intermediate:
        if not intermediate.is_dir():
            raise SystemExit(f"reusable intermediate does not exist: {intermediate}")
        old_selected = load_dataset(
            "iptihar/so101_vla_pink_cyan_order_old_selected_v2",
            intermediate / "old_selected_82",
        )
        pink_selected = load_dataset(
            "iptihar/so101_vla_recovery_pink_then_cyan_selected_v1",
            intermediate / "pink_selected_20",
        )
        cyan_selected = load_dataset(
            "iptihar/so101_vla_recovery_cyan_then_pink_selected_v1",
            intermediate / "cyan_selected_20",
        )
        actual_counts = [
            old_selected.meta.total_episodes,
            pink_selected.meta.total_episodes,
            cyan_selected.meta.total_episodes,
        ]
        if actual_counts != [82, 20, 20]:
            raise SystemExit(f"reusable intermediate count mismatch: {actual_counts}")
        print("reuse: validated complete intermediate subsets 82/20/20")
    else:
        old_selected = delete_episodes(
            old,
            excluded_episodes,
            output_dir=intermediate / "old_selected_82",
            repo_id="iptihar/so101_vla_pink_cyan_order_old_selected_v2",
        )
        selected_sources = {
            order: {
                source_episode(row["notes"])
                for row in supplement
                if row["order_type"] == order
            }
            for order in EXPECTED_TASKS
        }
        pink_selected = selected_copy(
            pink,
            selected_sources["pink_then_cyan"],
            intermediate / "pink_selected_20",
            "iptihar/so101_vla_recovery_pink_then_cyan_selected_v1",
        )
        cyan_selected = selected_copy(
            cyan,
            selected_sources["cyan_then_pink"],
            intermediate / "cyan_selected_20",
            "iptihar/so101_vla_recovery_cyan_then_pink_selected_v1",
        )

    # LeRobot 0.4.4 packet-concatenation can produce duplicate DTS when AV1
    # files from multiple source datasets are appended into one MP4. Rotate at
    # 1 MB (all selected MP4s are larger) so each source video is copied into a
    # separate destination file with zero timestamp offset and no remux concat.
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    selected_datasets = [old_selected, pink_selected, cyan_selected]
    aggregate_datasets(
        repo_ids=[dataset.repo_id for dataset in selected_datasets],
        aggr_repo_id="iptihar/so101_vla_pink_cyan_order_recovery_v2",
        roots=[dataset.root for dataset in selected_datasets],
        aggr_root=args.output.resolve(),
        video_files_size_in_mb=1,
    )
    merged = LeRobotDataset(
        "iptihar/so101_vla_pink_cyan_order_recovery_v2",
        root=args.output.resolve(),
    )
    if merged.meta.total_episodes != 122:
        raise SystemExit(f"merged count is {merged.meta.total_episodes}, expected 122")
    if merged.meta.fps != 30:
        raise SystemExit(f"merged FPS is {merged.meta.fps}, expected 30")
    features = merged.meta.features
    if features["action"]["shape"] != (6,) and features["action"]["shape"] != [6]:
        raise SystemExit("merged action is not 6D")
    if features["observation.state"]["shape"] != (6,) and features["observation.state"]["shape"] != [6]:
        raise SystemExit("merged state is not 6D")
    for key in ("observation.images.front", "observation.images.side"):
        if key not in features:
            raise SystemExit(f"merged dataset lacks {key}")
        if list(features[key]["shape"]) != [480, 640, 3]:
            raise SystemExit(f"merged camera shape mismatch for {key}: {features[key]['shape']}")

    import pandas as pd

    episode_files = sorted((args.output / "meta/episodes").glob("**/*.parquet"))
    if not episode_files:
        raise SystemExit("merged dataset has no episode metadata parquet")
    episode_rows = pd.concat(
        (pd.read_parquet(path) for path in episode_files), ignore_index=True
    )
    task_counts = Counter(item[0] for item in episode_rows["tasks"])
    if task_counts != Counter({ACT_TASK: 61, CYAN_THEN_PINK_TASK: 61}):
        raise SystemExit(f"merged task balance mismatch: {task_counts}")

    clean_by_episode = {int(row["merged_episode"]): row for row in clean_selection}
    retained_old_episodes = sorted(set(range(100)) - set(excluded_episodes))
    provenance: list[dict[str, object]] = []
    for new_index, old_index in enumerate(retained_old_episodes):
        row = clean_by_episode[old_index]
        provenance.append(
            {
                "recovery_v2_episode": new_index,
                "source_group": "clean_v1_retained",
                "source_episode": old_index,
                "pair_id": row["pair_id"],
                "order_type": row["order_type"],
                "source_episode_id": row["episode_id"],
            }
        )
    offset = len(provenance)
    for order in ("pink_then_cyan", "cyan_then_pink"):
        rows_by_source = sorted(
            (
                source_episode(row["notes"]),
                row,
            )
            for row in supplement
            if row["order_type"] == order
        )
        for position, (raw_index, row) in enumerate(rows_by_source):
            provenance.append(
                {
                    "recovery_v2_episode": offset + position,
                    "source_group": f"recovery_{order}_selected",
                    "source_episode": raw_index,
                    "pair_id": row["pair_id"],
                    "order_type": order,
                    "source_episode_id": row["episode_id"],
                }
            )
        offset += len(rows_by_source)
    if len(provenance) != 122:
        raise SystemExit("provenance length mismatch")
    with provenance_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(provenance)

    info = json.loads((args.output / "meta/info.json").read_text(encoding="utf-8"))
    print(f"built={args.output.resolve()}")
    print(f"episodes={info['total_episodes']} frames={info['total_frames']}")
    print(f"provenance={provenance_path}")
    print("push_to_hub=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
