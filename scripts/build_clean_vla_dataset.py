#!/usr/bin/env python3
"""Build local cleaned and merged SmolVLA datasets without touching raw roots."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from project_config import load_config, require


REPLACEMENT_RE = re.compile(r"replacement_raw_dataset_episode=(\d+)")
RAW_RE = re.compile(r"(?:^|; )raw_dataset_episode=(\d+)")
REJECTED_CYAN = {1, 3}


def source_episode(notes: str) -> int:
    replacements = REPLACEMENT_RE.findall(notes)
    if replacements:
        return int(replacements[-1])
    match = RAW_RE.search(notes)
    if not match:
        raise SystemExit(f"accepted row has no source episode note: {notes!r}")
    return int(match.group(1))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    experiment = load_config(root / "configs/smolvla_experiment.yaml")
    rig = load_config(root / "configs/rig.local.yaml")
    manifest_path = root / require(experiment, "collection.manifest")
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 100 or any(row["accepted"].lower() != "true" for row in rows):
        raise SystemExit("clean build requires exactly 100 accepted manifest rows")

    raw_pink = Path(require(rig, "recording.dataset_roots.vla.pink_then_cyan")).resolve()
    raw_cyan = Path(require(rig, "recording.dataset_roots.vla.cyan_then_pink")).resolve()
    clean_cyan = root / "outputs/datasets/vla_cyan_then_pink_clean_v1"
    merged_root = root / "outputs/datasets/vla_pink_cyan_order_clean_v1"
    for path in (raw_pink, raw_cyan):
        if not (path / "meta/info.json").is_file():
            raise SystemExit(f"missing raw dataset metadata: {path}")
    for path in (clean_cyan, merged_root):
        if path.exists():
            raise SystemExit(f"refuse to overwrite existing output: {path}")

    selected: dict[str, list[tuple[str, int]]] = {
        "pink_then_cyan": [],
        "cyan_then_pink": [],
    }
    for row in rows:
        selected[row["order_type"]].append((row["episode_id"], source_episode(row["notes"])))
    if set(selected["cyan_then_pink"][i][1] for i in range(len(selected["cyan_then_pink"]))) != set(range(52)) - REJECTED_CYAN:
        raise SystemExit("Cyan source selection does not equal 50 valid episodes after exclusions")
    if set(source for _, source in selected["pink_then_cyan"]) != set(range(50)):
        raise SystemExit("Pink source selection does not equal source episodes 0..49")

    from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    pink = LeRobotDataset(
        "iptihar/so101_vla_pink_then_cyan_v1", root=raw_pink
    )
    cyan = LeRobotDataset(
        "iptihar/so101_vla_cyan_then_pink_v1", root=raw_cyan
    )
    if pink.meta.total_episodes != 50 or cyan.meta.total_episodes != 52:
        raise SystemExit(
            f"unexpected raw counts: pink={pink.meta.total_episodes}, cyan={cyan.meta.total_episodes}"
        )

    print(f"Creating cleaned Cyan dataset at {clean_cyan}")
    clean = delete_episodes(
        cyan,
        episode_indices=sorted(REJECTED_CYAN),
        output_dir=clean_cyan,
        repo_id="iptihar/so101_vla_cyan_then_pink_clean_v1",
    )
    if clean.meta.total_episodes != 50:
        raise SystemExit(f"clean Cyan count is {clean.meta.total_episodes}, expected 50")

    print(f"Creating merged dataset at {merged_root}")
    merged = merge_datasets(
        [pink, clean],
        output_repo_id="iptihar/so101_vla_pink_cyan_order_clean_v1",
        output_dir=merged_root,
    )
    if merged.meta.total_episodes != 100:
        raise SystemExit(f"merged count is {merged.meta.total_episodes}, expected 100")

    selection_path = root / "manifests/vla_clean_selection.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "episode_id", "pair_id", "order_type", "raw_source_episode",
                "clean_source_episode", "merged_episode", "excluded_raw_episodes",
            ],
        )
        writer.writeheader()
        for row in rows:
            raw = source_episode(row["notes"])
            clean_index = raw - sum(rejected < raw for rejected in REJECTED_CYAN) if row["order_type"] == "cyan_then_pink" else raw
            merged_index = clean_index if row["order_type"] == "pink_then_cyan" else 50 + clean_index
            writer.writerow(
                {
                    "episode_id": row["episode_id"],
                    "pair_id": row["pair_id"],
                    "order_type": row["order_type"],
                    "raw_source_episode": raw,
                    "clean_source_episode": clean_index,
                    "merged_episode": merged_index,
                    "excluded_raw_episodes": "1,3" if row["order_type"] == "cyan_then_pink" else "",
                }
            )

    print(f"clean_cyan_episodes={clean.meta.total_episodes}")
    print(f"merged_episodes={merged.meta.total_episodes}")
    print(f"selection_manifest={selection_path}")
    print("push_to_hub=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
