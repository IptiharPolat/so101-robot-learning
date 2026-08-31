#!/usr/bin/env python3
"""Create a balanced ACT dataset without modifying either source dataset."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from lerobot.datasets import dataset_tools
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from project_config import ACT_TASK


RAW_REPO_ID = "iptihar/so101_pink_cyan_sequence_act_raw50_v1"
CORRECTION_REPO_ID = "iptihar/so101_pink_cyan_sequence_act_l1_correction_v1"
OUTPUT_REPO_ID = "iptihar/so101_pink_cyan_sequence_act_v1"
EXCLUDED_RAW_EPISODES = (1, 6, 9)


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-layout-qc", type=Path, required=True)
    parser.add_argument("--correction-layout-qc", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit(f"refusing existing output root: {args.output_root}")
    raw_rows = load_rows(args.raw_layout_qc)
    correction_rows = load_rows(args.correction_layout_qc)
    if len(raw_rows) != 50 or len(correction_rows) != 3:
        raise SystemExit(
            f"unexpected source counts: raw={len(raw_rows)} correction={len(correction_rows)}"
        )
    if any(row["observed_layout"] != "L1" for row in correction_rows):
        raise SystemExit("correction QC contains a non-L1 episode")

    kept_raw_rows = [
        row for row in raw_rows if row["episode_index"] not in EXCLUDED_RAW_EPISODES
    ]
    source_counts = Counter(row["observed_layout"] for row in kept_raw_rows)
    source_counts.update(row["observed_layout"] for row in correction_rows)
    if source_counts != Counter({f"L{i}": 10 for i in range(1, 6)}):
        raise SystemExit(f"derived layout counts are not balanced: {dict(source_counts)}")

    raw = LeRobotDataset(RAW_REPO_ID, root=args.raw_root)
    correction = LeRobotDataset(CORRECTION_REPO_ID, root=args.correction_root)
    if raw.meta.total_episodes != 50 or correction.meta.total_episodes != 3:
        raise SystemExit("source dataset metadata count mismatch")
    if raw.meta.features != correction.meta.features:
        raise SystemExit("source dataset features differ")
    if raw.meta.fps != correction.meta.fps:
        raise SystemExit("source dataset FPS differs")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="act-retained47-", dir=args.output_root.parent
    ) as temporary:
        retained_root = Path(temporary) / "retained47"
        # LeRobot 0.4.4 hardcodes AV1 in delete_episodes even when both sources
        # are H.264. Override only this process-local helper so the filtered
        # source remains codec-compatible with the H.264 correction dataset.
        original_copy_videos = dataset_tools._copy_and_reindex_videos

        def copy_videos_h264(src_dataset, dst_meta, episode_mapping):
            metadata = original_copy_videos(
                src_dataset,
                dst_meta,
                episode_mapping,
                vcodec="h264",
                pix_fmt="yuv420p",
            )
            # LeRobot 0.4.4 uses control-row length/FPS for the filtered video
            # intervals. Camera streams can contain a few more frames per
            # episode, so that approximation accumulates seconds of A/V drift.
            # The filter itself keeps the original per-episode video ranges;
            # rebuild the intervals from those exact source durations.
            for video_key in src_dataset.meta.video_keys:
                cumulative_by_file: defaultdict[tuple[int, int], float] = defaultdict(float)
                for old_index, new_index in sorted(
                    episode_mapping.items(), key=lambda item: item[1]
                ):
                    source_episode = src_dataset.meta.episodes[old_index]
                    chunk_key = f"videos/{video_key}/chunk_index"
                    file_key = f"videos/{video_key}/file_index"
                    from_key = f"videos/{video_key}/from_timestamp"
                    to_key = f"videos/{video_key}/to_timestamp"
                    output_file = (
                        int(metadata[new_index][chunk_key]),
                        int(metadata[new_index][file_key]),
                    )
                    start = cumulative_by_file[output_file]
                    duration = float(source_episode[to_key]) - float(source_episode[from_key])
                    metadata[new_index][from_key] = start
                    metadata[new_index][to_key] = start + duration
                    cumulative_by_file[output_file] += duration
            return metadata

        dataset_tools._copy_and_reindex_videos = copy_videos_h264
        try:
            retained = dataset_tools.delete_episodes(
                raw,
                episode_indices=list(EXCLUDED_RAW_EPISODES),
                output_dir=retained_root,
                repo_id="iptihar/so101_pink_cyan_sequence_act_retained47_v1",
            )
        finally:
            dataset_tools._copy_and_reindex_videos = original_copy_videos

        merged = dataset_tools.merge_datasets(
            [retained, correction],
            output_repo_id=OUTPUT_REPO_ID,
            output_dir=args.output_root,
        )
        if merged.meta.total_episodes != 50:
            raise RuntimeError(f"merged episode count is {merged.meta.total_episodes}, expected 50")

    combined = [
        ("raw50", row["episode_index"], row["observed_layout"])
        for row in kept_raw_rows
    ] + [
        ("l1_correction", row["episode_index"], row["observed_layout"])
        for row in correction_rows
    ]
    repeats: defaultdict[str, int] = defaultdict(int)
    manifest_rows = []
    for final_index, (source, source_index, layout) in enumerate(combined):
        repeats[layout] += 1
        manifest_rows.append(
            {
                "episode_id": f"act_formal_balanced_{final_index:03d}",
                "layout_id": layout,
                "repeat_id": repeats[layout],
                "first_color": "pink",
                "second_color": "cyan",
                "task": ACT_TASK,
                "source_dataset": source,
                "source_episode_index": source_index,
                "accepted": "true",
                "notes": "derived locally; source datasets preserved",
            }
        )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    provenance = {
        "output_repo_id": OUTPUT_REPO_ID,
        "output_root": str(args.output_root.resolve()),
        "push_to_hub": False,
        "raw_root": str(args.raw_root.resolve()),
        "correction_root": str(args.correction_root.resolve()),
        "excluded_raw_episode_indices": list(EXCLUDED_RAW_EPISODES),
        "retained_raw_episodes": 47,
        "correction_episodes": 3,
        "total_episodes": 50,
        "layout_counts": dict(sorted(source_counts.items())),
        "task": ACT_TASK,
    }
    args.output_provenance.parent.mkdir(parents=True, exist_ok=True)
    args.output_provenance.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
