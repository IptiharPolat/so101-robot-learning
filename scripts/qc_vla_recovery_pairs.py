#!/usr/bin/env python3
"""Read-only numeric QC for the strict SmolVLA recovery pairs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_EPISODE = re.compile(r"(?:replacement_)?raw_dataset_episode=(\d+)")


def selected_episode(row: dict[str, str]) -> int:
    matches = RAW_EPISODE.findall(row.get("notes", ""))
    if not matches:
        raise ValueError(f"{row['episode_id']}: raw dataset episode is missing")
    return int(matches[-1])


def load(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no data parquet under {root}")
    return pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["episode_index", "action", "observation.state"],
            )
            for path in paths
        ],
        ignore_index=True,
    )


def vectors(series: pd.Series) -> np.ndarray:
    value = np.stack([np.asarray(item, dtype=np.float64) for item in series])
    if value.ndim != 2 or value.shape[1] != 6 or not np.isfinite(value).all():
        raise ValueError(f"expected finite [N,6], got {value.shape}")
    return value


def episode_metrics(data: pd.DataFrame, episode_index: int) -> dict:
    episode = data[data["episode_index"] == episode_index]
    if episode.empty:
        raise ValueError(f"raw episode {episode_index} is absent")
    action = vectors(episode["action"])
    state = vectors(episode["observation.state"])
    delta = np.diff(action, axis=0)
    body = np.max(np.abs(delta[:, :5]), axis=1)
    gripper = np.abs(delta[:, 5])
    return {
        "frames": int(len(episode)),
        "initial_state": state[0].tolist(),
        "body_step_p99": float(np.percentile(body, 99)),
        "body_step_max": float(np.max(body)),
        "gripper_step_max": float(np.max(gripper)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "manifests/vla_recovery_episode_schedule.csv",
    )
    parser.add_argument(
        "--cyan-root",
        type=Path,
        default=ROOT / "outputs/datasets/vla_recovery_cyan_then_pink_raw_v1",
    )
    parser.add_argument(
        "--pink-root",
        type=Path,
        default=ROOT / "outputs/datasets/vla_recovery_pink_then_cyan_raw_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/vla_recovery_numeric_pair_qc.json",
    )
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 40:
        raise SystemExit(f"expected 40 manifest rows, got {len(rows)}")

    datasets = {
        "cyan_then_pink": load(args.cyan_root),
        "pink_then_cyan": load(args.pink_root),
    }
    pairs = []
    for position in range(0, len(rows), 2):
        members = rows[position : position + 2]
        if members[0]["pair_id"] != members[1]["pair_id"]:
            raise SystemExit(f"manifest rows {position + 1}-{position + 2} are not paired")
        takes = []
        for row in members:
            source_episode = selected_episode(row)
            metrics = episode_metrics(datasets[row["order_type"]], source_episode)
            takes.append(
                {
                    "episode_id": row["episode_id"],
                    "order_type": row["order_type"],
                    "source_episode": source_episode,
                    **metrics,
                }
            )
        state_linf = float(
            np.max(
                np.abs(
                    np.asarray(takes[0]["initial_state"])
                    - np.asarray(takes[1]["initial_state"])
                )
            )
        )
        smooth = all(
            take["body_step_p99"] <= 6
            and take["body_step_max"] <= 10
            and take["gripper_step_max"] <= 10
            for take in takes
        )
        pairs.append(
            {
                "pair_id": members[0]["pair_id"],
                "layout": members[0]["base_micro_layout_id"],
                "initial_state_linf": state_linf,
                "initial_state_pass": state_linf <= 3,
                "smooth_motion_pass": smooth,
                "numeric_pass": state_linf <= 3 and smooth,
                "takes": takes,
            }
        )

    payload = {
        "manifest": str(args.manifest.resolve()),
        "pairs": len(pairs),
        "numeric_pass_count": sum(pair["numeric_pass"] for pair in pairs),
        "numeric_fail_pair_ids": [
            pair["pair_id"] for pair in pairs if not pair["numeric_pass"]
        ],
        "pair_results": pairs,
        "limitations": (
            "Read-only numeric gate only. Visual initial-position match, exact demonstrated "
            "order, contacts, and C1/C2 outcomes still require video/operator review."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"numeric_pass={payload['numeric_pass_count']}/{payload['pairs']}")
    if payload["numeric_fail_pair_ids"]:
        print("numeric_fail_pair_ids=" + ",".join(payload["numeric_fail_pair_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
