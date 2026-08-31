#!/usr/bin/env python3
"""Compare SmolVLA checkpoints under opposite task strings on fixed observations."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import prepare_observation_for_inference


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/cloud_runs/smolvla_full_30k_b8_wandb_v1"
EVAL = ROOT / "outputs/evaluations"
PINK_TASK = (
    "First pick up the pink cube and place it in the center target area, then "
    "pick up the cyan cube and place it in the center target area."
)
CYAN_TASK = (
    "First pick up the cyan cube and place it in the center target area, then "
    "pick up the pink cube and place it in the center target area."
)
SOURCES = {
    "L1_M08": "smolvla_30k_l1_m08_pink_then_cyan_retry_001",
    "L2_M08": "smolvla_30k_l2_m08_cyan_then_pink_001",
    "L3_M08": "smolvla_30k_l3_m08_cyan_then_pink_001",
    "L4_M08": "smolvla_30k_l4_m08_pink_then_cyan_001",
    "L5_M08": "smolvla_30k_l5_m08_cyan_then_pink_001",
}


def first_rgb(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_observation(root: Path) -> dict[str, np.ndarray]:
    row = pd.read_parquet(root / "data/chunk-000/file-000.parquet").iloc[0]
    return {
        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
        "observation.images.front": first_rgb(
            root / "videos/observation.images.front/chunk-000/file-000.mp4"
        ),
        "observation.images.side": first_rgb(
            root / "videos/observation.images.side/chunk-000/file-000.mp4"
        ),
    }


def processed_batch(observation, task, preprocessor, device):
    frame = prepare_observation_for_inference(
        observation.copy(), device=device, task=task, robot_type="so_follower"
    )
    return preprocessor(frame)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["005000", "010000", "015000", "020000", "025000", "030000"],
    )
    parser.add_argument("--layouts", nargs="+", default=list(SOURCES))
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evaluations/smolvla_checkpoint_language_compare",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA unavailable")
    unknown = sorted(set(args.layouts) - set(SOURCES))
    if unknown:
        raise SystemExit(f"FAIL: unknown layouts: {unknown}")

    args.output.mkdir(parents=True, exist_ok=True)
    observations = {
        layout: load_observation(EVAL / SOURCES[layout]) for layout in args.layouts
    }
    device = torch.device("cuda")
    rows = []

    for step in args.checkpoints:
        checkpoint = RUN / "checkpoints" / step / "pretrained_model"
        if not (checkpoint / "model.safetensors").is_file():
            raise SystemExit(f"FAIL: missing checkpoint {checkpoint}")
        policy = SmolVLAPolicy.from_pretrained(checkpoint).eval().to(device)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        torch.manual_seed(args.seed)
        noise = torch.randn(
            1,
            policy.config.chunk_size,
            policy.config.max_action_dim,
            device=device,
        )

        for layout in args.layouts:
            outputs = {}
            for order, task in (("pink_then_cyan", PINK_TASK), ("cyan_then_pink", CYAN_TASK)):
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                batch = processed_batch(observations[layout], task, preprocessor, device)
                amp = (
                    torch.autocast(device_type="cuda")
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.inference_mode(), amp:
                    chunk = policy.predict_action_chunk(batch, noise=noise.clone())
                    chunk = postprocessor(chunk)
                outputs[order] = chunk.detach().float().cpu().numpy()

            pink = outputs["pink_then_cyan"]
            cyan = outputs["cyan_then_pink"]
            delta = pink - cyan
            per_step_l2 = np.linalg.norm(delta[0], axis=-1)
            rows.append(
                {
                    "checkpoint": step,
                    "layout_id": layout,
                    "source_dataset": SOURCES[layout],
                    "seed": args.seed,
                    "first_action_l2": float(per_step_l2[0]),
                    "first_10_mean_l2": float(per_step_l2[:10].mean()),
                    "chunk_mean_l2": float(per_step_l2.mean()),
                    "chunk_rmse": float(np.sqrt(np.mean(delta**2))),
                    "chunk_max_abs": float(np.max(np.abs(delta))),
                    "pink_first_action": json.dumps(pink[0, 0].round(6).tolist()),
                    "cyan_first_action": json.dumps(cyan[0, 0].round(6).tolist()),
                }
            )
            print(
                f"checkpoint={step} layout={layout} "
                f"first10_l2={per_step_l2[:10].mean():.6f} "
                f"chunk_rmse={np.sqrt(np.mean(delta**2)):.6f}",
                flush=True,
            )

        del policy, preprocessor, postprocessor, noise
        gc.collect()
        torch.cuda.empty_cache()

    csv_path = args.output / "language_action_differences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for step in args.checkpoints:
        subset = [row for row in rows if row["checkpoint"] == step]
        summary.append(
            {
                "checkpoint": step,
                "layouts": len(subset),
                "median_first_10_mean_l2": float(
                    np.median([row["first_10_mean_l2"] for row in subset])
                ),
                "median_chunk_rmse": float(
                    np.median([row["chunk_rmse"] for row in subset])
                ),
            }
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
