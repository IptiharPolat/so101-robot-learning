#!/usr/bin/env python3
"""Compare arbitrary local SmolVLA checkpoints on identical saved observations."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from compare_smolvla_checkpoints_language import (
    CYAN_TASK,
    PINK_TASK,
    load_observation,
    processed_batch,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def labelled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=labelled_path, required=True)
    parser.add_argument("--source", action="append", type=labelled_path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluations/smolvla_recovery_checkpoint_language_compare"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA unavailable")
    checkpoints = dict(args.checkpoint)
    sources = dict(args.source)
    if len(checkpoints) != len(args.checkpoint) or len(sources) != len(args.source):
        raise SystemExit("FAIL: duplicate checkpoint or source label")
    for label, path in checkpoints.items():
        if not (path / "model.safetensors").is_file():
            raise SystemExit(f"FAIL: {label} missing model.safetensors under {path}")
    for label, path in sources.items():
        if not (path / "meta/info.json").is_file():
            raise SystemExit(f"FAIL: {label} is not a saved rollout dataset: {path}")

    observations = {label: load_observation(path) for label, path in sources.items()}
    device = torch.device("cuda")
    rows = []
    for checkpoint_label, checkpoint in checkpoints.items():
        policy = SmolVLAPolicy.from_pretrained(checkpoint).eval().to(device)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        torch.manual_seed(args.seed)
        noise = torch.randn(
            1, policy.config.chunk_size, policy.config.max_action_dim, device=device
        )
        for source_label, observation in observations.items():
            outputs = {}
            output_metrics = {}
            for order, task in (("pink_then_cyan", PINK_TASK), ("cyan_then_pink", CYAN_TASK)):
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                batch = processed_batch(observation, task, preprocessor, device)
                amp = torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
                with torch.inference_mode(), amp:
                    chunk = policy.predict_action_chunk(batch, noise=noise.clone())
                    chunk = postprocessor(chunk)
                outputs[order] = chunk.detach().float().cpu().numpy()
                values = outputs[order][0]
                body_step = np.max(np.abs(np.diff(values[:, :5], axis=0)), axis=1)
                gripper_step = np.abs(np.diff(values[:, 5]))
                output_metrics[order] = {
                    "first_body_gap_linf": float(
                        np.max(np.abs(values[0, :5] - observation["observation.state"][:5]))
                    ),
                    "body_step_p95": float(np.percentile(body_step, 95)),
                    "body_step_max": float(np.max(body_step)),
                    "gripper_step_p95": float(np.percentile(gripper_step, 95)),
                    "gripper_step_max": float(np.max(gripper_step)),
                }
            delta = outputs["pink_then_cyan"] - outputs["cyan_then_pink"]
            per_step_l2 = np.linalg.norm(delta[0], axis=-1)
            row = {
                "checkpoint": checkpoint_label,
                "checkpoint_path": str(checkpoint),
                "source": source_label,
                "source_path": str(sources[source_label]),
                "seed": args.seed,
                "first_action_l2": float(per_step_l2[0]),
                "first_10_mean_l2": float(per_step_l2[:10].mean()),
                "chunk_mean_l2": float(per_step_l2.mean()),
                "chunk_rmse": float(np.sqrt(np.mean(delta**2))),
                "chunk_max_abs": float(np.max(np.abs(delta))),
                "worst_first_body_gap_linf": max(
                    item["first_body_gap_linf"] for item in output_metrics.values()
                ),
                "worst_body_step_p95": max(
                    item["body_step_p95"] for item in output_metrics.values()
                ),
                "worst_body_step_max": max(
                    item["body_step_max"] for item in output_metrics.values()
                ),
                "worst_gripper_step_p95": max(
                    item["gripper_step_p95"] for item in output_metrics.values()
                ),
                "worst_gripper_step_max": max(
                    item["gripper_step_max"] for item in output_metrics.values()
                ),
            }
            rows.append(row)
            print(
                f"checkpoint={checkpoint_label} source={source_label} "
                f"first10_l2={row['first_10_mean_l2']:.6f} "
                f"chunk_rmse={row['chunk_rmse']:.6f}",
                flush=True,
            )
        del policy, preprocessor, postprocessor, noise
        gc.collect()
        torch.cuda.empty_cache()

    summary = []
    for label in checkpoints:
        subset = [row for row in rows if row["checkpoint"] == label]
        summary.append(
            {
                "checkpoint": label,
                "sources": len(subset),
                "median_first_10_mean_l2": float(
                    np.median([row["first_10_mean_l2"] for row in subset])
                ),
                "median_chunk_rmse": float(np.median([row["chunk_rmse"] for row in subset])),
                "median_worst_first_body_gap_linf": float(
                    np.median([row["worst_first_body_gap_linf"] for row in subset])
                ),
                "median_worst_body_step_p95": float(
                    np.median([row["worst_body_step_p95"] for row in subset])
                ),
                "maximum_worst_body_step": float(
                    np.max([row["worst_body_step_max"] for row in subset])
                ),
                "median_worst_gripper_step_p95": float(
                    np.median([row["worst_gripper_step_p95"] for row in subset])
                ),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "language_action_differences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
