#!/usr/bin/env python3
"""Benchmark ACT temporal-ensemble inference without opening robot hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from project_config import ACT_TASK, load_config, require


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "outputs/evaluations/act_30k_single_trial_l1_001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_rgb_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise SystemExit(f"FAIL: could not decode first frame from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(np.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/act_eval_temporal_diagnostic.yaml")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 5:
        raise SystemExit("FAIL: use at least 1 warmup and 5 measured iterations")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is unavailable")

    config = load_config(args.config)
    policy_cfg = require(config, "policy")
    if require(policy_cfg, "n_action_steps") != 1:
        raise SystemExit("FAIL: diagnostic n_action_steps must be 1")
    if float(require(policy_cfg, "temporal_ensemble_coeff")) != 0.01:
        raise SystemExit("FAIL: diagnostic temporal_ensemble_coeff must be 0.01")

    snapshot = Path(
        snapshot_download(
            repo_id=str(require(policy_cfg, "repo_id")),
            revision=str(require(policy_cfg, "revision")),
            local_files_only=True,
        )
    )
    actual_sha = sha256(snapshot / "model.safetensors")
    expected_sha = str(require(policy_cfg, "weight_sha256"))
    if actual_sha != expected_sha:
        raise SystemExit(f"FAIL: model SHA mismatch: {actual_sha}")

    overrides = [
        "--device=cuda",
        f"--use_amp={'true' if bool(policy_cfg.get('use_amp', False)) else 'false'}",
        "--n_action_steps=1",
        "--temporal_ensemble_coeff=0.01",
    ]
    loaded_cfg = PreTrainedConfig.from_pretrained(
        snapshot,
        local_files_only=True,
        cli_overrides=overrides,
    )
    if not isinstance(loaded_cfg, ACTConfig):
        raise SystemExit(f"FAIL: expected ACTConfig, got {type(loaded_cfg).__name__}")
    loaded_cfg.pretrained_path = str(snapshot)
    policy = ACTPolicy.from_pretrained(
        snapshot,
        config=loaded_cfg,
        local_files_only=True,
    ).to("cuda")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=loaded_cfg,
        pretrained_path=str(snapshot),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    parquet = SOURCE_DATASET / "data/chunk-000/file-000.parquet"
    first_row = pd.read_parquet(parquet).iloc[0]
    observation = {
        "observation.state": np.asarray(
            first_row["observation.state"], dtype=np.float32
        ).copy(),
        "observation.images.front": first_rgb_frame(
            SOURCE_DATASET
            / "videos/observation.images.front/chunk-000/file-000.mp4"
        ),
        "observation.images.side": first_rgb_frame(
            SOURCE_DATASET
            / "videos/observation.images.side/chunk-000/file-000.mp4"
        ),
    }
    device = torch.device("cuda")
    policy.reset()

    for _ in range(args.warmup):
        predict_action(
            observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            use_amp=loaded_cfg.use_amp,
            task=ACT_TASK,
            robot_type="so_follower",
        )
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    last_action = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        last_action = predict_action(
            observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            use_amp=loaded_cfg.use_amp,
            task=ACT_TASK,
            robot_type="so_follower",
        )
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000)

    mean_ms = statistics.fmean(latencies_ms)
    p95_ms = percentile(latencies_ms, 0.95)
    target_period_ms = 1000 / 30
    inference_budget_ms = 25.0
    status = "PASS" if mean_ms <= inference_budget_ms and p95_ms <= inference_budget_ms else "FAIL"
    result = {
        "status": status,
        "hardware_opened": False,
        "source_observation": str(SOURCE_DATASET.relative_to(PROJECT_ROOT)),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "policy_revision": require(policy_cfg, "revision"),
        "weight_sha256": actual_sha,
        "chunk_size": loaded_cfg.chunk_size,
        "n_action_steps": loaded_cfg.n_action_steps,
        "temporal_ensemble_coeff": loaded_cfg.temporal_ensemble_coeff,
        "use_amp": loaded_cfg.use_amp,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "target_fps": 30,
        "target_period_ms": round(target_period_ms, 3),
        "inference_budget_ms": inference_budget_ms,
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(statistics.median(latencies_ms), 3),
        "p95_ms": round(p95_ms, 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "sustained_fps_from_mean": round(1000 / mean_ms, 2),
        "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "last_action": [round(float(value), 4) for value in last_action.squeeze(0)],
        "gate": "mean and p95 must both be <= 25 ms, reserving at least 8.333 ms for camera and robot I/O",
    }
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
