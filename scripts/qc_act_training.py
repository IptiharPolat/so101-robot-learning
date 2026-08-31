#!/usr/bin/env python3
"""Offline QC for a completed ACT training run before model publication."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


METRIC_PATTERN = re.compile(r"\b(loss|grdn|lr|updt_s|data_s):([^\s]+)")
ERROR_PATTERNS = (
    re.compile(r"Traceback"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemory", re.IGNORECASE),
)


def numeric(text: str) -> float:
    return float(text.rstrip(","))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--exit-code-file", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    log_path = args.log.expanduser().resolve()
    gpu_path = args.gpu_csv.expanduser().resolve()
    exit_path = args.exit_code_file.expanduser().resolve()
    for path in (output_dir, log_path, gpu_path, exit_path):
        if not path.exists():
            raise SystemExit(f"FAIL: required path not found: {path}")

    exit_code = int(exit_path.read_text().strip())
    if exit_code != 0:
        raise SystemExit(f"FAIL: training exit code is {exit_code}")

    text = log_path.read_text(errors="replace")
    errors = [pattern.pattern for pattern in ERROR_PATTERNS if pattern.search(text)]
    if errors:
        raise SystemExit(f"FAIL: training log contains fatal patterns: {errors}")
    if "End of training" not in text:
        raise SystemExit("FAIL: training log has no 'End of training' marker")

    metric_rows: list[dict[str, float]] = []
    for line in text.splitlines():
        if "step:" not in line:
            continue
        row = {name: numeric(value) for name, value in METRIC_PATTERN.findall(line)}
        if row:
            metric_rows.append(row)
    if not metric_rows:
        raise SystemExit("FAIL: no training metric rows found")
    non_finite = [
        index
        for index, row in enumerate(metric_rows)
        if any(not math.isfinite(value) for value in row.values())
    ]
    if non_finite:
        raise SystemExit(f"FAIL: non-finite metrics in rows: {non_finite[:10]}")

    checkpoint_root = output_dir / "checkpoints"
    required_steps = [10000, 20000, 30000]
    missing_checkpoints = [
        step
        for step in required_steps
        if not (checkpoint_root / f"{step:06d}" / "pretrained_model" / "model.safetensors").is_file()
    ]
    if missing_checkpoints:
        raise SystemExit(f"FAIL: missing checkpoints: {missing_checkpoints}")
    last = checkpoint_root / "last"
    if not last.is_symlink() or last.resolve() != (checkpoint_root / "030000").resolve():
        raise SystemExit("FAIL: checkpoints/last does not resolve to 030000")

    memory_values: list[float] = []
    utilization_values: list[float] = []
    power_values: list[float] = []
    with gpu_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            memory_values.append(numeric(row[1]))
            utilization_values.append(numeric(row[2]))
            power_values.append(numeric(row[3]))
    if not memory_values:
        raise SystemExit("FAIL: GPU monitor CSV contains no samples")

    losses = [row["loss"] for row in metric_rows if "loss" in row]
    report = {
        "status": "PASS",
        "exit_code": exit_code,
        "metric_rows": len(metric_rows),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "checkpoints": required_steps,
        "last_checkpoint": str(last.resolve()),
        "gpu_samples": len(memory_values),
        "gpu_peak_memory_mib": max(memory_values),
        "gpu_peak_utilization_percent": max(utilization_values),
        "gpu_peak_power_w": max(power_values),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
