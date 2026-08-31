#!/usr/bin/env python3
"""Plot offline ACT gripper comparison CSVs."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_act_temporal_gripper import signal_metrics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/evaluations/act_gripper_temporal_offline_compare"
TRIALS = [
    "ACT30K-L1-TEMPORAL-001",
    "ACT30K-L1-TEMPORAL-002",
]


fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
active_summaries = []
for axis, evaluation_id in zip(axes, TRIALS, strict=True):
    data = pd.read_csv(SOURCE / f"{evaluation_id.lower()}_gripper.csv")
    recorded = data["recorded_gripper"].to_numpy()
    changing = np.flatnonzero(np.abs(np.diff(recorded)) > 0.05)
    active_frames = (
        min(len(data), int(changing[-1]) + 2 + 25)
        if len(changing)
        else min(len(data), 25)
    )
    data = data.iloc[:active_frames]
    active_summaries.append(
        {
            "evaluation_id": evaluation_id,
            "active_frames": active_frames,
            "active_duration_s": round(active_frames / 25, 3),
            "recorded_temporal": signal_metrics(data["recorded_gripper"].to_numpy()),
            "offline_temporal_0_01": signal_metrics(
                data["offline_temporal_gripper"].to_numpy()
            ),
            "offline_no_ensemble": signal_metrics(
                data["offline_no_ensemble_gripper"].to_numpy()
            ),
        }
    )
    axis.plot(
        data["timestamp_s"],
        data["recorded_gripper"],
        label="recorded temporal action",
        linewidth=2.0,
        color="#222222",
    )
    axis.plot(
        data["timestamp_s"],
        data["offline_temporal_gripper"],
        label="offline temporal 0.01",
        linewidth=1.4,
        color="#2878b5",
    )
    axis.plot(
        data["timestamp_s"],
        data["offline_no_ensemble_gripper"],
        label="offline no ensemble",
        linewidth=0.8,
        alpha=0.75,
        color="#d95319",
    )
    axis.set_title(evaluation_id)
    axis.set_ylabel("Gripper command")
    axis.grid(alpha=0.25)
axes[-1].set_xlabel("Time (s)")
axes[0].legend(loc="upper right", ncol=3, fontsize=8)
fig.suptitle("ACT gripper output on identical recorded observations")
fig.tight_layout()
output = SOURCE / "gripper_temporal_on_off.png"
fig.savefig(output, dpi=160)
summary_output = SOURCE / "active_window_summary.json"
summary_output.write_text(json.dumps(active_summaries, indent=2), encoding="utf-8")
print(output)
print(summary_output)
