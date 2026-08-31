# SmolVLA recovery-v2 20K offline diagnostic

Date: 2026-08-31

Status: `OFFLINE_DIAGNOSTIC_COMPLETE_FORMAL_RATE_BLOCKED_BY_RESET_MISMATCH`

This audit is offline only. It did not open a camera or serial port, control the
robot, record, upload, or train. Operator-reported outcomes are kept separate
from metrics computed from saved data.

## Data integrity

- Found 14 recovery-20K rollout datasets: 12 daylight formal attempts and 2
  night diagnostics.
- All state/action arrays are finite and 6D.
- All 28 videos decode to their advertised frame count at 640x480 and 30 FPS.
- No black, overbright, or exact consecutive duplicate frames were detected.
- The two night trials remain excluded from formal behavior claims.

Machine-readable evidence is in
`reports/smolvla_recovery20k_rollout_audit.json`.

## Paired-start validity

The training pair gate required the two opposite instructions to begin within
5 normalized joint units and from the same physical cube layout. The current
screening rollouts do not consistently meet that requirement.

| Layout | Initial arm-state L-infinity | Initial visual check | Pair status |
|---|---:|---|---|
| L1_M09 | 5.02 | Cyan agrees within about 5 pixels; overall scene appears close | Borderline, just outside the frozen pose gate |
| L2_M09 | 2.64 | Initial scene is visually close; Cyan agrees within about 1 pixel | Pass |
| L3_M09 | 10.79 | Cyan is visible in one start and absent/occluded in the other | Fail |
| L4_M09 | 103.68 for attempt 1; 5.79 for attempt 2 | Attempt 1 uses a clearly different arm pose; attempt 2 contains an operator hand and a substantially changed cube scene | Fail |
| L5_M09 | 2.42 | Arm pose passes, but the detected Cyan start shifts about 52 pixels | Fail visual reset |

Consequently, the earlier 7/10 first-selection, 6/10 order, and 6/10
end-to-end figures are useful screening summaries but are not publishable
formal rates. L2 is the only strict same-scene pair in this batch. Its
Cyan-to-Pink rollout completed; its Pink-to-Cyan rollout eventually completed
the requested successful placement order but first approached Cyan, so strict
first-object switching still failed.

## Startup hesitation

The L1_M09 Pink-to-Cyan success was not a software freeze. Saved state begins
small motion within 0.17 seconds and exceeds 2 normalized units by 0.40
seconds, but it does not depart more than 5 units from the initial body pose
until 48.53 seconds. This agrees qualitatively with the operator's report of a
long hesitant start: the policy was producing small motions before committing
to the task.

Across the 12 daylight attempts:

- median duration: 69.07 seconds;
- maximum duration: 101.10 seconds;
- recovery-v2 training demonstration median: 21.90 seconds;
- recovery-v2 training demonstration maximum: 31.53 seconds.

The rollout median is therefore about 3.15 times the demonstration median.

## Chunk boundaries and gripper cycling

The recorder saved the action actually sent after the per-motor safety limit.
At the approximate 50-action refresh boundaries:

- median boundary body-step P95 across formal rollouts: 5.46 normalized units;
- median non-boundary body-step P95: 2.53 normalized units;
- boundary/non-boundary ratio: about 2.16.

This supports a chunk-refresh discontinuity. Switching to `chunk20` would
refresh more often and is therefore a responsiveness diagnostic, not an
automatic smoothness fix.

Using gripper state above 10 for at least five frames as a repeatable proxy:

- training demonstrations: median 3 high segments, P95 3, maximum 4;
- daylight rollouts: median 15 high segments, maximum 23.

This proxy can split one physical close into several threshold crossings, but
the magnitude of the difference still shows substantial gripper cycling and
recovery behavior beyond the demonstrations.

## Transport clearance

A side-camera contact sheet sampled the middle of the second gripper-high
interval in all 20 Cyan-to-Pink recovery demonstrations. In many samples the
Pink cube and gripper remain close to the table plane rather than showing a
clear vertical-lift phase. Together with two operator-reported contacts while
Pink traveled to C2, this supports the hypothesis that the demonstrations do
not provide enough transport clearance.

This remains qualitative: the dataset stores joint positions rather than a
calibrated Cartesian end-effector height, and no trusted SO-101 URDF/kinematic
calibration was applied in this audit. Do not report a height in millimeters.

## Checkpoint evidence

The original clean-v1 run remains locally available at 5K, 10K, 15K, 20K,
25K, and 30K. The recovery-v2 run saved 2.5K, 5K, 7.5K, 10K, 12.5K, 15K,
17.5K, and 20K on AutoDL; only recovery-20K has been copied locally so far.
The remote intermediates were not deleted.

An identical-noise offline comparison used the same saved L1/L2/L5
observations for the locally available candidates:

| Checkpoint | Median first-10-step language L2 | Median full-chunk RMSE |
|---|---:|---:|
| old 10K | 0.4939 | 0.4322 |
| old 15K | 0.8798 | 0.5907 |
| old 20K | 0.4647 | 0.1989 |
| old 30K | 1.1602 | 0.2952 |
| recovery 20K | 0.8341 | 6.9910 |

Recovery-20K reacts strongly to the instruction over the full chunk,
especially on L2/L5, but the real-robot screen shows that sensitivity is not
always correct and often becomes a Cyan-first bias. This metric must not be
used alone to select a checkpoint.

## Decision

Do not discard intermediate checkpoints and do not start another training run
yet. Recommended sequence:

1. When AutoDL is next powered on, copy every remaining recovery checkpoint:
   2.5K, 5K, 7.5K, 10K, 12.5K, 15K, and 17.5K as `pretrained_model` only;
   leave `training_state` remote.
2. Run the identical-noise offline language comparison on all recovery
   checkpoints using fixed, valid observations.
3. Select at most two non-20K candidates using both early language sensitivity
   and action continuity, not training loss.
4. Add a pre-rollout paired-start gate that captures state plus both camera
   frames before motion. Reject the trial if body-state L-infinity exceeds 5,
   either cube moved, either cube is not visible in the front view, the center
   area is not empty, or an operator hand is present.
5. Test each selected checkpoint only on one strictly matched L2_M09 opposite-
   instruction pair before expanding hardware trials.
6. If the best checkpoint still collides, collect targeted high-clearance
   paired demonstrations: secure grasp, vertical lift, horizontal transfer
   above both C1/C2, vertical descent, release, and retreat.

No Phase-B unfreezing or additional full training is justified until the
intermediate recovery checkpoints and valid paired starts have been tested.
