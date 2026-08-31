# ACT 25 Hz temporal diagnostic trial

Date: 2026-08-26 (Asia/Shanghai)
Evaluation ID: `ACT30K-L1-TEMPORAL-001`
Status: **AUTONOMOUS SUCCESS WITH FIRST-PLACE QUALITY ISSUE**
Diagnostic success count: **1/1**
Formal ACT success count: **not applicable**

## Trial identity

- Policy: `iptihar/act_so101_pink_cyan_sequence_v1`
- Revision: `d5d6fa1afa56b928808091306b2edcc7d01c200b`
- Checkpoint: 30,000 steps
- Layout: L1, seen during training
- Control rate: 25 Hz
- Inference overrides: CUDA AMP, `n_action_steps=1`, temporal ensemble `0.01`
- Task: `First pick up the pink cube and place it in the center target area, then pick up the cyan cube and place it in the center target area.`
- Local dataset: `outputs/evaluations/act_30k_l1_temporal_diag_001`
- Upload: disabled
- Frames: 593
- Duration: 23.72 seconds per camera

## Scoring

| Field | Result |
|---|---|
| `first_object_correct` | true — Pink first |
| `first_grasp_success` | true |
| `first_place_success` | true — Pink stabilized inside the center target area |
| `first_release_before_second` | true — Pink detached before contact with Cyan |
| `second_object_correct` | true — Cyan second |
| `second_grasp_success` | true |
| `second_place_success` | true |
| `both_inside_center` | true |
| `order_success` | true |
| `end_to_end_success` | true |
| `collision` | false |
| `timeout` | false |
| operator intervention | none reported or visible |

## Quality issue

The operator reported that the first cube was lifted/carried slightly during
placement. Both videos confirm a delayed first release: after the gripper began
its release phase, Pink remained caught or carried for several seconds before
detaching and settling inside the target area. It was fully released before
the policy contacted Cyan, so this does not invalidate sequential success.

This is logged as `delayed_first_release_object_carry`, closely related to the
project's `object_slip` failure family. It is not marked
`first_release_failure` because release ultimately completed at the correct
stage and the object remained inside the target area. The trial is therefore a
success with degraded placement quality, not a perfect placement.

## Data and media QC

- One episode, 593 Parquet rows, exact task label
- Two H.264 videos, 640×480 at 25 FPS, 593 frames / 23.72 seconds each
- Stable 6D state and action shapes
- Zero NaN or Inf in state/action
- Clear autonomous Pink → Cyan sequence in both views
- Both cubes visible inside the center target area at the end
- Process exited through right arrow, encoded both videos, and disconnected the
  follower and cameras cleanly

## Interpretation

The modified inference mode corrected the gross grasp miss observed in the
original 30 Hz, 100-action-open-loop trial on this L1 repetition. However, one
baseline failure and one modified success are not enough to prove that temporal
ensembling is generally better. The changed 25 Hz control rate also prevents
counting this as a formal apples-to-apples result.

The next gate should repeat the same no-intervention 25 Hz L1 diagnostic three
times. All ordinary failures must be retained. Continue to other layouts only
if first grasp remains reliable and delayed first release is not systematic.
