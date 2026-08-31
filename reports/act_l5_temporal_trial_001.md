# ACT 25 Hz temporal L5 trial 001

Date: 2026-08-26 (Asia/Shanghai)
Evaluation ID: `ACT30K-L5-TEMPORAL-001`
Status: **AUTONOMOUS SUCCESS WITH GRASP-OFFSET QUALITY ISSUE**

## Trial identity and data QC

- Fixed 30K ACT weights, seen layout L5
- 25 Hz, CUDA AMP, `n_action_steps=1`, temporal ensemble `0.01`
- Local dataset: `outputs/evaluations/act_30k_l5_temporal_001`
- Upload disabled
- 1 episode, 2,257 frames, 90.28 seconds per camera
- Two H.264 videos, 640×480 at 25 FPS
- Exact canonical Pink → Cyan task label
- Stable 6D state/action, zero NaN/Inf
- Normal right-arrow completion and clean follower/camera disconnect

## Physical result

The operator reported a basically successful Pink-then-Cyan sequence. Pink was
grasped with the grasp point slightly offset toward the robot's left side, but
the cube remained acquired and was placed successfully. Cyan was then grasped
and placed successfully. Terminal frames from both cameras confirm both cubes
inside the center target area.

## Scoring

| Field | Result |
|---|---|
| `first_object_correct` | true — Pink first |
| `first_grasp_success` | true |
| `first_place_success` | true |
| `first_release_before_second` | true |
| `second_object_correct` | true — Cyan second |
| `second_grasp_success` | true |
| `second_place_success` | true |
| `both_inside_center` | true |
| `order_success` | true |
| `end_to_end_success` | true |
| `collision` | false |
| `timeout` | false |

Quality classification: `pink_grasp_left_offset`.

The offset is retained as a grasp-precision warning, but it did not invalidate
the autonomous task completion.
