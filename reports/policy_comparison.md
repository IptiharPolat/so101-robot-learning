# ACT, Diffusion Policy, and SmolVLA comparison

## Scope

This comparison joins the policies into one engineering narrative without
pretending that they used identical tasks or datasets. ACT and Diffusion were
first studied on single-cube manipulation. ACT was then used for the fixed
two-cube sequence, followed by SmolVLA for language-conditioned ordering.

## Training and inference

| Property | ACT | Diffusion Policy | SmolVLA |
|---|---|---|---|
| Main objective | Supervised action-chunk regression | Noise-prediction/denoising over action trajectories | Flow Matching action expert conditioned on visual-language features |
| Initialization | From scratch | From scratch | `lerobot/smolvla_base` |
| Language input | No | No | Yes |
| Action generation | One forward pass predicts a chunk | Iterative reverse diffusion predicts a chunk | Integrates a learned velocity field from noise to an action chunk |
| Real-time concern | Chunk staleness and temporal aggregation | Denoising latency and chunk transitions | VLM latency, long chunks, and refresh discontinuities |
| Observed strength | Compact and repeatable on demonstrated fixed tasks | Flexible trajectory distribution | Can condition on two fixed language orders and pretrained visual semantics |
| Observed weakness | Cannot switch order from language | Gripper stayed near open in the audited rollout | Cyan-first bias, Pink grasp instability, hesitation, and low-clearance contacts |

## What the loss does not prove

All three objectives average errors over many action coordinates and time
steps. A low training loss can coexist with failure at the few decision points
that determine task success: choosing the first cube, centering the gripper,
closing it, releasing before the second operation, and avoiding contact during
transport. Checkpoint selection therefore used reload checks, offline action
audits, and real-robot rollouts rather than loss alone.

## Evaluation interpretation

### ACT

The earlier single-cube ACT system completed 16 of 20 trials across five fixed
positions. The current two-cube ACT system also produced saved autonomous
successes, but its 32-rollout batch was intentionally not annotated. ACT task
text is dataset metadata and must not be interpreted as language use.

### Diffusion Policy

The model, synchronous rollout, asynchronous server/client path, and optional
chunk blending were exercised. Offline action inspection and real-robot
behavior showed that the gripper output remained in the near-open region, so
more smoothing could not solve the underlying grasp failure. This branch is a
useful negative result: action continuity and semantic task completion are
separate problems.

### SmolVLA

The recovery-v2 run used 122 episodes arranged into 61 strict opposite-order
pairs. The 20K checkpoint produced qualitative complete executions in both
directions. However, some screening pairs had unmatched initial arm poses or
cube visibility, and strict later checks exposed a repeated Cyan-first bias.
The project therefore demonstrates a complete VLA fine-tuning and deployment
pipeline plus successful examples, not a statistically established language
switching rate.

## Engineering lessons

1. Pair opposite instructions at the same physical layout and starting pose.
2. Keep target slots tied to order (`C1`, `C2`), not to color.
3. Audit gripper action distributions before tuning trajectory smoothing.
4. Treat action-chunk boundaries as a deployment variable and record the
   executed, safety-clipped action.
5. Preserve failed trials and target new demonstrations at the failed phase.
6. Separate infrastructure evidence, qualitative demonstrations, and formal
   evaluation metrics.
