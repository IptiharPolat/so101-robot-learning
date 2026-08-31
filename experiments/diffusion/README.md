# Diffusion Policy experiment

## Status

`DIAGNOSTIC_NEGATIVE_RESULT`

This was an earlier SO-101 single-cube grasp experiment. It is included in the
unified project because it informed deployment and action-chunk design, not
because it outperformed ACT.

## Recovered configuration

- Dataset recorded in the local model training config:
  `iptihar/so101_grasp_50_v2`.
- Local training config: 80K steps, `n_action_steps=8`, and
  `num_inference_steps=10`.
- A separate 30K Hub checkpoint was also used in a guarded synchronous
  real-robot test with `n_action_steps=15` and ten denoising steps.
- An asynchronous client/server path was tested locally to decouple policy
  inference from the robot control loop.

The historical launchers contain obsolete `/dev/ttyACM*` and numeric camera
indices. They are deliberately not copied into this public project. Any future
hardware rerun must use `configs/rig.local.yaml` and persistent `/dev/*/by-id`
paths.

## Result

The policy could generate and execute arm motion, but the audited gripper
predictions stayed near the open range and did not reliably acquire the cube.
Chunk smoothing cannot repair a grasp command that never closes sufficiently,
so no success rate or success video is published.

## Action-chunk blending experiment

The LeRobot checkout was extended with an optional
`action_chunk_blend_steps` configuration. At the start of a newly sampled
chunk, the first actions are linearly blended from the last executed action to
the new trajectory. Validation rejects values outside
`[0, n_action_steps]`, and unit tests cover enabled, disabled, and invalid
settings.

This reduces a continuity discontinuity only. It does not change the learned
gripper distribution and did not convert the policy into a successful grasping
system. The curated patch is stored in
`patches/diffusion_action_chunk_blending.patch` and targets LeRobot commit
`0f392484458cb5ebca0310c0c4c47390a31c80ed`.
