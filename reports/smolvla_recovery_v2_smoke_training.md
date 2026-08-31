# SmolVLA recovery-v2 cloud smoke training

Date: 2026-08-30

Status: `SMOKE_PASSED_FULL_TRAINING_NOT_AUTHORIZED`

This was a 200-step infrastructure and learning-signal smoke test. It is not a
formal training result and is not evidence of language-conditioned robot
success.

## Data transfer and scope

- Explicitly authorized destination: the configured private AutoDL SSH host.
- Local/remote dataset: `vla_pink_cyan_order_recovery_v2`.
- Remote dataset size: approximately 541 MB.
- Remote preflight: 122 episodes, 81,726 frames, 61/61 canonical task labels,
  two cameras, finite 6D state/action.
- Hugging Face dataset/model upload: disabled.
- W&B: disabled for smoke.
- Formal training: not started.

## Environment

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB.
- Driver: 580.76.05.
- PyTorch: 2.7.1+cu126, CUDA available.
- Transformers: 4.57.6.
- LeRobot: the pinned checkout under the remote AutoDL workspace.
- Video backend: TorchCodec.

## Training configuration

- Base policy: `lerobot/smolvla_base`.
- Steps: 200.
- Batch size: 4.
- Seed: 20260829.
- Image transforms: disabled.
- Camera mapping: `front -> camera1`, `side -> camera2`.
- Train expert only: true.
- Freeze vision encoder: true.
- Optimizer LR: 1e-4.
- Scheduler requested warmup/decay: 1,000/30,000; smoke auto-scaled to
  6/200 steps by LeRobot.
- Total/trainable parameters: 450,046,176 / 99,880,992.
- Save frequency: 200; evaluation disabled.

## Result

- Training reached step 200 and exited with status 0.
- No CUDA OOM, NaN/Inf, decoder error, or data-loader failure.
- Update time stabilized around 0.16-0.18 seconds/step after initialization.
- Data time stabilized around 0.004-0.005 seconds/step.
- Loss examples: step 10 = 0.157, step 50 = 0.103, step 100 = 0.088,
  step 150 = 0.079, step 200 = 0.071.
- Gradient norm examples: step 10 = 2.843, step 100 = 1.596,
  step 200 = 1.037.
- The short-run curve is finite and generally decreases, but should not be
  interpreted as task success or checkpoint selection evidence.

## Checkpoint validation

- Remote run: the configured cloud output directory for
  `smolvla_recovery_v2_smoke_b4`.
- Run size: approximately 1.3 GB.
- Model file: approximately 865 MB.
- SHA-256:
  `455cbbbac53100b80c122a7129ce0b34572b1d240c805b8f9bffb85eb47b0ac9`.
- `SmolVLAPolicy`, `DataProcessorPipeline` preprocessor, and
  `DataProcessorPipeline` postprocessor reloaded successfully from
  `checkpoints/last/pretrained_model`.
- Reloaded parameter counts match training: 450,046,176 total and 99,880,992
  trainable.
- No GPU process remained after completion.

## Gate

The recovery-v2 cloud pipeline is ready for separately authorized formal
training. The planned Phase-A run is 20K steps, batch size 8, checkpoint every
2.5K steps, W&B scalar logging enabled, W&B artifacts disabled, and Hugging
Face upload disabled. Do not start it without explicit authorization.
