# SmolVLA recovery-v2 checkpoint offline selection

Date: 2026-08-31

Status: `OFFLINE_SHORTLIST_10K_AND_17P5K`

## Integrity

All eight `pretrained_model` directories are local and contain the policy
weights plus pre/postprocessor configuration. Each local weight SHA-256 matches
the corresponding AutoDL file. No `training_state` was downloaded.

## Method

Every checkpoint was evaluated on the same saved L1, L2, and L5 observations
under the two canonical task strings. Flow-Matching noise was held identical
between instructions. The comparison is offline and did not open robot or
camera hardware.

Language difference measures sensitivity, not correctness. Action continuity
is computed on the raw postprocessed 50-action prediction before the real-robot
`max_relative_target` guard. Large raw steps predict more clipping and a higher
risk of visible chunk-boundary motion.

| Checkpoint | Median first-10 language L2 | Median full-chunk RMSE | Median worst first body gap | Median worst body-step P95 | Median worst gripper-step P95 |
|---|---:|---:|---:|---:|---:|
| 2.5K | 2.6658 | 4.0966 | 3.2178 | 12.0352 | 2.1685 |
| 5K | 0.4908 | 0.2799 | 4.4378 | 5.0570 | 2.0214 |
| 7.5K | 1.2771 | 4.1669 | 3.7039 | 11.8619 | 2.1219 |
| 10K | 2.1670 | 2.6568 | 2.4391 | 8.3737 | 2.3919 |
| 12.5K | 1.1897 | 9.5620 | 2.4230 | 9.6907 | 1.2541 |
| 15K | 1.8603 | 7.7400 | 1.8707 | 9.3545 | 2.2249 |
| 17.5K | 1.6118 | 5.6887 | 1.4072 | 9.1418 | 0.7382 |
| 20K | 0.8341 | 6.9910 | 1.5742 | 9.1919 | 0.7464 |

## Shortlist

1. **10K primary**: strongest early language difference after excluding the
   very rough 2.5K checkpoint; lower body-step P95 than 2.5K, 7.5K, 12.5K,
   15K, 17.5K, and 20K.
2. **17.5K secondary**: lower early language difference than 10K but the
   smallest first-body gap and lowest gripper-step P95 of the language-sensitive
   candidates. It is the complementary smoothness candidate.

Recovery-20K remains the measured baseline. The remaining checkpoints are
retained locally and are not declared useless; they are simply not first in the
hardware queue.

## Next gate

Before another robot rollout, add a paired-start capture gate for body pose,
both front/side frames, cube visibility, cube locations, empty center area, and
absence of operator hands. Then test exactly one strict L2_M09 opposite-order
pair at 10K. Test 17.5K only after the 10K pair is scored. Do not compare models
using unmatched resets.

Raw results:
`outputs/evaluations/smolvla_recovery_all_checkpoint_language_compare/`.
