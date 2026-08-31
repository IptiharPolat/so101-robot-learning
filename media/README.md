# Demo media provenance

These GIFs are lightweight presentation derivatives. The original local
evaluation datasets and both camera streams remain unchanged under `outputs/`
and are excluded from Git.

| Public asset | Source trial | Edit | Evidence boundary |
|---|---|---|---|
| `act_single_cube_success.gif` | Earlier single-cube ACT evaluation `eval_act_so101_grasp_100_v3_test1`, front camera, 29.23 s | Full rollout, 2x speed, 8 FPS, 360 px wide | Representative single colored-box grasp-and-place success from the earlier ACT project. This is separate from the later two-cube ordering ACT experiment. |
| `smolvla_20k_pink_then_cyan.gif` | `recovery20k_l1_m09_p2c_formal_001`, front camera, 81.50 s | First 44 s of startup hesitation removed, remainder about 2.22x speed, 8 FPS, 360 px wide | Operator-reported unassisted Pink -> Cyan success. The removed hesitation is disclosed and this trial is not used as a formal switch-rate estimate. |
| `smolvla_20k_cyan_then_pink.gif` | `recovery20k_l5_m09_c2p_formal_001`, front camera, 63.10 s | Full rollout, about 3.03x speed, 8 FPS, 360 px wide | Operator-reported Cyan -> Pink success with both cubes visible in the center target area at the end. This is a qualitative capability example. |

All three GIFs use a 96-color generated palette and Bayer dithering. They have
no audio. The transformations change presentation duration only; they do not
reorder frames or splice separate attempts together.

Diffusion Policy does not have a success GIF because the audited real-robot
branch did not reliably close the gripper. Its contribution is documented as a
negative result in `experiments/diffusion/README.md`.
