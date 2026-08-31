# ACT formal dataset QC

Date: 2026-08-25 (Asia/Shanghai)
Raw dataset root: `outputs/datasets/act_formal_v1`
Correction root: `outputs/datasets/act_l1_correction_v1`
Accepted balanced root: `outputs/datasets/act_formal_balanced_v1`
Configured Hub ID: `iptihar/so101_pink_cyan_sequence_act_v1`
Upload: **completed as a private dataset on 2026-08-25 14:45 CST**
Hub commit: `cd6183684f54f942d1fa29f0b7f6390983ec2250`
Overall status: **PASS — UPLOADED AND VERIFIED; TRAINING NOT STARTED**

Remote verification details are recorded in `reports/act_formal_upload.md`.

## Final balanced result

The accepted dataset contains 47 selected raw episodes plus three separately
recorded L1 correction episodes. It contains exactly 50 episodes and exactly ten
episodes for each of L1–L5. Both source datasets remain unchanged.

| Check | Result |
|---|---|
| Episodes / data frames | 50 / 33,205 |
| Layout balance | L1=10, L2=10, L3=10, L4=10, L5=10 |
| Layout image validation | 50/50 match; zero mismatches |
| Task labels | One exact canonical Pink → Cyan task string |
| State/action | Stable `[6]`; zero NaN/Inf |
| Final video codec | H.264, yuv420p, 640×480 |
| Full video decode | 33,430/33,430 frames per camera |
| Video faults | 0 black, 0 overbright, 0 exact consecutive duplicate frames |
| Five-phase sampling | 50/50 samples at 5%, 25%, 50%, 75%, and 95% for both cameras |
| Final placement | 50/50 sampled endpoints show both cubes in the center target area |

The front stream is split into two valid MP4 files because it exceeds LeRobot's
200 MB per-video limit; the second file contains the three correction episodes.
The side stream remains one MP4. This is valid LeRobot v3 layout.

## Raw 50 outcome

The raw 50-episode recording completed cleanly and passes file, schema, label,
numeric, video-decode, endpoint, and representative fixed-order checks. It does
not yet pass the formal collection Gate because nine episode start layouts differ
from the fixed randomized manifest. The observed layout counts are L1=7, L2=11,
L3=10, L4=10, and L5=12 rather than ten per layout.

The raw dataset remains unchanged and will not be uploaded as the balanced
formal v1. The correction was completed by excluding raw episodes 1, 6, and 9,
recording three new L1 episodes in a separate correction set, and creating the
accepted derived dataset.

## Checks that passed

| Check | Result |
|---|---|
| Recording process | Exit code 0; both cameras, follower, and leader disconnected cleanly |
| Episodes / data frames | 50 / 33,725 |
| Task labels | One exact canonical Pink → Cyan task string |
| State/action | Stable `[6]`; zero NaN/Inf |
| Camera schema | `front` and `side`, 640×480, H.264, 30 FPS |
| Full video decode | 33,792/33,792 frames per camera; 1126.4 s |
| Video faults | 0 black, 0 overbright, 0 exact consecutive duplicate frames |
| Episode duration | 17.03–33.40 s; median 21.87 s |
| Gripper trace | Every episode has at least two large open/close excursions |
| Final placement | 50/50 sampled endpoints show both cubes in the center target area |
| Temporal order evidence | 25% sheets show Pink-first activity; 50–75% sheets show Cyan-second activity; operator accepted each saved take |

The temporal-sheet check is representative evidence plus operator acceptance; it
is not a claim that an automatic vision model semantically scored every video
frame.

## Layout mismatches

Layout identity was checked from the front-camera 5% frame using independently
detected Pink and Cyan centroids. The positions form five tight, clearly
separated layout clusters.

| Episode | Planned | Observed |
|---:|---|---|
| 1 | L1 | L2 |
| 5 | L5 | L4 |
| 6 | L4 | L5 |
| 8 | L1 | L2 |
| 9 | L2 | L5 |
| 28 | L1 | L2 |
| 29 | L2 | L1 |
| 30 | L1 | L4 |
| 31 | L4 | L5 |

Zero-based episode indices are used in this table. Machine-readable evidence is
stored at `outputs/qc/act_formal_v1/layout_qc.json` and
`outputs/qc/act_formal_v1/media_qc.json`; phase contact sheets are under
`outputs/qc/act_formal_v1/contact_sheets/`.

## Completed correction

1. Recorded exactly three successful L1 episodes into a new local correction root.
2. Kept the raw 50-episode dataset immutable.
3. Built the derived dataset from 47 retained raw episodes plus three corrections.
4. Re-ran schema, label, video, action, endpoint, and layout-count QC successfully.
5. Uploaded only the accepted balanced dataset after separate explicit authorization and verified it by remote readback.

Excluded excess episodes are episode 1 (observed L2), episode 6
(observed L5), and episode 9 (observed L5). All three are manifest mismatches;
excluding them produces retained counts L1=7 and L2–L5=10 each before adding the
three L1 corrections.

During derivation, LeRobot 0.4.4's default `delete_episodes` behavior was found
to use AV1 and reconstruct filtered video intervals from control-row count/FPS.
Because camera streams contained slightly more frames, that approximation caused
up to about 2.2 seconds of cumulative image/action offset. The invalid derived
attempt was removed and rebuilt in H.264 with intervals accumulated from the
exact source video durations. The rebuilt dataset then passed 50/50 layout-frame
validation and complete video decoding.

Provenance is recorded in `reports/act_formal_balanced_provenance.json`. The
balanced manifest SHA-256 is
`63859f5eabb0ba11eda4a4c208c33fb121a65c37d2ca0c2076d003f55d05a3f1`.

This is an ACT fixed-order dataset check. It does not demonstrate language
understanding.
