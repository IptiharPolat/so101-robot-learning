# SmolVLA dataset pairing QC

Date: 2026-08-27 (Asia/Shanghai)

## Decision

`CLEANED_LOCAL_DATASET_READY`

All 100 planned schedule rows have been invoked locally. The structural
schedule and full media integrity checks pass. The operator reviewed all pair
content and reported no remaining issues. Cleaning and merge outputs are
local-only and have not been uploaded or used for training.

## Schedule and source mapping

- Manifest rows: 100
- Strict pairs: 50
- Pink → Cyan rows: 50
- Cyan → Pink rows: 50
- Planned rows remaining: 0
- Manifest status: 100 `accepted` rows after operator content review
- Valid source episodes selected for cleaning: 50 Pink → Cyan and 50 Cyan → Pink
- Source-index mapping is unique and present in metadata for every selected row

Two rejected Cyan → Pink source episodes are retained but excluded:

- `vla_002` rejected source episode `1`; corrected replacement source episode `2`
- `vla_004` rejected source episode `3`; corrected replacement source episode `4`

The Cyan raw dataset therefore contains 52 episodes while only 50 are eligible
for the cleaned dataset. The Pink raw dataset contains exactly 50 episodes.
Raw datasets are not modified in place.

## Automated dataset checks

| Check | Pink → Cyan | Cyan → Pink |
|---|---:|---:|
| Episodes in raw root | 50 | 52 |
| Valid source episodes selected | 50 | 50 |
| Total frames | 33,148 | 33,889 |
| State/action shape | 6 / 6 | 6 / 6 |
| NaN/Inf/non-numeric values | 0 | 0 |
| Camera streams | front + side | front + side |
| Resolution/FPS | 640×480 / 30 | 640×480 / 30 |
| Decoded frames == metadata frames | pass | pass |
| Black frames | 0 | 0 |
| Overbright frames | 0 | 0 |
| Exact consecutive duplicate frames | 0 | 0 |
| Sample coverage per camera | 50×5 | 52×5 |

Reports and contact sheets:

- [`vla_media_qc_pink_then_cyan.json`](vla_media_qc_pink_then_cyan.json)
- [`vla_media_qc_cyan_then_pink.json`](vla_media_qc_cyan_then_pink.json)
- [`vla_media_qc_pink_then_cyan/`](vla_media_qc_pink_then_cyan/)
- [`vla_media_qc_cyan_then_pink/`](vla_media_qc_cyan_then_pink/)

## Cleaning and merge result

- Clean Cyan output: `outputs/datasets/vla_cyan_then_pink_clean_v1`
- Clean Cyan episodes: 50 (raw source episodes 1 and 3 excluded)
- Merged output: `outputs/datasets/vla_pink_cyan_order_clean_v1`
- Merged episodes: 100
- Merged frames: 65,691
- Merged task labels: 50 Pink→Cyan and 50 Cyan→Pink by frame count
- Selection mapping: `manifests/vla_clean_selection.csv`
- Clean and merged schema/media checks: pass

## Training gate

The cleaned merged dataset is ready for a separate SmolVLA smoke-test
authorization. Training must use the new local merged root, not either raw
root, and must remain upload-disabled until explicitly authorized.

Prepared commands (not started):

- `commands/train_smolvla_smoke.sh` — 200 steps, batch size 4.
- `commands/train_smolvla_full.sh` — 20,000 steps, checkpoints every 5,000 steps.
