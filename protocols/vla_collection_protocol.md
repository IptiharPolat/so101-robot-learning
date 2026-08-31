# SmolVLA paired collection protocol

Stage 2 preparation is active. Recording remains blocked until the operator gives
fresh, explicit authorization for the recording session.

L1-L5 are now workspace coverage regions rather than five exact ACT positions:

- L1: left workspace;
- L2: right workspace;
- L3: near-depth band;
- L4: far-depth band;
- L5: cross-workspace and diagonal relationships.

Each region contains ten fixed micro-layouts in
`manifests/vla_workspace_coverage.csv`. For every `micro_layout_id`, record one
Pink → Cyan and one Cyan → Pink demonstration from substantially matched initial
states. Interleave the two orders; never collect all 50 of one order first. C1
is always the first placement point and C2 the second, independent of color.
Use only the two canonical English task strings in
`configs/smolvla_experiment.yaml`.

Reject a pair when the layouts are materially different, camera/features differ, either task label is wrong, or the performed order does not match its instruction.

## Schedule-locked collection

- Follow `manifests/vla_episode_schedule.csv` from planned order 1 through 100.
- A micro-layout must have `physical_status=verified` before its launcher may
  execute. Normalized coordinates alone are not permission to move the robot.
- Record exactly one episode per invocation. The launcher selects one of two raw
  dataset roots and adds `--resume=true` only when that source dataset already
  has earlier episodes.
- Do not skip an earlier schedule row. The launcher checks both manifest status
  and current source-dataset episode count before execution.
- After the first demonstration of a pair, restore both cubes to the same marked
  layout without changing cameras, lighting, target area, or arm base. Then run
  the adjacent opposite-order row.
- Keep upload disabled throughout raw collection. QC and merging happen later
  and never modify either raw source dataset in place.

## Operator controls

- `→`: finish and accept the current episode early.
- `←`: discard the current take and re-record the same episode.
- `Esc`: stop the recording session safely; inspect the saved dataset and
  manifest before resuming.

The default launcher is offline only:

```bash
commands/record_vla_episode.sh --episode-id vla_000
```

It must print `DRY RUN ONLY` and must not open hardware. The eventual execution
form is documented by the readiness report but remains unauthorized until the
operator explicitly starts VLA recording.
