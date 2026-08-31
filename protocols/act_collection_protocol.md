# ACT collection protocol

Task label (exact):

`First pick up the pink cube and place it in the center target area, then pick up the cyan cube and place it in the center target area.`

## Shared procedure

1. Follow the selected manifest in `planned_order`.
2. Reset the named layout using its pre-defined physical markers.
3. Confirm both cubes and the full center target area are visible in `front` and `side`.
4. Demonstrate: approach pink → grasp pink → place/release at C1 → clear the gripper → approach cyan → grasp cyan → place/release at C2 → retreat.
5. Keep the transition between the two pick-and-place segments purposeful and consistent.
6. Re-record immediately for wrong order, collision, dropped cube, camera fault, incomplete release, timeout, or large idle pause.
7. Mark schedule status and annotation row after each saved episode.
8. Keep `dataset.push_to_hub=false` throughout local collection and QC.

Controls during collection:

- `→`: accept/save the completed episode and enter reset.
- `←`: discard the current attempt and rerecord the same schedule row.
- `Esc`: stop collection; upload remains disabled.

## Pilot

- Manifest: `manifests/act_episode_schedule.csv`
- Command draft: `commands/record_act_pilot.sh`
- Scale: L1–L3 × four repeats = 12 episodes.

## Formal ACT v1

- Manifest: `manifests/act_formal_episode_schedule.csv`
- Command draft: `commands/record_act_formal.sh`
- Dataset: `iptihar/so101_pink_cyan_sequence_act_v1`
- Local root: `outputs/datasets/act_formal_v1`
- Scale: L1–L5 × ten repeats = 50 episodes.
- Fixed schedule seed: `20260824`.
- Follow the manifest exactly; do not replace failed takes with unscheduled
  layouts or random extra data.

ACT success must only be described as fixed-order visuomotor execution, never language understanding.
