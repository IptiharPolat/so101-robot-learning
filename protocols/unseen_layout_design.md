# U1–U3 unseen-layout definition and freeze

Definition started: 2026-08-26 (Asia/Shanghai)
Final freeze: pending physical markers and reach checks

These layouts are frozen before ACT v2 targeted collection and retraining. They
must never be used for demonstrations, correction data, checkpoint selection,
or deployment-parameter tuning. The original 30K ACT model was already trained
before their physical coordinates were fixed, so any evaluation of that model
must describe them as post-training-frozen unseen layouts rather than fully
preregistered layouts.

## Coordinate convention

Directions are defined only in the fixed front-camera image:

- up: wall side
- down: operator side
- left: robot side
- right: away from the robot
- `s`: one cube edge length

Positions refer to cube centers. Use the outside edge of the yellow target
boundary as the reference. Placement marks must be tiny neutral pencil dots or
another camera-inconspicuous neutral method and must be completely covered by
the cubes at the start of a trial.

## Layouts under acceptance

### U1 — operator-selected right-side vertical layout

- mark `UA`: operator-selected position above and right of the target in the
  fixed front-camera image
- mark `UB`: operator-selected position right and slightly below the target's
  vertical midpoint in the fixed front-camera image
- Pink at `UA`
- Cyan at `UB`
- front-camera evidence: `outputs/camera_check/layouts/U1/front.jpg`
- side-camera home-view evidence: `outputs/camera_check/layouts/U1/side.jpg`
- visual status: front view passed; wrist/side reach visibility pending

### U2 — U1 color swap

- use the exact same `UA` and `UB` marks
- Cyan at `UA`
- Pink at `UB`
- front-camera evidence: `outputs/camera_check/layouts/U2/front.jpg`
- side-camera home-view evidence: `outputs/camera_check/layouts/U2/side.jpg`
- visual status: strict color swap confirmed; wrist/side reach visibility pending

### U3 — operator-selected wide opposite diagonal

- mark `UC`: operator-selected position left and below the target in the fixed
  front-camera image
- mark `UD`: operator-selected position above and right of the target
- Pink at `UC`
- Cyan at `UD`
- front-camera evidence: `outputs/camera_check/layouts/U3/front.jpg`
- side-camera home-view evidence: `outputs/camera_check/layouts/U3/side.jpg`
- visual status: wide diagonal and front visibility passed; robot-side Pink
  clearance plus wrist/side reach visibility pending

## Acceptance gate

Each layout requires all of the following before `manifests/layouts.csv` can be
changed from `predeclared` to `accepted_for_formal`:

- both cubes fully visible and separated in the fixed front camera
- center target area and both C1/C2 references visible in both home views
- each cube becomes visible in the wrist/side camera during the slow reach check
- no cube overlaps the yellow target boundary at the start
- conservative collision clearance from the base, wall, target, and other cube
- slow no-policy reach to Pink, C1, Cyan, C2, and safe retreat
- one saved front screenshot and one saved side screenshot

If a frozen coordinate is unsafe or occluded, reject it and document the
failure. Do not silently move it and retain the same layout ID.
