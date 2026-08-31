# Layout and center-target design

The semantic center target area is the gray interior of the taped rectangle. A
user-approved deviation retains the existing yellow tape as a fixed boundary.
The yellow boundary must not move or change between training and evaluation,
and must not be used to encode order or color. The gray interior must fit both
cubes at once and remain completely visible in both camera streams.

- C1 receives the first selected cube.
- C2 receives the second selected cube.
- C1 is the yellow internal marker farther from the robot base.
- C2 is the yellow internal marker nearer the robot base.
- C1 and C2 encode order only and are never bound to pink or cyan.
- Pink → Cyan means pink at C1 and cyan at C2.
- Cyan → Pink means cyan at C1 and pink at C2.

Before collection, physically mark and complete every row in
`manifests/layouts.csv`. Across L1–L5, swap pink/cyan left-right and
front-back relationships. Keep both cubes separated, unoccluded, and inside the
conservative workspace. U1–U3 are already named and assigned to the unseen
evaluation split; their coordinates must be fixed before formal training.

The operator confirmed this C1/C2 mapping on 2026-08-24. Do not accept a layout
until a slow manual reach assessment confirms access to
both cubes, C1, C2, and a safe retreat path, and both cameras show the full
sequence without the first placed cube obstructing the second.
