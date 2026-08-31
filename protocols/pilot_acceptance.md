# ACT pilot acceptance gate

The pilot passes only if all conditions hold:

- 12/12 episodes complete Pink → Cyan.
- Pink is placed and released before any cyan-cube manipulation.
- Cyan is then grasped, placed, and released.
- Both cubes finish inside the center target area at distinct C1/C2 placements.
- Both camera streams are present, complete, readable, correctly named, and visually usable.
- Colors are distinguishable; no freeze, major drop, blur, or overexposure invalidates the sequence.
- Every task label exactly matches the canonical ACT string.
- State/action contain no NaN/Inf and have stable shape across all frames/episodes.
- Two clear gripper close/open cycles occur.
- No excessive pause occurs between tasks; demonstration strategy is consistent.
- Episode duration covers the complete two-cube sequence.
- No obvious collision, target-capacity, or workspace hazard exists.

Any failure stops formal 50-episode collection until corrected and a new 12/12 pilot passes.
