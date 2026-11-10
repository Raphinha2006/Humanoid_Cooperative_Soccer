# Additional gate — normal WM update count

Date: 2026-11-10. Recorded before the matched training study is launched.

The smoke flag retains the requested 2048 environments, batch1024 and H32 actor
rollout, but reduces WM fitting to one fit rather than the normal ten fits. The
completed smoke checks are valid reduced-fit resource checks, not full normal
updates. They are excluded from scientific comparisons.

Before starting any 512-update arm, additionally run held/recorded/closed_loop
from the same start checkpoint for exactly one new update WITHOUT `--smoke`.
Use all normal settings, including ten length8 WM fits, H32, seed104, fresh RNG,
current kinematics, matched starts, and a one-hour safety cap. Validate successful
completion and save logs. These three gates are also excluded from comparisons.

No algorithm, reward, checkpoint selection or training budget changes. The frozen
main supervisor source stays unchanged. `tools/complete_cooperation_preflight.py`
waits for the existing preflight, runs these gates serially, and only then invokes
the original `--skip-preflight` study. Failure stops this extension and its own
process group; unrelated jobs are untouched.
