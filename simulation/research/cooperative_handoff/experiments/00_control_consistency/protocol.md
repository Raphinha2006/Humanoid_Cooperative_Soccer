# 00 — Control consistency and credible receiver baseline

Status: PRE-REGISTERED, before new experimental results.
Date: 2026-11-10. All simulator results must be labelled MuJoCo results, not hardware.

## Question and exclusions

Does the sender approximation in imagined receiver training account for measurable
forecast/action-effect errors or receiver degradation? The sender remains a frozen
native policy. This experiment cannot establish instability caused by learning
partners. Do not change physics, actor architecture, reward weights, optimiser,
gradient clipping, replay capacity, or the strict success criterion in this study.

## Mandatory common repairs

Evaluation uses the previous residual in the same reward function used for
imagination, while retaining a separately labelled legacy return for comparison.
Strict success is unchanged: ordered physical contacts, >=2 m travel, front pocket,
low ball speed, >=0.8 s final hold, facing, and no fall. The receiver residual and
frozen skill interfaces remain unchanged. Controller memory is explicit and never
predicted by the physical WM.

## Correctness gates

1. Reconstruct native sender commands on recorded physical trajectories through
   pre-kick, passing, blend-out, and post-kick phases. Compare commands and memory
   with the actual controller, including filtered commands, yaw origin, held visual
   observations and histories. Max command error <=1e-4 rad, no NaN/Inf.
2. Closed-loop imagined sender must have finite derivatives to physical inputs;
   deterministic memories must update without reading future true states/actions.
3. Replay windows never cross resets/falls, and each mode samples the same valid
   H-step start distribution. Recorded-action mode is explicitly an oracle-input
   diagnostic, not a deployable control method.
4. A full training update at 2048 environments, batch1024, micro128, width2048,
   H32, njmax1024 must pass without capacity overflow. Use one project job on GPU1;
   preserve other users' jobs. If memory is insufficient, reduce microbatch first;
   environment/batch changes must apply to every arm and be logged before training.

## Frozen checkpoint diagnostics

Checkpoints are selected before new evaluation: the common iteration141 start,
the original-learning-rate arm's development-selected best (historical iteration537),
and that arm's latest completed checkpoint. Resolve paths and SHA256 before running.
Evaluate native/start/peak/late policies 5 times on the fixed development initials
(seed220000,512 worlds), preserving actual qpos/qvel and per-world outcomes. This
measures simulator repeat variation; repetitions are not independent training seeds.
Use development seed250000 for a separate generalisation probe, not as final test.
The untouched seed240000/1024 set remains reserved and must not be evaluated here.

Record strict rate, ordered contact, final/maximum hold, handoff, falls, contact
speed where available, corrected/legacy return, capacity and solver limits. Save
physical trajectories and actions for fixed WM error analysis around true contact.
No checkpoint reselection based on these diagnostics.

## Matched training comparison

Arms: held sender target; recorded future sender controls; full closed-loop sender
with explicit native skill memory and phase evolution. All arms use identical
H32-valid replay start sampling; the legacy length1 sampler is retained only for
backward-compatible reproduction, not mixed into the comparison.

Start each run from the same iteration141 actor/WM/optimiser checkpoint. Seeds
104,105,106 define fresh sampling/exploration independently; reset training RNG
after loading the common checkpoint so they are genuine new continuation seeds.
Keep 2048 environments, batch1024, micro128, 10 WM fits/update of length8, H32,
njmax1024, actor/WM/critic LR2e-4, existing Adam betas/clips/reward. Each arm performs
512 new updates and 32 collection steps/update (33,554,432 new training transitions),
plus the separately reported identical 300-step replay warmup (614,400 transitions).
Evaluate every32 new updates on seed220000/512; save archives every30min and latest
at every evaluation. A 6h/run safety limit is a censored run, never a completed
matched-budget comparison. Queue arms serially; fail the queue on invalid physics,
NaN/Inf, memory exhaustion or insufficient matched samples. Preserve partial runs.

An end-to-end smoke update is excluded from scientific comparisons. Full runs
start only after correctness/memory gates and frozen diagnostics are recorded.

## Amendment 1 — controller observation timing (before repaired runs)

The controller conformance gate exposed cached body/camera transforms from the
last pre-integration physics substep being mixed with post-integration qpos/qvel.
The existing imagined receiver uses FK of current qpos. A compact physical-state
WM cannot reproduce this mixed-time observation exactly without a separate cache.
This is an implementation-consistency finding, not a proven cause of policy decay.

Keep legacy mode for historical checkpoint repeatability. Add explicit `current`
controller kinematics: refresh kinematics from qpos before prepare(), without
advancing physics or changing contacts/solver parameters. All three training arms
use this same common repair. Run frozen checkpoint audits separately in legacy
and current modes; never attribute their score differences to sender imagination.
Record the observation mode in every run/configuration. The conformance gate
applies to current mode. No reward-weight or optimiser changes are introduced.

## Analysis and decision

Report final strict rate, last-quarter mean, peak-to-final drop, curve area over
new real transitions, and raw per-world outcomes. Report mean/spread across the
three continuation seeds, plus paired-initial intervals with their limited scope.
Separate selected-best and unfiltered-current curves. Model error is measured on
the same frozen trajectories and at 0.16/0.32/0.64/1s, grouped by true contact phase.
Action-effect ranking probes use paired finite perturbations and common initial
states; contact discontinuities preclude claiming an exact simulator derivative.

If fixes remove degradation, record an implementation explanation and stop the
historical-degradation novelty claim. If closed-loop modelling helps beyond the
recorded-action diagnostic, proceed to partner-update factors. If low-level skills
remain inadequate, prioritise reception skill learning before co-evolution.
