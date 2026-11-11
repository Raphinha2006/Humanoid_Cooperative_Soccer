# Research decision log

## 2026-11-10 — research reframed and execution authorised

The user approved the proposed staged research project. The first milestone is a
credible simulator baseline and a falsifiable mechanism study, not another
uncontrolled long training run. The existing sender is a frozen skill; historical
receiver degradation is not evidence of instability caused by two learning agents.

Use existing G1 resources for the first mechanism study; eventual hardware is
multiple Booster K1 robots with onboard perception and small intent messages.
Keep original runs immutable, keep independent seed 240000 unused, and pre-register
comparisons before running them. Run only one new project GPU job at a time.

## 2026-11-10 — common controller consistency repair

Preserved legacy code and checkpoints before implementation. Replay now carries
explicit sender skill histories and command/visual memory. Three sender treatments
share H32-valid replay starts. Evaluation accounts for the previous residual action;
strict success is unchanged.

The first conformance check exposed a timing mismatch: cached MuJoCo body transforms
and post-integration qpos were read as if they described the same instant. The
repaired controller uses the same current-state Torch FK convention as imagination.
Legacy/current evaluation remains separate, because this changes policy inputs.

Passed eight CPU checks and a 299-step, 16-world CUDA controller comparison:
sender target maximum error 1.1921e-7 rad, receiver target error 0, sender memory
error 0, finite state gradients. This validates interface consistency, not learned
world-model accuracy or improved reception.

The finite preflight queue is running on physical GPU1 in an isolated source
snapshot. Fixed-state checkpoint audits precede frozen forecasts, full-size smoke
updates and the matched continuation study. The reserved final test remains unused.

## 2026-11-10 — public research repository authorised

Publish a curated simulation/research repository with readable Chinese entry points,
parameters, provenance, protocols, progress and tasks. Keep the existing private
robotics repository private. Exclude credentials, server details, raw checkpoints,
downloaded policies/assets and unrelated robot work. Update the public status after
meaningful measured changes; running experiments are not reported as completed.

## 2026-11-10 — diagnostics complete and matched study launched

Both observation-mode audits and 144 frozen forecast summaries completed. Results,
per-world outcomes, checkpoint/trajectory hashes and reproducible PNG/PDF figures
were published in the public research repository. Legacy peak/late means were
22.03/2.54%; current means were 13.05/2.42%. These are repeated fixed-initial-state
audits, not independent training repeats or an independent final test.

The original smoke flag uses one WM fit. Added a committed supplement requiring
three normal ten-fit/H32 updates without that flag, preserving the frozen main
supervisor. All three passed with no capacity overflow. The extension then launched
seed104_held, the first matched arm, on GPU1. No new method improvement is claimed.

## 2026-11-11 — one arm complete, resource admission revised

seed104_held completed 512 additional updates and 33,554,432 transitions in about
1h53m. Development strict reception was 5.66% at start, peaked at8.59%, and ended
at4.69% (another final evaluation4.88%). No sustained improvement is established.
The remaining methods/seeds are untested in full runs.

The original queue exited at00:51:56 Asia/Shanghai after30min of waiting for
10,000MiB free onGPU1. No training OOM occurred. A new recovery runner verifies
completed budgets and frozen hashes, then skips the completed arm. Following the
user's resource-threshold question, use provisional admission7GiB held/recorded
and8.5GiB closed_loop, with complete child-process memory samples. Environment,
batch, microbatch, algorithm and scientific budgets stay fixed. Wait up to12h.

## 2026-11-11 09:21 Asia/Shanghai — first seed comparison complete

All seed104 arms completed 512 added updates/33,554,432 transitions. Final strict
development rates: held4.69%, recorded2.54%, closed_loop0.39%; no stable gain over
the common start. Only one continuation seed is complete. The queue has started
seed105_recorded with no new OOM/capacity failures. Sampled process peaks were
4706MiB recorded and7062MiB closed_loop; settings/budgets remain unchanged.
