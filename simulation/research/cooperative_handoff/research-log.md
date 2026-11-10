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
