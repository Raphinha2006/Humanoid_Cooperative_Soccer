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

## 2026-11-11 — reception metric audit

Following the user’s concern about a sparse, difficult acceptance metric, summarized the existing seed104 development logs without new simulation. All three arms also lost mean maximum continuous capture time. Strict acceptance alone hides intermediate abilities, but the evidence does not support dismissing all degradation as sampling noise. Published a supplementary analysis and proposed controlled incoming-ball strata, contact-aligned continuous metrics and capability maps for a future pre-registered protocol. Current nine-arm budgets, selection and heldout isolation remain unchanged.

## 2026-11-11 — evaluation/reward plan revised and GPU queue recovered

The user approved revising the plan and requested reward/task validity checks,
then authorised additional GPU1 work. Added a next-phase protocol, configurable
engineering design values, four-layer metrics, separate stopping latency and
control persistence, failures/censoring, fixed post-contact windows, capability
grids, counterexample ranking, bounded event rewards and short-budget gates.
No new reward/evaluator is yet active. Three primary reward-design papers were
read and referenced; mathematical shaping invariance is not task correctness.

The fourth original arm failed at 2026-11-11T09:36:58.853970+08:00 after
iteration230; latest checkpoint228 lacks replay. Preserved the entire interrupted
attempt and logs with a receipt, then restarted that arm from common iteration141
under the committed clean-retry record. Remaining six original-budget arms are
serially queued; original source/config hashes are unchanged. Snapshot 2026-11-11 10:16
CST confirms running seed105_recorded, last logged iteration155.
CPU-only constructed reward probes completed without using GPU1; they are algebraic
diagnostics and do not replace physical rollout validation. Independent test unused.

## 2026-11-11 12:38 Asia/Shanghai — fourth matched arm complete

seed105_recorded clean retry completed512 updates/33,554,432 new transitions in
2.28h. Strict reception5.27%→0.39%, ordered contact58.59%→58.59%, ever-captured
12.70%→1.76%, mean maximum continuous control0.281→0.044s, return39.66→38.94.
The persistence metric is not stopping latency and does not identify the cause.
Two recorded continuations now show late degradation; no overall method ranking
is established. Original-reward seed105_closed_loop is running as arm5/9, without
a post-training evaluation yet at this snapshot. V2 evaluator/reward not active.
Published raw results and raw two-seed curves, with no heldout use or source changes.

## 2026-11-11 17:23 Asia/Shanghai — fifth arm interrupted; waiting recovery

Inspection found seed105_closed_loop failed at14:50:38 on a427,032,576-byte Warp
allocation during collision detection. Last logged iteration440, latest checkpoint441,
no final; only4/9 arms complete. Last development eval429 gives strict0.59%,
legacy mean max persistence0.165s, versus start4.88%/0.289s. No stable improvement
established, no terminal ranking from a censored run. All partial outputs were
archived under a committed recovery record; checkpoint lacks replay. A new
supervisor waits for8.5GiB free onGPU1 before a clean common-start retry, then
the remaining four arms. No training child at the snapshot; configuration and
source fingerprints unchanged. New evaluator/reward not active; final heldout unused.
