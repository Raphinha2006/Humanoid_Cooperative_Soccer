# Admission refinement — 2026-11-11

The uniform 10,000 MiB threshold was conservative scheduling headroom, not a
measured minimum or a paper setting. Normal gates logged Torch reserved memory
of 2.07 GiB for held/recorded and 3.55 GiB for closed_loop. These exclude Warp
and CUDA allocations and are not total process peaks. Shared free-memory deltas
are also affected by unrelated jobs and cannot establish an exact minimum.

Use mode-specific conservative admission: 7,168 MiB (7 GiB) for held/recorded,
8,704 MiB (8.5 GiB) for closed_loop. These are provisional admission margins,
not claims of minimum requirements. Keep all scientific settings unchanged.
Record the actual child process GPU memory through startup, warmup, normal WM
fits, H32 actor updates and evaluation; distinguish sampled peaks from exact
allocator peaks. A training failure still halts the queue for inspection.

Stop only the waiting recovery runner, preserve its status and original logs,
and launch a new labelled runner with the explicit thresholds. The completed
held arm is skipped. Do not change parallel environments, batch or microbatch.
