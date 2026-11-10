# Cooperative humanoid handoff research

The current objective is a falsifiable study of world-model-based cooperation,
followed by continuous face-to-face passing on Booster K1 hardware. The initial
mechanism prototype uses existing G1 resources. No new method improvement has yet
been established.

- `research-state.yaml`: current stage, execution location and remaining work.
- `findings.md`: evidence, limitations and scientific questions.
- `research-log.md`: decisions and deviations.
- `experiments/00_control_consistency/protocol.md`: pre-registered first experiment.
- `literature/survey.md`: nearest related work and novelty boundaries.

Protocol commits precede new experiments. Historical runs remain immutable.
Independent seed240000/1024 is reserved; development diagnostics never use it.

The finite supervisor `tools/run_cooperation_pilot.py` executes correctness tests,
legacy/current observation audits, frozen WM diagnostics, full-scale smoke updates,
and then a serial 3-control-mode × 3-continuation-seed study. It writes a durable
heartbeat, commands, source hashes and logs, and halts the queue on failed checks.
There is one new project GPU child at a time; unrelated jobs are never managed.

## Controller observation timing

`controller_kinematics: legacy` preserves cached-transform behaviour for history
reproduction. `current` derives camera observations and sender geometry from the
same current physical state and FK convention used by imagination. This is a
common consistency repair and a separately reported evaluation factor.

Sender modes are `held`, `recorded` and `closed_loop`. Recorded future sender
controls are an oracle-input diagnostic. Closed-loop mode carries native histories,
filtered commands, yaw references and held visual observations explicitly. These
memories are not learned dynamics outputs.

The reduced-fit smoke is followed by `tools/complete_cooperation_preflight.py`:
three normal ten-WM-fit/H32 gates, then the original matched study. See the normal
update gate supplement. All three normal gates passed before the first full arm.
Public results and readable status: https://github.com/Raphinha2006/Humanoid_Cooperative_Soccer
