# Current understanding

- Historical receiver training improved imagined objectives without stable strict
  reception improvement. It also contained known controller/reward accounting
  inconsistencies. Causes remain unproven.
- The historical sender policy was frozen. Partner-learning nonstationarity is a
  future controlled factor, not an explanation already established by those runs.
- Current physical state prediction and strict ordered-contact/hold evaluation
  measure different things. Lower state RMSE does not establish useful action
  sensitivities.
- WM/policy co-improvement, multi-agent world models, action-effect supervision,
  and skill feasibility chaining all have close prior work. These are baselines
  and design ingredients, not novelty claims on their own.

# Candidate research question

Can a model of physical handoff conditions, conditioned on a partner's changing
capability, improve sustained cooperative skill composition while retaining
previously learned collaboration and generalising to unseen partner checkpoints?

# Lessons and constraints

- Repair and isolate implementation inconsistencies before causal claims.
- Do not use held-out evaluation for checkpoint selection.
- Report raw candidate performance as well as accepted checkpoint performance.
- Keep state/action interfaces, real transition budgets, and update counts matched.
- A mechanism that only rejects updates without expanding capability is insufficient.
- Prototype G1 weights cannot be transplanted into K1 joint outputs.

# Open questions

- Does receiver degradation persist after execution/imagination consistency fixes?
- How much error comes from the sender approximation near ball impacts?
- Does action-effect ranking predict policy improvement better than position RMSE?
- Is an explicit handoff representation useful beyond an equal-capacity critic?

# 2026-11-10 fixed-state audit

- Five repeats on identical 512-world development initials reproduce the historical
  peak-to-late drop: legacy 22.03% to 2.54%; current 13.05% to 2.42%. Repeat standard
  deviations and per-world data are published; these are not training replications.
- Current observation timing changes old policy performance. Treat it as a common
  repair and report each mode separately, not as the sender-control treatment.
- On fixed native trajectories, recorded sender controls do not eliminate contact
  forecast error. A late model can have lower position RMSE but worse reception.
  Sender approximation is not established as the dominant cause; action-effect
  interventions remain necessary.
- All three normal ten-fit/H32 resource gates passed with 2048 worlds and batch1024,
  no capacity overflow. The first matched continuation arm is running on GPU1.

# 2026-11-11 first matched arm

The repaired held-sender continuation seed104 completes its budget but does not
establish sustained improvement: start5.66%, selected peak8.59%, final4.69%;
last four scheduled development evaluations average5.27%. This is one continuation
replicate, not an independent final test. Other sender treatments have not yet
completed, so no treatment ranking or sender-causality claim is supported.
