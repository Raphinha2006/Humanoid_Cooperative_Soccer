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
