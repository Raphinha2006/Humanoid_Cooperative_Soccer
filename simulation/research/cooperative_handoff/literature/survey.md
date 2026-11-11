# Closest prior work and contribution boundaries

| Work | Source | Relevance and boundary |
|---|---|---|
| SuperMimic — Liao et al., 2026 | https://arxiv.org/abs/2610.28480 | Differentiable WM/policy training; paper excludes object/terrain interaction. The present adapter is not an official full reproduction. |
| VLAW — Guo et al., 2026 | https://arxiv.org/abs/2602.12063 | Real-rollout data and synthetic data drive iterative WM/policy co-improvement. Iteration itself is not new. |
| WorldSync / WorldEcho — Chen et al., 2026 | https://arxiv.org/abs/2608.24885 | Off-expert action following and intervention-effect supervision. Action-sensitivity diagnostics already have direct precedent. |
| SeqWM — Zhao et al., ICLR 2026 | https://arxiv.org/abs/2509.13095 | Sequential agent-wise world models, communicated predictions, cooperative planning, quadruped hardware. Required close multi-agent baseline. |
| Sequential Dexterity — Chen et al., 2023 | https://arxiv.org/abs/2309.00987 | Transition feasibility and backward skill fine-tuning; feasibility alone is not new. |
| SCaR — Chen et al., NeurIPS 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/ca92ff06d973ece92cecc561757d500e-Abstract-Conference.html | Dual regularisation for inter/intra-skill dependencies; skill-boundary matching alone is not new. |
| Joga — Narula, Nguyen, Fey et al., project | https://joga.csail.mit.edu/ | G1 football skills, active vision, perception/dynamics alignment. Public deployment resources are not a complete training release. |
| SoccerBot — Shao et al., 2026 | https://arxiv.org/abs/2611.06564 | Unified football skills and transition state initialisation; single-robot experiments, external ball tracking. |
| From Motor Control to Team Play — Liu et al., Science Robotics 2022 | https://arxiv.org/abs/2105.12196 | Physically simulated humanoid teamwork already exists. Football teamwork alone is not the scientific contribution. |

Sources were inspected during project design. A candidate gap is a research
hypothesis, not a claim that all other literature has excluded this direction.

## Reward design checks added 2026-11-11

- Ng, Harada and Russell (ICML1999), [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). Potential differences preserve optimal policies under the relevant MDP assumptions. This does not validate the base task objective, approximate WM, or optimisation.
- Pan, Bhatia and Steinhardt (ICLR2022), [The Effects of Reward Misspecification](https://arxiv.org/abs/2201.03544). Stronger/longer optimisation can increase proxy reward while reducing true outcomes. Motivation for independent event metrics; not a causal diagnosis of our run.
- Muslimani et al. (2025), [Towards Improving Reward Design in RL](https://arxiv.org/abs/2503.05996). Trajectory-distribution rankings help assess reward alignment. Motivation for a fixed trajectory preference audit; no independent human study has been conducted in this project.
