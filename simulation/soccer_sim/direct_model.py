"""Two-robot/ball adaptation of SuperMimic collect-fit-imagine architecture.

ActorCritic and FK are reused from the pinned Apache-2.0 community implementation.
Task dynamics add a second quaternion, ball, and frozen skill-controller memory.
"""

import torch
from torch import nn

from .supermimic_core.kinematics import DifferentiableKinematics, build_kinematic_tree
from .supermimic_core.math import quat_apply, quat_inv, quat_mul
from .supermimic_core.networks import (
    ActorCritic,
    ActorCriticCfg,
    RunningNorm,
    _mlp,
    _rotvec_to_quat,
)


def sl(layout, name):
    return slice(*layout[name])


def context(s, cfg):
    return s.new_tensor(cfg["physics_context"]).expand(len(s), -1)


def goal(s, layout):
    pos = s[:, sl(layout, "r1_position")]
    q = s[:, sl(layout, "r1_quaternion")]
    rel = quat_apply(quat_inv(q), s[:, sl(layout, "ball_position")] - pos)
    vel = quat_apply(quat_inv(q), s[:, sl(layout, "ball_velocity")][:, :3])
    return torch.cat(
        (rel, vel, s[:, sl(layout, "time")], torch.ones_like(rel[:, :1])), -1
    )


def make_actor(size, hidden=128):
    return ActorCritic(
        ActorCriticCfg(
            obs_dim=size,
            goal_dim=8,
            state_dim=size,
            context_dim=3,
            critic_goal_dim=8,
            action_dim=20,
            history_len=1,
            latent_dim=16,
            encoder_hidden=(hidden, hidden),
            decoder_hidden=(hidden, hidden),
            critic_hidden=(hidden, hidden),
        )
    )


def act(actor, state, layout, cfg):
    raw = actor.act(state, goal(state, layout))
    return raw.tanh() * state.new_tensor(cfg["action_mask"]) * cfg["action_limit"]


class DirectDynamics(nn.Module):
    def __init__(self, size, layout, dt, hidden=256):
        super().__init__()
        self.layout, self.dt = layout, dt
        frozen = tuple(j for i in range(2) for j in range(*layout[f"r{i}_quaternion"]))
        self.norm = RunningNorm(size, frozen=frozen)
        self.net = _mlp(size + 20, (hidden, hidden), size, "silu", output_gain=0.01)
        self.term = _mlp(size, (128, 64), 1)
        nn.init.constant_(self.term[-1].bias, -5)

    def forward(self, s, a):
        delta = self.net(torch.cat((self.norm(s), a), -1))
        std = (self.norm.var + 1e-5).sqrt().clamp_min(0.05)
        out = s + delta * std
        # Avoid redundant body-position prediction; FK derives all link states.
        chunks = []
        for i in range(2):
            start = self.layout[f"r{i}_position"][0]
            quat = quat_mul(
                _rotvec_to_quat(delta[:, start + 3 : start + 6] * 0.1),
                s[:, start + 3 : start + 7],
            )
            quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            stop = self.layout[f"r{i}_qd"][1]
            xyz = (
                s[:, start : start + 3]
                + s[:, start + 7 : start + 10] * self.dt
                + delta[:, start : start + 3] * std[start : start + 3]
            )
            chunks.append(torch.cat((xyz, quat, out[:, start + 7 : stop]), -1))
        bp = self.layout["ball_position"][0]
        bv = self.layout["ball_velocity"][0]
        ball = (
            s[:, bp : bp + 3]
            + s[:, bv : bv + 3] * self.dt
            + delta[:, bp : bp + 3] * std[bp : bp + 3]
        )
        t = s[:, sl(self.layout, "time")] + self.dt
        return torch.cat(
            (*chunks, ball, out[:, bv : bv + 6], t, out[:, self.layout["time"][1] :]),
            -1,
        )

    def termination(self, s):
        return self.term(self.norm(s)).squeeze(-1).sigmoid()

    def loss(self, pred, target):
        err = (pred - target) / (self.norm.var + 1e-5).sqrt().clamp_min(0.05)
        # Quaternion sign is invariant; episode data uses continuous signs.
        return (
            err.square().mean()
            + 3 * err[:, sl(self.layout, "ball_position")].square().mean()
        )


def make_fk(env):
    ids = env.link_ids[1]
    names = [env.m.body(i).name for i in ids]
    joints = [env.m.joint(i).name.rsplit("/", 1)[-1] for i in env.skills[1].j]
    tree = build_kinematic_tree(env.m, names, joints)
    fk = DifferentiableKinematics(tree)
    feet = [
        names.index("A2/" + side + "_ankle_roll_link") for side in ["left", "right"]
    ]
    return fk, feet


def reward(s, a, previous, layout, cfg, fk=None, feet=None):
    g = goal(s, layout)
    rel, vel = g[:, :3], g[:, 3:6]
    pocket = (
        (rel[:, :2] - s.new_tensor([cfg.get("receive_pocket_x", 0.24), 0]))
        .square()
        .sum(-1)
    )
    near = torch.exp(-pocket / 0.08)
    slow = torch.exp(-vel[:, :2].square().sum(-1) / 0.16)
    r = near * (0.3 + 0.7 * slow)
    r = r - 0.015 * a.square().mean(-1) - 0.02 * (a - previous).square().mean(-1)
    for i in range(2):
        pos = s[:, sl(layout, f"r{i}_position")]
        q = s[:, sl(layout, f"r{i}_quaternion")]
        upright = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
        r = (
            r
            - 4 * (cfg.get("reward_min_height", 0.48) - pos[:, 2]).relu()
            - 2 * (1 - upright).square()
        )
    if fk is not None:
        bp, _bq = fk(
            s[:, sl(layout, "r1_position")],
            s[:, sl(layout, "r1_quaternion")],
            s[:, sl(layout, "r1_q")],
        )
        foot = bp[:, feet]
        ball = s[:, sl(layout, "ball_position")]
        # Encourage a foot in reach and prevent reward from ball behind the body.
        distance = (foot[:, :, :2] - ball[:, None, :2]).square().sum(-1).min(-1).values
        r = r + 0.1 * torch.exp(-distance / 0.04) - 0.5 * (-rel[:, 0]).relu()
    return r
