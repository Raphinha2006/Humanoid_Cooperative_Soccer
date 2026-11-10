"""Compact physical WM and SuperMimic actor/critic. Deterministic skill memory is outside WM."""

import torch
from torch import nn

from ..supermimic_core.kinematics import DifferentiableKinematics, build_kinematic_tree
from ..supermimic_core.math import quat_apply, quat_inv, quat_mul
from ..supermimic_core.networks import (
    ActorCritic,
    ActorCriticCfg,
    RunningNorm,
    _mlp,
    _rotvec_to_quat,
)


def receiver(s):
    return torch.cat([s[:, 75:150], s[:, 150:159]], -1)


def goal(s):
    r = receiver(s)
    inv = quat_inv(r[:, 3:7])
    return torch.cat(
        [
            quat_apply(inv, r[:, 75:78] - r[:, :3]),
            quat_apply(inv, r[:, 78:81]),
            s[:, 159:160],
            torch.ones_like(s[:, :1]),
        ],
        -1,
    )


def make_actor():
    a = ActorCritic(
        ActorCriticCfg(
            obs_dim=84,
            goal_dim=8,
            state_dim=84,
            context_dim=3,
            action_dim=12,
            critic_goal_dim=8,
            history_len=1,
            latent_dim=32,
            encoder_hidden=(1024, 512, 256),
            decoder_hidden=(1024, 512, 256),
            critic_hidden=(256, 128),
        )
    )
    nn.init.zeros_(a.decoder[-1].weight)
    nn.init.zeros_(a.decoder[-1].bias)
    return a


def residual(actor, s):
    return 0.15 * actor.act(receiver(s), goal(s)).tanh()


def context(s):
    return s.new_tensor([1.0, 0.43, 0.11]).expand(len(s), -1)


def value(actor, s):
    return actor.value(receiver(s), goal(s), context(s))


def reward(s, delta, previous):
    g = goal(s)
    rel = g[:, :3]
    speed = s[:, 153:155].norm(dim=-1)
    near = torch.exp(-((rel[:, :2] - s.new_tensor([0.35, 0])).square().sum(-1)) / 0.08)
    slow = torch.exp(-speed.square() / 0.25)
    upright = 1 - 2 * s[:, 79:81].square().sum(-1)
    r = (
        near * (0.2 + 0.8 * slow)
        - 0.15 * delta.square().mean(-1) / 0.0225
        - 0.02 * (delta - previous).square().mean(-1) / 0.0225
    )
    r = (
        r
        - 10 * (0.65 - s[:, 77]).relu()
        - 2 * (1 - upright).square()
        - 0.5 * (-rel[:, 0]).relu()
    )
    return r


class PhysicalWM(nn.Module):
    def __init__(self, width=2048):
        super().__init__()
        self.norm = RunningNorm(160, frozen=tuple(range(3, 7)) + tuple(range(78, 82)))
        self.net = _mlp(222, (width, width, width), 157, output_gain=0.01)
        self.term = _mlp(160, (256, 128), 1)
        nn.init.constant_(self.term[-1].bias, -5.0)

    def forward(self, s, qcmd):
        raw = self.net(torch.cat([self.norm(s), qcmd], -1))
        # 157 outputs = 2*(3 pos+3 rotation+6 vel+31 q+31 qd) + 9 ball.
        parts = []
        cursor = 0
        for off in [0, 75]:
            d = raw[:, cursor : cursor + 74]
            cursor += 74
            quat = quat_mul(_rotvec_to_quat(d[:, 3:6] * 0.05), s[:, off + 3 : off + 7])
            quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            pos = (
                s[:, off : off + 3] + s[:, off + 7 : off + 10] * 0.02 + d[:, :3] * 0.02
            )
            vel = s[:, off + 7 : off + 13] + d[:, 6:12]
            q = (
                s[:, off + 13 : off + 44]
                + s[:, off + 44 : off + 75] * 0.02
                + d[:, 12:43] * 0.05
            )
            qd = s[:, off + 44 : off + 75] + d[:, 43:74]
            parts.append(torch.cat([pos, quat, vel, q, qd], -1))
        ball = torch.cat(
            [
                s[:, 150:153] + s[:, 153:156] * 0.02 + raw[:, 148:151] * 0.02,
                s[:, 153:159] + raw[:, 151:157],
            ],
            -1,
        )
        return torch.cat(parts + [ball, s[:, 159:160] + 0.02], -1)

    def loss(self, p, t):
        pn = self.norm(p)
        tn = self.norm(t)
        mask = torch.ones(160, device=p.device, dtype=torch.bool)
        mask[3:7] = False
        mask[78:82] = False
        mask[159] = False
        l = (pn[:, mask] - tn[:, mask]).square().mean(-1)
        for off in [0, 75]:
            l = l + 4 * (
                1 - (p[:, off + 3 : off + 7] * t[:, off + 3 : off + 7]).sum(-1).square()
            )
        l = l + 3 * (pn[:, 150:156] - tn[:, 150:156]).square().mean(-1)
        # Extra physical weighting near a collision, independent of learned reward.
        contact = (t[:, 150:153] - t[:, 75:78]).norm(dim=-1) < 0.95
        return (l * (1 + contact.float())).mean()

    def termination(self, s):
        return self.term(self.norm(s)).squeeze(-1).sigmoid()


class ImagineController(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.native = env.mods["receiving"]
        ids = []
        b = env.m.body("A2/neck_camera_frame").id
        while b > 0:
            ids.append(b)
            b = int(env.m.body_parentid[b])
        ids = sorted(ids)
        names = [env.m.body(i).name for i in ids]
        joints = [env.m.joint(i).name.rsplit("/", 1)[-1] for i in env.cpu.skills[1].j]
        self.fk = DifferentiableKinematics(
            build_kinematic_tree(env.m, names, joints)
        ).to(env.device)
        self.cam = names.index("A2/neck_camera_frame")

    def forward(self, s, h, previous, held, delta):
        rs = receiver(s)
        pos, quat = self.fk(rs[:, :3], rs[:, 3:7], rs[:, 13:44])
        q, h, raw, held = self.native.observation(
            rs,
            pos[:, self.cam],
            quat[:, self.cam],
            h,
            previous,
            s.new_zeros(len(s), 3),
            s.new_zeros(len(s)),
            held,
        )
        # Native torque-speed saturation, matching the real backend.
        q = torch.cat([q[:, :12] + delta, q[:, 12:]], -1)
        return q, h, raw, held


def clip_command(s, target, env):
    q = torch.cat([s[:, 13:44], s[:, 88:119]], -1)
    v = torch.cat([s[:, 44:75], s[:, 119:150]], -1)
    maxv = env.max_vel.repeat(2)
    lim = env.max_torque.repeat(2)
    vel = v.clamp(-2 * maxv, 2 * maxv)
    hi = torch.minimum(lim * (1 - vel / maxv), lim)
    lo = torch.maximum(lim * (-1 - vel / maxv), -lim)
    result = target.clamp(
        q + (lo + env.kd2 * v) / env.kp2, q + (hi + env.kd2 * v) / env.kp2
    )
    return torch.where(
        env.posctrl.repeat(2),
        target.clamp(env.lower.repeat(2), env.upper.repeat(2)),
        result,
    )
