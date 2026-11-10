"""Exact frozen sender controller with explicit, differentiable skill memory.

The physical WM predicts no controller state. Snapshots are taken AFTER prepare(),
at the same physical state as the replay action. That action is used at t=0;
advance() computes the next command from the predicted next physical state.
"""

import math

import torch
from torch import nn

from ..supermimic_core.kinematics import (
    DifferentiableKinematics,
    build_kinematic_tree,
)


def camera_kinematics(env, robot):
    prefix = f"A{robot + 1}/"
    ids = []
    body = env.m.body(prefix + "neck_camera_frame").id
    while body > 0:
        ids.append(body)
        body = int(env.m.body_parentid[body])
    names = [env.m.body(i).name for i in sorted(ids)]
    joints = [env.m.joint(i).name.rsplit("/", 1)[-1] for i in env.cpu.skills[robot].j]
    fk = DifferentiableKinematics(build_kinematic_tree(env.m, names, joints)).to(env.device)
    return fk, names.index(prefix + "neck_camera_frame")


class SenderController(nn.Module):
    names = ("locomotion", "passing")

    def __init__(self, env):
        super().__init__()
        self.mods = nn.ModuleDict({name: env.mods[name] for name in self.names})
        self.cfg = env.cfg
        self.device = env.device
        self.fk, self.cam = camera_kinematics(env, 0)
        self.widths = {}
        self.memory_dim = 0
        for name, mod in self.mods.items():
            widths = {"history": mod.offsets[-1], "previous": 31, "command": 3,
                      "held": 3 if mod.s["ball"] == "point" else 4,
                      "yaw": 1, "history_valid": 1, "command_valid": 1}
            self.widths[name] = widths
            self.memory_dim += sum(widths.values())
        # Reproduce the real Python elapsed += .02 schedule exactly, including
        # its phase boundary rounding. No future physical state enters this table.
        elapsed = 0.0
        phase, alpha, yaw_reset = [], [], []
        for _ in range(max(1024, int(self.cfg.get("duration", 6.0) / .02) + 128)):
            kick = self.cfg["kick_start"] <= elapsed < self.cfg["kick_end"]
            fade = self.cfg["kick_end"] <= elapsed < self.cfg["kick_end"] + .3
            phase.append(1 if kick else 2 if fade else 0)
            alpha.append(min(1., (elapsed - self.cfg["kick_start"] + .02) / .3)
                         if kick else min(1., (elapsed - self.cfg["kick_end"] + .02) / .3)
                         if fade else 0.)
            yaw_reset.append(kick and elapsed < self.cfg["kick_start"] + .01)
            elapsed += .02
        self.register_buffer("phase_table", torch.tensor(phase, device=env.device))
        self.register_buffer("alpha_table", torch.tensor(alpha, device=env.device))
        self.register_buffer("yaw_reset_table", torch.tensor(yaw_reset, device=env.device))

    def pack(self, states):
        return torch.cat([states[name][key] for name in self.names
                          for key in self.widths[name]], -1)

    def unpack(self, packed):
        states, cursor = {}, 0
        for name in self.names:
            states[name] = {}
            for key, width in self.widths[name].items():
                states[name][key] = packed[:, cursor:cursor + width]
                cursor += width
        return states

    @torch.no_grad()
    def snapshot(self, env):
        states = {}
        for name in self.names:
            memory = env.memory[(0, name)]
            states[name] = {}
            for key, width in self.widths[name].items():
                if key.endswith("_valid"):
                    valid = memory[key[:-6]] is not None
                    value = torch.full((env.n, width), float(valid), device=env.device)
                else:
                    value = memory[key]
                    if value is None:
                        value = torch.zeros(env.n, width, device=env.device)
                    else:
                        value = value.reshape(env.n, -1)
                        if key == "command" and value.shape[-1] == 2:
                            value = torch.cat([value, value.new_zeros(env.n, 1)], -1)
                states[name][key] = value
        return self.pack(states)

    def _native(self, name, state, camera_pos, camera_quat, memory, command, active):
        mod = self.mods[name]
        n = 2 if mod.s["cmd"] == "ball" else 3
        inst = command[:, :n].clamp(
            command.new_tensor(mod.s["cmd_lo"]), command.new_tensor(mod.s["cmd_hi"]))
        filt = torch.where(memory["command_valid"] > .5,
                           .2 * inst + .8 * memory["command"][:, :n], inst)
        cmd = torch.cat([filt, filt.new_zeros(len(filt), 1)], -1) if n == 2 else filt
        q, history, raw, held = mod.observation(
            state, camera_pos, camera_quat, memory["history"], memory["previous"],
            cmd, memory["yaw"].squeeze(-1), memory["held"],
            history_valid=memory["history_valid"].squeeze(-1) > .5)
        updated = dict(history=history, previous=raw, command=cmd, held=held,
                       yaw=memory["yaw"], history_valid=torch.ones_like(memory["history_valid"]),
                       command_valid=torch.ones_like(memory["command_valid"]))
        return q, {k: torch.where(active[:, None], v, memory[k])
                   for k, v in updated.items()}

    def forward(self, s, packed):
        states = self.unpack(packed)
        tick = (s[:, 159].detach() / .02).round().long().clamp(0, len(self.phase_table) - 1)
        phase = self.phase_table[tick]
        alpha = self.alpha_table[tick, None]
        rs = torch.cat([s[:, :75], s[:, 150:159]], -1)
        pos, quat = self.fk(rs[:, :3], rs[:, 3:7], rs[:, 13:44])
        camera_pos, camera_quat = pos[:, self.cam], quat[:, self.cam]
        zero = s.new_zeros(len(s), 3)
        walk, states["locomotion"] = self._native(
            "locomotion", rs, camera_pos, camera_quat, states["locomotion"], zero,
            torch.ones(len(s), device=s.device, dtype=torch.bool))
        q = s[:, 3:7]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2].square() + q[:, 3].square()))
        reset_yaw = self.yaw_reset_table[tick]
        states["passing"]["yaw"] = torch.where(
            reset_yaw[:, None], yaw[:, None], states["passing"]["yaw"])
        delta = s[:, 75:77] - s[:, 150:152]
        angle = torch.atan2(delta[:, 1], delta[:, 0]) - yaw + math.radians(
            self.cfg["pass_direction_bias_deg"])
        command = torch.stack([angle.cos() * self.cfg["kick_speed"],
                               angle.sin() * self.cfg["kick_speed"],
                               torch.zeros_like(angle)], -1)
        command = torch.where((phase == 1)[:, None], command, zero)
        passing, states["passing"] = self._native(
            "passing", rs, camera_pos, camera_quat, states["passing"], command, phase != 0)
        target = torch.where((phase == 1)[:, None], walk * (1 - alpha) + passing * alpha,
                            torch.where((phase == 2)[:, None],
                                        passing * (1 - alpha) + walk * alpha, walk))
        return target, self.pack(states)
