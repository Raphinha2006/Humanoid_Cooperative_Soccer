"""True batched MuJoCo Warp, two G1s and one physical ball per world."""

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

from ..joga_env import JogaPassEnv
from ..supermimic_core.math import quat_apply, quat_inv
from .native import Native


@wp.kernel
def pd_kernel(
    q: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    cmd: wp.array2d(dtype=float),
    qi: wp.array(dtype=int),
    vi: wp.array(dtype=int),
    ai: wp.array(dtype=int),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
):
    w, j = wp.tid()
    ctrl[w, ai[j]] = kp[j] * (cmd[w, j] - q[w, qi[j]]) - kd[j] * v[w, vi[j]]


@wp.kernel
def tick_clock(t: wp.array(dtype=float)):
    t[0] = t[0] + 0.005


@wp.kernel
def mark_contacts(
    cone: int,
    frame: wp.array(dtype=wp.mat33),
    friction: wp.array(dtype=vec5),
    dim: wp.array(dtype=int),
    addr: wp.array2d(dtype=int),
    world: wp.array(dtype=int),
    force: wp.array2d(dtype=float),
    njmax: int,
    nacon: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    owner: wp.array(dtype=int),
    ball: int,
    clock: wp.array(dtype=float),
    times: wp.array2d(dtype=float),
):
    c = wp.tid()
    if c < nacon[0]:
        pair = geom[c]
        other = -1
        if pair[0] == ball:
            other = pair[1]
        if pair[1] == ball:
            other = pair[0]
        if other >= 0:
            side = owner[other]
            w = world[c]
            if side >= 0:
                f = contact_force_fn(
                    cone, frame, friction, dim, addr, force, njmax, nacon, w, c, False
                )
                if f[0] > 0.1:
                    wp.atomic_min(times, w, side, clock[0])


class BatchEnv:
    def __init__(self, cfg, nworld, device="cuda:0"):
        self.cfg = cfg
        self.n = nworld
        self.device = torch.device(device)
        self.cpu = JogaPassEnv(cfg)
        self.m = self.cpu.m
        self.spec = self.cpu.spec
        wp.init()
        wp.set_device(device)
        self.stream = wp.get_stream(device)
        torch.cuda.set_stream(
            torch.cuda.ExternalStream(self.stream.cuda_stream, device=self.device)
        )
        self.m.opt.ls_iterations = 50
        self.wm = mjw.put_model(self.m)
        self.wd = mjw.put_data(
            self.m, self.cpu.d, nworld=nworld,
            nconmax=int(cfg.get("gpu_nconmax", 96)),
            njmax=int(cfg.get("gpu_njmax", 512)),
        )
        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.ctrl = wp.to_torch(self.wd.ctrl)
        self.xpos = wp.to_torch(self.wd.xpos)
        self.xquat = wp.to_torch(self.wd.xquat)
        self.indices = []
        for p in self.cpu.skills:
            self.indices.append(
                {
                    k: torch.tensor(getattr(p, k), device=device)
                    if k in ["q", "v", "a"]
                    else getattr(p, k)
                    for k in ["q", "v", "a", "root_q", "root_v", "body", "cam"]
                }
            )
        self.mods = {
            n: Native(self.spec, n).to(device)
            for n in ["locomotion", "passing", "receiving"]
        }
        self.controller_fk = {}
        self.controller_camera = {}
        if cfg.get("controller_kinematics", "legacy") == "current":
            from .controller import camera_kinematics
            self.controller_fk = {i: camera_kinematics(self, i) for i in range(2)}
        for name in ["kp", "kd", "max_torque", "max_vel", "lower", "upper"]:
            setattr(
                self,
                name,
                torch.tensor(self.spec[name], device=device, dtype=torch.float32),
            )
        self.posctrl = torch.tensor(
            self.spec["pos_ctrl_mask"], device=device, dtype=torch.bool
        )
        self.target = torch.zeros(nworld, 62, device=device)
        self.qcmd = torch.zeros_like(self.target)
        # PD executes each physics step, without copying state to CPU.
        self.qa = torch.cat([p["q"] for p in self.indices])
        self.va = torch.cat([p["v"] for p in self.indices])
        self.aa = torch.cat([p["a"] for p in self.indices])
        self.kp2 = self.kp.repeat(2)
        self.kd2 = self.kd.repeat(2)
        self.qa_wp = wp.from_torch(self.qa.to(torch.int32))
        self.va_wp = wp.from_torch(self.va.to(torch.int32))
        self.aa_wp = wp.from_torch(self.aa.to(torch.int32))
        self.kp_wp = wp.from_torch(self.kp2)
        self.kd_wp = wp.from_torch(self.kd2)
        self.cmd_wp = wp.from_torch(self.qcmd)
        self.contact_times = wp.full((nworld, 2), 1e6, dtype=float, device=device)
        self.contacts_t = wp.to_torch(self.contact_times)
        self.clock = wp.zeros(1, dtype=float, device=device)
        owner = np.full(self.m.ngeom, -1, dtype=np.int32)
        for g in range(self.m.ngeom):
            name = self.m.body(int(self.m.geom_bodyid[g])).name
            if name.startswith("A1/"):
                owner[g] = 0
            if name.startswith("A2/"):
                owner[g] = 1
        self.owner = wp.array(owner, dtype=int, device=device)
        self.ball_geom = self.m.geom("soccerball_geom").id
        self._physics(None)
        with wp.ScopedCapture() as cap:
            for _ in range(self.cpu.substeps):
                self._physics(None)
        self.graph = cap.graph
        self.reset(104)

    def _physics(self, _):
        wp.launch(
            pd_kernel,
            dim=(self.n, 62),
            inputs=[
                self.wd.qpos,
                self.wd.qvel,
                self.wd.ctrl,
                self.cmd_wp,
                self.qa_wp,
                self.va_wp,
                self.aa_wp,
                self.kp_wp,
                self.kd_wp,
            ],
        )
        mjw.step(self.wm, self.wd)
        wp.launch(tick_clock, dim=1, inputs=[self.clock])
        wp.launch(
            mark_contacts,
            dim=self.wd.naconmax,
            inputs=[
                self.wm.opt.cone,
                self.wd.contact.frame,
                self.wd.contact.friction,
                self.wd.contact.dim,
                self.wd.contact.efc_address,
                self.wd.contact.worldid,
                self.wd.efc.force,
                self.wd.njmax,
                self.wd.nacon,
                self.wd.contact.geom,
                self.owner,
                self.ball_geom,
                self.clock,
                self.contact_times,
            ],
        )

    def physical(self):
        parts = []
        for p in self.indices:
            q, v = p["root_q"], p["root_v"]
            parts += [
                self.qpos[:, q : q + 7],
                self.qvel[:, v : v + 6],
                self.qpos[:, p["q"]],
                self.qvel[:, p["v"]],
            ]
        return torch.cat(
            parts
            + [
                self.qpos[:, self.cpu.ball_q : self.cpu.ball_q + 3],
                self.qvel[:, self.cpu.ball_v : self.cpu.ball_v + 6],
                torch.full((self.n, 1), self.elapsed, device=self.device),
            ],
            -1,
        )

    def check_capacity(self):
        # MJWarp overflow flags persist across substeps; check training as well as evaluation.
        overflow = self.wd.overflow.numpy()
        capacity = int((overflow & 511).max())
        if capacity:
            raise RuntimeError(f"MuJoCo Warp training capacity overflow: {capacity}")
        return {"capacity_overflow_max": capacity,
                "solver_limit_world_fraction": float(np.mean((overflow & 1536) != 0))}

    def receiver(self, s):
        return torch.cat([s[:, 75:150], s[:, 150:159]], -1)

    def reset(self, seed):
        self.completed_world_episodes = getattr(self, "completed_world_episodes", 0) + (
            self.n if getattr(self, "tick", 0) >= 300 else 0
        )
        self.phase_transitions = getattr(
            self, "phase_transitions", {"pre_kick": 0, "kick": 0, "post_kick": 0}
        )
        self.reset_seed = seed
        self.cpu.reset(seed)
        self.elapsed = 0.0
        self.tick = 0
        self.total_steps = getattr(self, "total_steps", 0)
        mjw.reset_data(self.wm, self.wd)
        q = (
            torch.tensor(self.cpu.d.qpos, device=self.device, dtype=torch.float32)
            .expand(self.n, -1)
            .clone()
        )
        v = (
            torch.tensor(self.cpu.d.qvel, device=self.device, dtype=torch.float32)
            .expand(self.n, -1)
            .clone()
        )
        gen = torch.Generator(device=self.device).manual_seed(seed)
        noise = (
            torch.rand((self.n, 2), device=self.device, generator=gen) * 2 - 1
        ) * torch.tensor(self.cfg["receiver_xy_noise"], device=self.device)
        p = self.indices[1]
        q[:, p["root_q"] : p["root_q"] + 2] += noise
        q[:, self.cpu.ball_q : self.cpu.ball_q + 2] += (
            torch.rand((self.n, 2), device=self.device, generator=gen) * 2 - 1
        ) * torch.tensor(self.cfg["ball_xy_noise"], device=self.device)
        self.qpos.copy_(q)
        self.qvel.copy_(v)
        self.wd.qacc_warmstart.zero_()
        mjw.forward(self.wm, self.wd)
        self.contact_times.fill_(1e6)
        self.clock.zero_()
        self.memory = {}
        for i, names in enumerate([["locomotion", "passing"], ["receiving"]]):
            for name in names:
                mod = self.mods[name]
                cpu = self.cpu.skills[i].mods[name]
                self.memory[(i, name)] = {
                    "history": None,
                    "previous": torch.zeros(self.n, 31, device=self.device),
                    "command": None,
                    "held": torch.tensor(
                        cpu["point"] if mod.s["ball"] == "point" else cpu["box"],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    .expand(self.n, -1)
                    .clone(),
                    "yaw": torch.zeros(self.n, device=self.device),
                }
                if cpu["history"] is not None:
                    self.memory[(i, name)]["history"] = (
                        torch.tensor(
                            np.concatenate([h.ravel() for h in cpu["history"]]),
                            device=self.device,
                            dtype=torch.float32,
                        )
                        .expand(self.n, -1)
                        .clone()
                    )
                    self.memory[(i, name)]["previous"] = (
                        torch.tensor(
                            cpu["previous"], device=self.device, dtype=torch.float32
                        )
                        .expand(self.n, -1)
                        .clone()
                    )
        self.target.copy_(
            torch.tensor(
                np.concatenate([p.target for p in self.cpu.skills]),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.qcmd.copy_(
            torch.tensor(
                np.concatenate([p.qcmd for p in self.cpu.skills]),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.max_hold = torch.zeros(self.n, device=self.device)
        self.handoff_hold = torch.zeros(self.n, device=self.device)
        self.hold = torch.zeros(self.n, device=self.device)
        self.captured = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.fallen = self.captured.clone()
        self.contact = torch.zeros(self.n, 2, dtype=torch.bool, device=self.device)
        self.max_travel = torch.zeros(self.n, device=self.device)
        self.initial_ball = self.qpos[:, self.cpu.ball_q : self.cpu.ball_q + 2].clone()
        return self.physical()

    def native(self, i, name, command):
        p = self.indices[i]
        s = self.physical()
        rs = torch.cat([s[:, 75 * i : 75 * (i + 1)], s[:, 150:159]], -1)
        m = self.memory[(i, name)]
        mod = self.mods[name]
        n = 2 if mod.s["cmd"] == "ball" else 3
        inst = command[:, :n].clamp(
            torch.tensor(mod.s["cmd_lo"], device=self.device),
            torch.tensor(mod.s["cmd_hi"], device=self.device),
        )
        filt = inst if m["command"] is None else 0.2 * inst + 0.8 * m["command"]
        m["command"] = filt
        cmd = (
            torch.cat([filt, torch.zeros(self.n, 1, device=self.device)], -1)
            if n == 2
            else filt
        )
        q, h, raw, held = mod.observation(
            rs,
            self.controller_camera[i][0] if self.controller_fk else self.xpos[:, p["cam"]],
            self.controller_camera[i][1] if self.controller_fk else self.xquat[:, p["cam"]],
            m["history"],
            m["previous"],
            cmd,
            m["yaw"],
            m["held"],
        )
        m.update(history=h, previous=raw, held=held)
        return q

    def prepare(self):
        physical = None
        if self.controller_fk:
            # step() ends with integration; cached xpos/xquat otherwise describe
            # the preceding substep while physical() already exposes new qpos.
            # Use the same FK arithmetic as imagination, avoiding camera-visibility
            # discontinuities caused by different float32 transform implementations.
            physical = self.physical()
            for i, (fk, camera) in self.controller_fk.items():
                off = 75 * i
                pos, quat = fk(physical[:, off:off + 3], physical[:, off + 3:off + 7],
                               physical[:, off + 13:off + 44])
                self.controller_camera[i] = (pos[:, camera], quat[:, camera])
        zero = torch.zeros(self.n, 3, device=self.device)
        walk = self.native(0, "locomotion", zero)
        kick = self.cfg["kick_start"] <= self.elapsed < self.cfg["kick_end"]
        if kick:
            p = self.indices[0]
            quat = physical[:, 3:7] if physical is not None else self.xquat[:, p["body"]]
            yaw = torch.atan2(
                2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
            )
            mem = self.memory[(0, "passing")]
            if self.elapsed < self.cfg["kick_start"] + 0.01:
                mem["yaw"] = yaw.clone()
            delta = (
                (physical[:, 75:77] if physical is not None else self.xpos[:, self.indices[1]["body"], :2])
                - self.qpos[:, self.cpu.ball_q : self.cpu.ball_q + 2]
            )
            angle = (
                torch.atan2(delta[:, 1], delta[:, 0])
                - yaw
                + np.deg2rad(self.cfg["pass_direction_bias_deg"])
            )
            cmd = torch.stack(
                [
                    angle.cos() * self.cfg["kick_speed"],
                    angle.sin() * self.cfg["kick_speed"],
                    torch.zeros_like(angle),
                ],
                -1,
            )
            passing = self.native(0, "passing", cmd)
            alpha = min(1.0, (self.elapsed - self.cfg["kick_start"] + 0.02) / 0.3)
            sender = walk * (1 - alpha) + passing * alpha
        elif (
            self.elapsed < self.cfg["kick_end"] + 0.3
            and self.elapsed >= self.cfg["kick_end"]
        ):
            passing = self.native(0, "passing", zero)
            alpha = min(1.0, (self.elapsed - self.cfg["kick_end"] + 0.02) / 0.3)
            sender = passing * (1 - alpha) + walk * alpha
        else:
            sender = walk
        receiver = self.native(1, "receiving", zero)
        self.target.copy_(torch.cat([sender, receiver], -1))
        return self.target

    def step(self, target):
        # Enforce native torque-speed clipping once per 50 Hz, then PD at 200 Hz.
        q = self.qpos[:, self.qa]
        v = self.qvel[:, self.va]
        vel = v.clamp(-2 * self.max_vel.repeat(2), 2 * self.max_vel.repeat(2))
        limit = self.max_torque.repeat(2)
        hi = torch.minimum(limit * (1 - vel / self.max_vel.repeat(2)), limit)
        lo = torch.maximum(limit * (-1 - vel / self.max_vel.repeat(2)), -limit)
        self.qcmd.copy_(
            target.clamp(
                q + (lo + self.kd2 * v) / self.kp2, q + (hi + self.kd2 * v) / self.kp2
            )
        )
        mask = self.posctrl.repeat(2)
        self.qcmd[:, mask] = target[:, mask].clamp(
            self.lower.repeat(2)[mask], self.upper.repeat(2)[mask]
        )
        wp.capture_launch(self.graph)
        self.elapsed += 0.02
        self.tick += 1
        self.total_steps += 1
        phase = (
            "pre_kick"
            if self.elapsed < self.cfg["kick_start"]
            else "kick"
            if self.elapsed < self.cfg["kick_end"]
            else "post_kick"
        )
        self.phase_transitions[phase] += self.n
        s = self.physical()
        p = s[:, 75:78]
        ball = s[:, 150:153]
        speed = s[:, 153:155].norm(dim=-1)
        dist = (ball[:, :2] - p[:, :2]).norm(dim=-1)
        for off in [0, 75]:
            self.fallen |= (s[:, off + 2] < self.cfg["fall_height"]) | (
                1 - 2 * s[:, off + 4 : off + 6].square().sum(-1)
                < self.cfg["fall_upright"]
            )
        # Contact forces are consumed separately by the final evaluator; no geometric proxy accepted as a true-contact success.
        self.max_travel = torch.maximum(
            getattr(self, "max_travel", torch.zeros_like(dist)),
            (ball[:, :2] - self.initial_ball).norm(dim=-1),
        )
        facing = self.facing(s)
        ordered = (self.contacts_t[:, 0] < self.contacts_t[:, 1]) & (
            self.contacts_t[:, 1] < 1e5
        )
        self.handoff_hold = torch.where(
            ordered & facing & ~self.fallen & (dist <= 1.2) & (speed < 0.6),
            self.handoff_hold + 0.02,
            0.0,
        )
        self.hold = torch.where(
            ordered
            & facing
            & (dist < self.cfg["capture_radius"])
            & (speed < self.cfg["capture_speed"])
            & ~self.fallen,
            self.hold + 0.02,
            0.0,
        )
        self.max_hold = torch.maximum(
            getattr(self, "max_hold", torch.zeros_like(self.hold)), self.hold
        )
        self.captured |= self.hold >= self.cfg["capture_hold"]
        return s

    def facing(self, s):
        ok = torch.ones(len(s), device=s.device, dtype=torch.bool)
        for a, b in [(0, 75), (75, 0)]:
            f = quat_apply(
                s[:, a + 3 : a + 7], s.new_tensor([1.0, 0.0, 0.0]).expand(len(s), -1)
            )[:, :2]
            d = s[:, b : b + 2] - s[:, a : a + 2]
            ok &= (f * d).sum(-1) / (f.norm(dim=-1) * d.norm(dim=-1)).clamp_min(
                1e-6
            ) > np.cos(np.deg2rad(self.cfg["facing_tolerance_deg"]))
        return ok

    def metrics(self):
        s = self.physical()
        rel = quat_apply(quat_inv(s[:, 78:82]), s[:, 150:153] - s[:, 75:78])
        speed = s[:, 153:155].norm(dim=-1)
        ordered = (self.contacts_t[:, 0] < self.contacts_t[:, 1]) & (
            self.contacts_t[:, 1] < 1e5
        )
        strict = (
            (self.hold >= self.cfg["capture_hold"])
            & (self.max_travel >= 2)
            & ~self.fallen
            & (rel[:, 0] > 0.03)
            & (rel[:, 0] < 0.55)
            & (rel[:, 1].abs() < 0.3)
        )
        self.strict = strict
        return {
            "front_pass_success": strict.float().mean().item(),
            "captured": self.captured.float().mean().item(),
            "fallen": self.fallen.float().mean().item(),
            "ordered_contact": ordered.float().mean().item(),
            "handoff_at_end": ((self.handoff_hold >= 0.8) & ~self.fallen)
            .float()
            .mean()
            .item(),
            "max_hold_mean": self.max_hold.mean().item(),
            "mean_hold": self.hold.mean().item(),
            "final_distance": (s[:, 150:152] - s[:, 75:77]).norm(dim=-1).mean().item(),
            "final_speed": speed.mean().item(),
        }
