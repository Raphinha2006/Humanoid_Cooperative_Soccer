"""Native G1 public Joga inference interface, reimplemented for Python MuJoCo.

Resource provenance is configs/joga_sources.json. This is not Joga training code.
Observations and actuator semantics follow the public demo sim.js v36.
"""

import json
import struct
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets/joga"


class JMLP:
    def __init__(self, path):
        b = path.read_bytes()
        if b[:4] != b"JMLP":
            raise ValueError("not a public JMLP weight")
        version, n, self.dim = struct.unpack_from("<III", b, 4)
        if version != 1:
            raise ValueError(version)
        self.eps = struct.unpack_from("<f", b, 16)[0]
        shape = [struct.unpack_from("<II", b, 20 + 8 * i) for i in range(n)]
        off = 20 + 8 * n

        def read(count):
            nonlocal off
            a = np.frombuffer(b, dtype="<f4", count=count, offset=off).astype(
                np.float64
            )
            off += 4 * count
            return a

        self.mean, self.std = read(self.dim), read(self.dim)
        self.layers = [(read(o * i).reshape(o, i), read(o)) for o, i in shape]
        if off != len(b):
            raise ValueError("weight byte count")

    def __call__(self, obs):
        x = (np.asarray(obs, dtype=np.float32).astype(np.float64) - self.mean) / (
            self.std + self.eps
        )
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i + 1 < len(self.layers):
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return x.astype(np.float32)


def bbox(p, r, k):
    x, y, z = p
    hu = np.arcsin(min(r / max(np.hypot(x, y), r + 1e-6), 1 - 1e-6))
    hv = np.arcsin(min(r / max(np.hypot(x, z), r + 1e-6), 1 - 1e-6))
    pu, pv = np.arctan2(y, x), np.arctan2(z, x)
    ua, ub = k["cx"] - k["fx"] * np.tan(
        np.clip([pu - hu, pu + hu], -np.deg2rad(89), np.deg2rad(89))
    )
    va, vb = k["cy"] - k["fy"] * np.tan(
        np.clip([pv - hv, pv + hv], -np.deg2rad(89), np.deg2rad(89))
    )
    x1, x2 = sorted([ua, ub])
    y1, y2 = sorted([va, vb])
    visible = x > 0.05 and x2 >= 0 and x1 < k["width"] and y2 >= 0 and y1 < k["height"]
    return np.clip(
        [x1, y1, x2, y2],
        0,
        [k["width"] - 1, k["height"] - 1, k["width"] - 1, k["height"] - 1],
    ), visible


class JogaSkills:
    def __init__(self, model, prefix, names):
        self.spec = json.loads((ASSETS / "skills.json").read_text())
        self.mods = {}
        self.j = np.array(
            [model.joint(prefix + n).id for n in self.spec["joint_names"]]
        )
        self.q = model.jnt_qposadr[self.j]
        self.v = model.jnt_dofadr[self.j]
        # Resolve transmission IDs instead of guessing actuator spelling.
        self.a = np.array(
            [np.flatnonzero(model.actuator_trnid[:, 0] == j)[0] for j in self.j]
        )
        self.body = model.body(prefix + "pelvis").id
        self.cam = model.body(prefix + "neck_camera_frame").id
        self.root_v = int(model.joint(prefix + "floating_base_joint").dofadr[0])
        self.root_q = int(model.joint(prefix + "floating_base_joint").qposadr[0])
        for name in names:
            s = self.spec["modules"][name]
            self.mods[name] = {
                "spec": s,
                "net": JMLP(ASSETS / s["weights"]),
                "default": np.array(s["default_q"]),
            }
        self.kp = np.array(self.spec["kp"])
        self.kd = np.array(self.spec["kd"])
        self.limit = np.array(self.spec["max_torque"])
        self.maxvel = np.array(self.spec["max_vel"])
        self.lower = np.array(self.spec["lower"])
        self.upper = np.array(self.spec["upper"])
        self.scale = np.array(self.spec["action_scale"])
        self.posctrl = np.array(self.spec["pos_ctrl_mask"])
        self.reset(names[0])

    def reset(self, active):
        self.active = active
        self.switch_from = None
        self.blend = 0
        self.target = self.mods[active]["default"].copy()
        self.qcmd = self.target.copy()
        for mod in self.mods.values():
            mod.update(
                history=None,
                previous=np.zeros(31),
                filtered=None,
                yaw_offset=0.0,
                point=np.zeros(3),
                box=np.ones(4),
                visible=False,
            )

    def yaw(self, d):
        r = d.xmat[self.body].reshape(3, 3)
        return np.arctan2(r[1, 0], r[0, 0])

    def infer(self, d, name, command, ball_body):
        m = self.mods[name]
        s = m["spec"]
        command = np.asarray(command)
        n = 2 if s["cmd"] == "ball" else 3
        inst = np.clip(command[:n], s["cmd_lo"], s["cmd_hi"])
        m["filtered"] = (
            inst if m["filtered"] is None else 0.2 * inst + 0.8 * m["filtered"]
        )
        cmd = np.r_[m["filtered"], 0] if n == 2 else m["filtered"]
        yaw = (self.yaw(d) - m["yaw_offset"] + np.pi) % (2 * np.pi) - np.pi
        cr = d.xmat[self.cam].reshape(3, 3)
        p = cr.T @ (d.xpos[ball_body] - d.xpos[self.cam])
        box, visible = bbox(
            p,
            s.get("bbox_radius", self.spec["intrinsics"]["ball_radius"]),
            self.spec["intrinsics"],
        )
        if s["ball"] == "bbox":
            visible = visible and np.linalg.norm(p) <= s["max_detect_dist"]
        if visible:
            m["point"] = p.copy()
            m["box"] = box.copy()
        m["visible"] = visible
        ball = np.r_[m["point"] if s["ball"] == "point" else m["box"], int(visible)]
        w, x, y, z = d.qpos[self.root_q + 3 : self.root_q + 7]
        gravity = np.array(
            [-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))]
        )
        terms = [
            d.qvel[self.root_v + 3 : self.root_v + 6],
            gravity,
            d.qpos[self.q] - m["default"],
            d.qvel[self.v],
            m["previous"].copy(),
            cmd,
            [yaw],
            ball,
        ]
        if m["history"] is None:
            m["history"] = [np.tile(t, (3, 1)) for t in terms]
        else:
            m["history"] = [
                np.concatenate((h[1:], np.asarray(t)[None]))
                for h, t in zip(m["history"], terms)
            ]
        obs = np.concatenate([h.ravel() for h in m["history"]]).astype(np.float32)
        if len(obs) != s["obs_dim"]:
            raise ValueError((name, len(obs), s["obs_dim"]))
        action = m["net"](obs)
        m["pending"] = action.copy()
        return m["default"] + self.scale * action

    def act(self, d, skill, command, ball_body, residual=None):
        if skill != self.active:
            self.switch_from = self.active
            self.active = skill
            self.blend = 0
            m = self.mods[skill]
            m["history"] = None
            m["filtered"] = None
            if skill in ["passing", "shooting", "dribbling"]:
                m["yaw_offset"] = self.yaw(d)
        q = self.infer(d, skill, command, ball_body)
        if self.switch_from is not None:
            old = self.infer(d, self.switch_from, np.zeros(3), ball_body)
            self.blend += 1
            alpha = min(1, self.blend / 15)
            q = (1 - alpha) * old + alpha * q
            if self.blend >= 15:
                self.switch_from = None
        if residual is not None:
            q[:12] += residual
        self.target = q.copy()
        qi, qd = d.qpos[self.q], d.qvel[self.v]
        vc = np.clip(qd, -2 * self.maxvel, 2 * self.maxvel)
        hi = np.minimum(self.limit * (1 - vc / self.maxvel), self.limit)
        lo = np.maximum(self.limit * (-1 - vc / self.maxvel), -self.limit)
        self.qcmd = np.clip(
            q, qi + (lo + self.kd * qd) / self.kp, qi + (hi + self.kd * qd) / self.kp
        )
        self.qcmd[self.posctrl] = np.clip(
            q[self.posctrl], self.lower[self.posctrl], self.upper[self.posctrl]
        )
        self.mods[skill]["previous"] = self.mods[skill]["pending"].copy()

    def torque(self, d):
        d.ctrl[self.a] = (
            self.kp * (self.qcmd - d.qpos[self.q]) - self.kd * d.qvel[self.v]
        )

    def memory(self):
        pieces = [
            self.qcmd,
            self.target,
            [self.blend, float(self.switch_from is not None)],
        ]
        for name, m in self.mods.items():
            hist = (
                np.zeros(m["spec"]["obs_dim"])
                if m["history"] is None
                else np.concatenate([h.ravel() for h in m["history"]])
            )
            filt = (
                np.zeros(3)
                if m["filtered"] is None
                else np.pad(m["filtered"], (0, 3 - len(m["filtered"])))
            )
            pieces.extend(
                [
                    hist,
                    m["previous"],
                    filt,
                    [m["yaw_offset"]],
                    m["point"],
                    m["box"],
                    [float(self.active == name)],
                ]
            )
        return np.concatenate([np.asarray(x).ravel() for x in pieces])
