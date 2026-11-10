"""Native MuJoCo stepping. Only reset() writes qpos/qvel; kicks apply joint torques."""

import math
from copy import deepcopy

import mujoco
import numpy as np

from .config import validate
from .scene import build_scene


class SoccerEnv:
    def __init__(self, config, *, robots=True):
        self.config = deepcopy(config)
        validate(self.config)
        self.xml = build_scene(self.config, robots=robots)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.ball_body = m.body("ball").id
        self.ball_geom = m.geom("ball_geom").id
        j = m.joint("ball_free").id
        self.ball_q = int(m.jnt_qposadr[j])
        self.ball_v = int(m.jnt_dofadr[j])
        self.targets = np.zeros(m.nu)
        self.qadrs = np.array(
            [m.jnt_qposadr[j] for j in m.actuator_trnid[:, 0]], dtype=int
        )
        self.vadrs = np.array(
            [m.jnt_dofadr[j] for j in m.actuator_trnid[:, 0]], dtype=int
        )
        self.joint_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j))
            for j in m.actuator_trnid[:, 0]
        ]
        self.actuator_lookup = {name: i for i, name in enumerate(self.joint_names)}
        self.home = np.array(
            [
                self.config["robot"]["home"].get(n.split("/")[-1], 0.0)
                for n in self.joint_names
            ]
        )
        self.reset()

    def reset(self, *, ball_position=None, ball_velocity=None):
        mujoco.mj_resetData(self.model, self.data)
        self.targets[:] = self.home
        self.data.qpos[self.qadrs] = self.home
        b = self.config["ball"]
        self.data.qpos[self.ball_q : self.ball_q + 3] = (
            b["initial_position"] if ball_position is None else ball_position
        )
        self.data.qvel[self.ball_v : self.ball_v + 3] = (
            b["initial_velocity"] if ball_velocity is None else ball_velocity
        )
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def set_joint_targets(self, robot_name, targets):
        """Targets in radians; bounded PD motor torques are evaluated at physics rate."""
        updates = []
        for name, value in targets.items():
            aid = self.actuator_lookup[robot_name + "/" + name]
            j = self.model.actuator_trnid[aid, 0]
            if (
                not np.isfinite(value)
                or not self.model.jnt_range[j, 0] <= value <= self.model.jnt_range[j, 1]
            ):
                raise ValueError(f"Joint target out of range: {name}={value}")
            updates.append((aid, value))
        for aid, value in updates:
            self.targets[aid] = value

    def set_contact(self, kind, *, friction=None, solref=None):
        """Hot update explicit contact pairs, avoiding MuJoCo geom mixing ambiguity."""
        if kind not in self.config["contacts"]:
            raise KeyError(kind)
        updated = deepcopy(self.config)
        if friction is not None:
            updated["contacts"][kind]["friction"] = list(friction)
        if solref is not None:
            updated["contacts"][kind]["solref"] = list(solref)
        validate(updated)
        for i in range(self.model.npair):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_PAIR, i)
            matches = (
                name == kind
                or (kind == "ball_foot" and name.endswith("_ball"))
                or (
                    kind == "foot_turf"
                    and name.endswith("_turf")
                    and name != "ball_turf"
                )
                or (kind == "ball_goal" and name.startswith("ball_goal_"))
            )
            if matches:
                self.model.pair_friction[i] = updated["contacts"][kind]["friction"]
                self.model.pair_solref[i] = updated["contacts"][kind]["solref"]
        self.config = updated

    def step(self, joint_torques=None):
        """One physical timestep. Optional torque action replaces the PD baseline."""
        m, d, c = self.model, self.data, self.config
        if joint_torques is None:
            tau = (
                c["robot"]["kp"] * (self.targets - d.qpos[self.qadrs])
                - c["robot"]["kd"] * d.qvel[self.vadrs]
            )
        else:
            tau = np.asarray(joint_torques, dtype=float)
            if tau.shape != (m.nu,) or not np.isfinite(tau).all():
                raise ValueError(f"Expected finite torque shape {(m.nu,)}")
        d.ctrl[:] = np.clip(
            tau, m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1]
        )
        # Explicit sphere aerodynamic drag only; no hidden constant rolling deceleration.
        v = d.qvel[self.ball_v : self.ball_v + 3]
        speed = np.linalg.norm(v)
        d.xfrc_applied[self.ball_body, :] = 0
        d.xfrc_applied[self.ball_body, :3] = (
            -0.5
            * c["physics"]["air_density"]
            * c["physics"]["air_drag_coefficient"]
            * math.pi
            * c["ball"]["radius"] ** 2
            * speed
            * v
        )
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError("Non-finite simulation state")
        return self.observe()

    def observe(self):
        return {
            "time": float(self.data.time),
            "ball_position": self.data.qpos[self.ball_q : self.ball_q + 3].copy(),
            "ball_velocity": self.data.qvel[self.ball_v : self.ball_v + 3].copy(),
            "ball_angular_velocity": self.data.qvel[
                self.ball_v + 3 : self.ball_v + 6
            ].copy(),
            "joint_position": self.data.qpos[self.qadrs].copy(),
            "joint_velocity": self.data.qvel[self.vadrs].copy(),
            "status": self.ball_status(),
        }

    def ball_status(self):
        x, y, z = self.data.qpos[self.ball_q : self.ball_q + 3]
        c = self.config
        r = c["ball"]["radius"]
        f = c["field"]
        g = c["goal"]
        if (
            abs(x) - r > f["length"] / 2
            and abs(y) + r < g["width"] / 2
            and z + r < g["height"]
        ):
            return "goal_positive_x" if x > 0 else "goal_negative_x"
        if abs(x) - r > f["length"] / 2 or abs(y) - r > f["width"] / 2:
            return "out"
        return "in"

    def ball_contacts(self):
        """Contact names and normal forces (N) for the most recent physics step."""
        records = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self.ball_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.ball_geom else contact.geom1
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            records.append(
                {
                    "geom": mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, int(other)
                    ),
                    "normal_force": float(force[0]),
                }
            )
        return records
