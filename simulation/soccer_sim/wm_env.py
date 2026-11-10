"""Two floating K1s; released walk/kick networks are frozen low-level skills.

The learned action is an eight-dimensional *skill command*, never a ball force.
State includes the skills' recurrent input (last action, phase and cross-fade).
"""

from pathlib import Path

import mujoco
import numpy as np
import yaml

from .config import load_config
from .env import SoccerEnv
from .skill_policies import NUbotsSkills

DEFAULT = Path(__file__).resolve().parents[1] / "configs/wm_pass_facing_gentle.yaml"


class SkillPassEnv:
    def __init__(self, settings=None):
        self.settings = (
            yaml.safe_load(DEFAULT.read_text()) if settings is None else settings
        )
        c = load_config()
        c["robot"]["base_mode"] = "free"
        c["robot"]["base_height"] = 0.555
        c["robot"]["instances"] = c["robot"]["instances"][:2]
        for r, xy in zip(
            c["robot"]["instances"], [[0, 0], self.settings["receiver_xy"]]
        ):
            r.update(position=xy, yaw=0.0)
        c["ball"]["initial_position"] = [5, 4, 0.096]
        self.env = SoccerEnv(c)
        self.m, self.d = self.env.model, self.env.data
        self.policy_stride = round(0.02 / self.m.opt.timestep)
        if self.policy_stride < 1 or not np.isclose(
            self.policy_stride * self.m.opt.timestep, 0.02
        ):
            raise ValueError(
                "Physics timestep must divide the released policies' 0.02 s period"
            )
        if not np.isclose(
            round(self.settings["control_dt"] / 0.02) * 0.02,
            self.settings["control_dt"],
        ):
            raise ValueError("WM control_dt must be a multiple of 0.02 s")
        self.skills = [
            NUbotsSkills(
                self.m,
                r["name"] + "/",
                self.settings.get("kick_kind", "nubots"),
                self.settings.get("kick_speed", 1.0),
            )
            for r in c["robot"]["instances"]
        ]
        for skill in self.skills:
            skill.kp *= self.settings.get("skill_kp_scale", 1.0)
            skill.kd *= self.settings.get("skill_kd_scale", 1.0)
        self.layout = {}
        self.size = 0
        self.root_q = [
            int(self.m.joint(r["name"] + "/world_joint").qposadr[0])
            for r in c["robot"]["instances"]
        ]
        self.link_ids = []
        for r in c["robot"]["instances"]:
            prefix = r["name"] + "/"
            self.link_ids.append(
                [
                    b
                    for b in range(self.m.nbody)
                    if self.m.body(b).name.startswith(prefix)
                ]
            )
        # Layout is established once and exported with every checkpoint.
        self.reset(0)

    def state(self):
        pieces = []

        def add(name, value):
            value = np.asarray(value).reshape(-1)
            start = sum(x.size for x in pieces)
            self.layout[name] = [start, start + value.size]
            pieces.append(value)

        for i, p in enumerate(self.skills):
            add(f"r{i}_position", self.d.xpos[p.body])
            add(f"r{i}_rotation", self.d.xmat[p.body])
            add(f"r{i}_velocity", self.d.qvel[p.root_v : p.root_v + 6])
            add(f"r{i}_q", self.d.qpos[p.q])
            add(f"r{i}_qd", self.d.qvel[p.v])
            add(f"r{i}_last_action", p.last_action)
            add(f"r{i}_target", p.target)
            add(f"r{i}_blend_from", p.blend_from)
            add(f"r{i}_blend_time", [p.blend_time])
            add(f"r{i}_phase", np.r_[np.cos(p.phase), np.sin(p.phase)])
            add(f"r{i}_skill", [float(p.previous_skill == "kick")])
            if p.kick_kind.startswith("bhuman"):
                add(f"r{i}_kick_memory", [p.kick_frequency, *p.old_ball])
                add(f"r{i}_walk_history", p.walk_history)
            add(f"r{i}_body_positions", self.d.xpos[self.link_ids[i]])
        add("ball_position", self.d.qpos[self.env.ball_q : self.env.ball_q + 3])
        add("ball_velocity", self.d.qvel[self.env.ball_v : self.env.ball_v + 6])
        add("time", [self.elapsed])
        add(
            "context",
            [
                self.m.pair_friction[0, 0],
                self.m.body_mass[self.env.ball_body],
                self.env.config["ball"]["radius"],
            ],
        )
        self.size = sum(x.size for x in pieces)
        return np.concatenate(pieces).astype(np.float32)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        self.env.reset()
        self.elapsed = 0.0
        self.receiver_xy = (
            np.array(self.settings["receiver_xy"])
            + rng.uniform(-1, 1, 2) * self.settings["receiver_xy_noise"]
        )
        self.d.qpos[self.root_q[1] : self.root_q[1] + 2] = self.receiver_xy
        self.sender_yaw = 0.0
        if self.settings.get("facing_mode") == "mutual":
            self.sender_yaw = float(
                np.arctan2(self.receiver_xy[1], self.receiver_xy[0])
            )
            for q, yaw in zip(self.root_q, [self.sender_yaw, self.sender_yaw + np.pi]):
                self.d.qpos[q + 3 : q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for p in self.skills:
            p.reset()
            self.d.qpos[p.q] = p.default
        mujoco.mj_forward(self.m, self.d)
        # Initialization only: settle the robot before placing the episode's ball.
        for k in range(round(self.settings["settle_time"] / self.m.opt.timestep)):
            if k % self.policy_stride == 0:
                for p in self.skills:
                    p.act(self.d, "walk", [0, 0, 0], [5, 4, 0.096])
            for p in self.skills:
                p.torque(self.d)
            self.env.step(self.d.ctrl.copy())
        xy = (
            np.array(self.settings["ball_xy"])
            + rng.uniform(-1, 1, 2) * self.settings["ball_xy_noise"]
        )
        if self.settings.get("facing_mode") == "mutual":
            cs, sn = np.cos(self.sender_yaw), np.sin(self.sender_yaw)
            xy = np.array([[cs, -sn], [sn, cs]]) @ xy
        self.d.qpos[self.env.ball_q : self.env.ball_q + 3] = [
            *xy,
            self.env.config["ball"]["radius"] + 0.001,
        ]
        self.d.qvel[self.env.ball_v : self.env.ball_v + 6] = 0
        mujoco.mj_forward(self.m, self.d)
        self.touched = [False, False]
        self.first_contact_times = [None, None]
        self.initial_ball_xy = xy.copy()
        self.max_ball_travel = 0.0
        self.captured = False
        self.capture_time = 0.0
        self.fallen = False
        self.min_distance = np.inf
        self.contact_steps = [0, 0]
        return self.state()

    def step(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        residual = self.settings.get("receiver_joint_residual_scale")
        expected = 20 if residual is not None else 8
        if action.shape != (expected,) or not np.isfinite(action).all():
            raise ValueError(f"Expected {expected} finite normalized skill commands")
        dt = self.settings["control_dt"]
        for k in range(round(dt / self.m.opt.timestep)):
            if k % self.policy_stride == 0:
                ball = self.d.qpos[self.env.ball_q : self.env.ball_q + 3]
                for i, p in enumerate(self.skills):
                    kick = (
                        i == 0
                        and self.settings["kick_start"]
                        <= self.elapsed
                        < self.settings["kick_end"]
                    )
                    if kick:
                        base = np.asarray(
                            self.settings.get("kick_direction", [0.0, 1.0])
                        )
                        if self.settings.get("kick_direction_mode") == "teammate":
                            delta = (
                                self.d.xpos[self.skills[1 - i].body]
                                - self.d.xpos[p.body]
                            )
                            R = self.d.xmat[p.body].reshape(3, 3)
                            yaw = np.arctan2(R[1, 0], R[0, 0])
                            base = (
                                np.array(
                                    [
                                        [np.cos(yaw), np.sin(yaw)],
                                        [-np.sin(yaw), np.cos(yaw)],
                                    ]
                                )
                                @ delta[:2]
                            )
                        angle = self.settings["kick_heading_scale"] * action[6 + i]
                        cs, sn = np.cos(angle), np.sin(angle)
                        direction = np.array([[cs, -sn], [sn, cs]]) @ base
                        if "kick_direction" not in self.settings:
                            direction = np.array(
                                [
                                    self.settings["kick_heading_scale"] * action[6 + i],
                                    1.0,
                                ]
                            )
                        command = direction / np.linalg.norm(direction)
                    else:
                        command = (
                            action[3 * i : 3 * i + 3]
                            * self.settings["walk_command_scale"]
                        )
                        if self.settings.get("facing_mode") == "mutual":
                            delta = (
                                self.d.xpos[self.skills[1 - i].body]
                                - self.d.xpos[p.body]
                            )
                            R = self.d.xmat[p.body].reshape(3, 3)
                            error = np.arctan2(delta[1], delta[0]) - np.arctan2(
                                R[1, 0], R[0, 0]
                            )
                            error = np.arctan2(np.sin(error), np.cos(error))
                            command[2] = np.clip(
                                self.settings.get("face_yaw_gain", 1.5) * error,
                                -self.settings.get("face_max_yaw_rate", 0.6),
                                self.settings.get("face_max_yaw_rate", 0.6),
                            )
                    p.act(self.d, "kick" if kick else "walk", command, ball)
                    if i == 1 and residual is not None:
                        # Real target-angle authority for receiving; no ball forces.
                        p.target[10:] = np.clip(
                            p.target[10:] + np.asarray(residual) * action[8:],
                            p.lower[10:], p.upper[10:],
                        )
            for p in self.skills:
                p.torque(self.d)
            self.env.step(self.d.ctrl.copy())
            touching = [False, False]
            for contact in self.env.ball_contacts():
                for i, name in enumerate(["A1/", "A2/"]):
                    if (
                        contact["geom"]
                        and contact["geom"].startswith(name)
                        and contact["normal_force"] > 0.1
                    ):
                        self.touched[i] = touching[i] = True
                        if self.first_contact_times[i] is None:
                            self.first_contact_times[i] = self.elapsed
                        self.contact_steps[i] += 1
            ball = self.d.qpos[self.env.ball_q : self.env.ball_q + 3]
            self.max_ball_travel = max(
                self.max_ball_travel, np.linalg.norm(ball[:2] - self.initial_ball_xy)
            )
            distance = np.linalg.norm(ball[:2] - self.d.xpos[self.skills[1].body, :2])
            self.min_distance = min(self.min_distance, distance)
            slow = (
                np.linalg.norm(self.d.qvel[self.env.ball_v : self.env.ball_v + 2])
                < self.settings["capture_speed"]
            )
            self.capture_time = (
                self.capture_time + self.m.opt.timestep
                if self.touched[1]
                and distance < self.settings.get("capture_radius", 0.3)
                and slow
                and self.facing_valid()
                else 0.0
            )
            self.captured |= self.capture_time >= self.settings["capture_hold"]
            for p in self.skills:
                self.fallen |= (
                    self.d.xpos[p.body, 2] < self.settings["fall_height"]
                    or self.d.xmat[p.body].reshape(3, 3)[2, 2]
                    < self.settings["fall_upright"]
                )
            self.elapsed += self.m.opt.timestep
        return self.state(), self.elapsed >= self.settings[
            "duration"
        ] - 1e-8 or self.fallen

    def metrics(self):
        ball = self.d.qpos[self.env.ball_q : self.env.ball_q + 3]
        stable = self.capture_time >= self.settings["capture_hold"]
        ordered = (
            all(t is not None for t in self.first_contact_times)
            and self.first_contact_times[0] < self.first_contact_times[1]
        )
        return {
            "sender_contact": self.touched[0],
            "receiver_contact": self.touched[1],
            "captured": self.captured,
            "stable_receive_at_end": bool(stable),
            "pass_success": bool(
                ordered
                and self.max_ball_travel
                >= self.settings.get("minimum_pass_travel", 0.3)
                and self.captured
                and stable
                and not self.fallen
            ),
            "fallen": bool(self.fallen),
            "min_receiver_distance": float(self.min_distance),
            "final_target_error": float(np.linalg.norm(ball[:2] - self.receiver_xy)),
            "final_receiver_distance": float(
                np.linalg.norm(ball[:2] - self.d.xpos[self.skills[1].body, :2])
            ),
            "final_ball_speed": float(
                np.linalg.norm(self.d.qvel[self.env.ball_v : self.env.ball_v + 2])
            ),
            "final_receiver_position": self.d.xpos[self.skills[1].body].tolist(),
            "final_ball": ball.tolist(),
            "contact_steps": self.contact_steps,
            "first_contact_times": self.first_contact_times,
            "max_ball_travel": float(self.max_ball_travel),
            "facing_valid_at_end": bool(self.facing_valid()),
            "heading_alignment": [
                float(
                    np.dot(
                        self.d.xmat[p.body].reshape(3, 3)[:2, 0],
                        (
                            (
                                self.d.xpos[self.skills[1 - i].body]
                                - self.d.xpos[p.body]
                            )[:2]
                        )
                        / max(
                            np.linalg.norm(
                                (
                                    self.d.xpos[self.skills[1 - i].body]
                                    - self.d.xpos[p.body]
                                )[:2]
                            ),
                            1e-8,
                        ),
                    )
                )
                for i, p in enumerate(self.skills)
            ],
        }

    def facing_valid(self):
        if self.settings.get("facing_mode") != "mutual":
            return True
        threshold = np.cos(np.deg2rad(self.settings.get("facing_tolerance_deg", 30.0)))
        for i, p in enumerate(self.skills):
            delta = (self.d.xpos[self.skills[1 - i].body] - self.d.xpos[p.body])[:2]
            forward = self.d.xmat[p.body].reshape(3, 3)[:2, 0]
            if (
                np.dot(forward, delta)
                / max(np.linalg.norm(delta) * np.linalg.norm(forward), 1e-8)
                < threshold
            ):
                return False
        return True
