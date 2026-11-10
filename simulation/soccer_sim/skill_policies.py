"""Independent released K1 walk/kick networks, with a common torque backend."""

import json

import numpy as np
import onnxruntime as ort
import yaml

from .pretrained import POLICIES


class NUbotsSkills:
    def __init__(self, model, prefix, kick_kind="nubots", kick_speed=1.0):
        self.base = POLICIES / "nubots/module/skill"
        self.specs = {}
        self.nets = {}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        for skill, directory in [("walk", "SimK1Walk"), ("kick", "SimK1Kick")]:
            self.specs[skill] = yaml.safe_load(
                (
                    self.base / directory / "data/config" / (directory + ".yaml")
                ).read_text()
            )
            self.nets[skill] = ort.InferenceSession(
                str(self.base / directory / "data" / (skill + ".onnx")),
                opts,
                providers=["CPUExecutionProvider"],
            )
        names = ["head_yaw", "head_pitch"]
        names += [
            side + "_" + j
            for side in ["left", "right"]
            for j in ["shoulder_pitch", "shoulder_roll", "elbow_pitch", "elbow_yaw"]
        ]
        names += [
            side + "_" + j
            for side in ["left", "right"]
            for j in [
                "hip_pitch",
                "hip_roll",
                "hip_yaw",
                "knee_pitch",
                "ankle_pitch",
                "ankle_roll",
            ]
        ]
        resolved = []
        for name in names:
            for candidate in [
                prefix + name + "_joint",
                prefix + "aa" + name + "_joint",
            ]:
                try:
                    model.joint(candidate)
                    resolved.append(candidate)
                    break
                except KeyError:
                    continue
            else:
                raise KeyError(prefix + name)
        self.j = np.array([model.joint(n).id for n in resolved])
        self.q = model.jnt_qposadr[self.j]
        self.v = model.jnt_dofadr[self.j]
        self.a = np.array([model.actuator(n).id for n in resolved])
        self.root_v = int(model.joint(prefix + "world_joint").dofadr[0])
        self.body = model.body(prefix + "trunk").id
        self.default = np.array(self.specs["walk"]["default_pose"])
        self.kp = np.array(self.specs["walk"]["kp"])
        self.kd = np.array(self.specs["walk"]["kd"])
        self.scale = np.array(self.specs["walk"]["action_scale_joint"])
        model.dof_damping[self.v] = 0  # damping is applied exactly once by torque().
        self.limit = np.max(np.abs(model.actuator_forcerange[self.a]), axis=1)
        self.lower = np.array(self.specs["walk"]["joint_lower"])
        self.upper = np.array(self.specs["walk"]["joint_upper"])
        self.kick_kind, self.kick_speed = kick_kind, kick_speed
        if kick_kind.startswith("bhuman"):
            name = (
                "K1_kick_inner_soft.onnx"
                if kick_kind == "bhuman_soft"
                else "K1_kick_inner.onnx"
            )
            self.bhuman_net = ort.InferenceSession(
                str(POLICIES / "bhuman/Config/NeuralNets/Walk" / name),
                opts,
                providers=["CPUExecutionProvider"],
            )
            self.bhuman_walk = ort.InferenceSession(
                str(POLICIES / "bhuman/Config/NeuralNets/Walk/K1_walk.onnx"),
                opts,
                providers=["CPUExecutionProvider"],
            )
            self.feet = [
                model.body(prefix + side + "_ankle_roll_link").id
                for side in ["left", "right"]
            ]
            self.ball_v = int(model.joint("ball_free").dofadr[0])
            self.offset = np.array([-0.2, 0, 0, 0.4, -0.25, 0] * 2)
            self.kick_kp = self.kp.copy()
            self.kick_kp[10:] = [80, 80, 80, 80, 50, 25] * 2
            self.kick_kd = self.kd.copy()
            self.kick_kd[10:] = [4, 4, 4, 4, 2, 2] * 2
            self.kp = self.kick_kp
            self.kd = self.kick_kd
        self.reset()

    def reset(self):
        self.last_action = np.zeros(22)
        self.phase = np.array([0.0, np.pi])
        self.target = self.default.copy()
        self.previous_skill = "walk"
        self.blend_from = self.target.copy()
        self.blend_time = 1.0
        self.kick_frequency = 0.0
        self.old_ball = np.zeros(2)
        self.walk_history = np.zeros((10, 32))
        self.history_ready = False

    def act(self, data, skill, command, ball_world):
        if skill == "kick" and self.kick_kind.startswith("bhuman"):
            return self.act_bhuman(data, command, ball_world)
        if skill == "walk" and self.kick_kind.startswith("bhuman"):
            return self.act_bhuman_walk(data, command)
        R = data.xmat[self.body].reshape(3, 3)
        gyro = data.qvel[self.root_v + 3 : self.root_v + 6]
        gravity = -R[2, :]
        q = data.qpos[self.q] - self.default
        qd = data.qvel[self.v].copy()
        if skill == "walk":
            q = q.copy()
            q[:2] = 0
            qd[:2] = 0
            ph = (
                self.phase
                if np.linalg.norm(command) >= 0.01
                else np.array([np.pi, np.pi])
            )
            obs = np.r_[
                gyro, gravity, command, q, qd, self.last_action, np.cos(ph), np.sin(ph)
            ]
        else:
            ball = R.T @ (np.asarray(ball_world) - data.xpos[self.body])
            obs = np.r_[
                gyro,
                gravity,
                np.asarray(command)[:2],
                ball[:2],
                np.cos(self.phase),
                np.sin(self.phase),
                q,
                qd,
                self.last_action,
            ]
        net = self.nets[skill]
        expected = 79 if skill == "walk" else 80
        if obs.shape != (expected,):
            raise ValueError(f"{skill} observation {obs.shape}, expected {expected}")
        action = net.run(
            None, {net.get_inputs()[0].name: obs.astype(np.float32)[None]}
        )[0].reshape(-1)
        if action.shape != (22,) or not np.isfinite(action).all():
            raise FloatingPointError("Invalid upstream output")
        self.last_action = action.copy()
        target = np.clip(self.default + self.scale * action, self.lower, self.upper)
        if skill == "walk":
            target[:2] = self.default[:2]
        if skill != self.previous_skill:
            self.blend_from = self.target.copy()
            self.blend_time = 0
        self.blend_time = min(0.3, self.blend_time + 0.02)
        alpha = self.blend_time / 0.3
        self.target = (1 - alpha) * self.blend_from + alpha * target
        self.previous_skill = skill
        self.phase = (self.phase + 2 * np.pi * 0.02 * 1.5 + np.pi) % (2 * np.pi) - np.pi
        return action

    def act_bhuman(self, data, command, ball_world):
        R = data.xmat[self.body].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        cs, sn = np.cos(yaw), np.sin(yaw)
        rotate = np.array([[cs, sn], [-sn, cs]])
        origin = data.xpos[self.feet].mean(0)
        ball = rotate @ (np.asarray(ball_world)[:2] - origin[:2])
        ball *= min(1.0, 2.0 / max(np.linalg.norm(ball), 1e-8))
        if self.previous_skill != "kick" or np.linalg.norm(ball - self.old_ball) > 0.2:
            self.old_ball = ball.copy()
        direction = np.arctan2(command[1], command[0])
        gyro = data.qvel[self.root_v + 3 : self.root_v + 6]
        phase = self.phase[0]
        velocity = rotate @ data.qvel[self.ball_v : self.ball_v + 2]
        velocity *= min(1.0, 2.0 / max(np.linalg.norm(velocity), 1e-8))
        obs = np.r_[
            -R[2, :],
            gyro,
            ball * 0.1,
            direction / np.pi,
            np.cos(phase),
            np.sin(phase),
            data.qpos[self.q[10:]] - self.offset,
            data.qvel[self.v[10:]] * 0.1,
            self.target[10:] - self.offset,
            self.kick_frequency,
            [0, 0, 0],
            velocity * 0.1,
            0.0,
            np.sin(direction),
            np.cos(direction),
            self.kick_speed * 0.1,
            self.old_ball * 0.1,
        ].astype(np.float32)
        if obs.shape != (59,):
            raise ValueError(obs.shape)
        net = self.bhuman_net
        output = net.run(None, {net.get_inputs()[0].name: obs[None]})[0].reshape(-1)
        if output.shape != (13,) or not np.isfinite(output).all():
            raise FloatingPointError("Invalid B-Human output")
        target = self.default.copy()
        target[10:] = np.clip(output[:12], -2, 2) + self.offset
        target = np.clip(target, self.lower, self.upper)
        if self.previous_skill != "kick":
            self.blend_from = self.target.copy()
            self.blend_time = 0.0
        self.blend_time = min(0.3, self.blend_time + 0.02)
        alpha = self.blend_time / 0.3
        self.target = (1 - alpha) * self.blend_from + alpha * target
        self.kick_frequency = np.clip(output[-1], -0.5, 0.5)
        self.phase = (
            self.phase + 2 * np.pi * 0.02 * (1.5 + self.kick_frequency) + np.pi
        ) % (2 * np.pi) - np.pi
        self.last_action[:] = 0.0
        self.last_action[10:] = output[:12]
        self.old_ball = ball.copy()
        self.previous_skill = "kick"
        return self.last_action.copy()

    def act_bhuman_walk(self, data, command):
        R = data.xmat[self.body].reshape(3, 3)
        frame = np.r_[
            -R[2, :],
            data.qvel[self.root_v + 3 : self.root_v + 6],
            data.qpos[self.q[10:]] - self.offset,
            self.target[10:] - self.offset,
            0,
            0,
        ]
        if not self.history_ready or self.previous_skill == "kick":
            self.walk_history[:] = frame
            self.history_ready = True
        self.walk_history[:-1] = self.walk_history[1:]
        self.walk_history[-1] = frame
        moving = np.linalg.norm(command) >= 0.01
        ph = [np.cos(self.phase[0]), np.sin(self.phase[0])] if moving else [0.0, 0.0]
        obs = np.r_[
            self.walk_history.ravel(),
            command,
            ph,
            data.qvel[self.v[10:]] * 0.1,
            self.kick_frequency,
            np.zeros(6),
        ].astype(np.float32)
        if obs.shape != (344,):
            raise ValueError(obs.shape)
        net = self.bhuman_walk
        output = net.run(None, {net.get_inputs()[0].name: obs[None]})[0].reshape(-1)
        target = self.default.copy()
        target[10:] = np.clip(output[:12], -2, 2) + self.offset
        self.target = np.clip(target, self.lower, self.upper)
        self.kick_frequency = np.clip(output[-1], -0.5, 0.5)
        if moving:
            self.phase = (
                self.phase + 2 * np.pi * 0.02 * (1.5 + self.kick_frequency) + np.pi
            ) % (2 * np.pi) - np.pi
        self.last_action[:] = 0.0
        self.last_action[10:] = output[:12]
        self.previous_skill = "walk"
        return self.last_action.copy()

    def torque(self, data):
        kp, kd = (
            (self.kick_kp, self.kick_kd)
            if self.kick_kind.startswith("bhuman")
            else (self.kp, self.kd)
        )
        data.ctrl[self.a] = np.clip(
            kp * (self.target - data.qpos[self.q]) - kd * data.qvel[self.v],
            -self.limit,
            self.limit,
        )


def source_manifest(kick_kind="nubots"):
    if kick_kind.startswith("bhuman"):
        return {
            "networks": json.loads((POLICIES / "bhuman/SOURCE.json").read_text()),
            "serial_joint_and_arm_interface": json.loads(
                (POLICIES / "nubots/SOURCE.json").read_text()
            ),
        }
    return json.loads((POLICIES / "nubots/SOURCE.json").read_text())
