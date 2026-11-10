"""Upstream policy adapters. Commands select skills; pretrained networks control joints."""

import json
from pathlib import Path

import numpy as np

from .config import ROOT

POLICIES = ROOT / "policies"


class BoosterVelocityPolicy:
    """Evaluate the exact exported float32 MLP and observation contract from the web demo."""

    def __init__(self, model, robot_prefix="", asset_dir=None):
        self.path = Path(asset_dir or POLICIES / "booster_mjlab/website/static/demo")
        self.spec = json.loads((self.path / "scene.json").read_text())
        self.blob = np.fromfile(self.path / self.spec["policy"]["file"], dtype="<f4")
        self.layers = self.spec["policy"]["layers"]
        self.joints = self.spec["joints"]
        self.q = np.array(
            [model.joint(robot_prefix + j["name"]).qposadr[0] for j in self.joints]
        )
        self.v = np.array(
            [model.joint(robot_prefix + j["name"]).dofadr[0] for j in self.joints]
        )
        self.a = np.array(
            [model.actuator(robot_prefix + j["name"]).id for j in self.joints]
        )
        self.default = np.array([j["default"] for j in self.joints])
        self.scale = np.array([j["action_scale"] for j in self.joints])
        self.body = model.body(robot_prefix + "Trunk").id
        self.gyro = model.sensor(robot_prefix + "imu_ang_vel").adr[0]
        self.last_action = np.zeros(22, dtype=np.float32)
        self.control_dt = self.spec["control_dt"]
        self.command = np.zeros(3)
        self.last_obs = None

    def reset(self):
        self.last_action[:] = 0

    def network(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for l in self.layers:
            op = l["op"]
            if op in ("sub", "div", "add", "mul"):
                c = self.blob[l["offset"] : l["offset"] + len(x)]
                a, b = (x, c) if l["tensor_first"] else (c, x)
                x = {
                    "sub": np.subtract,
                    "div": np.divide,
                    "add": np.add,
                    "mul": np.multiply,
                }[op](a, b)
            elif op == "gemm":
                w = self.blob[
                    l["w_offset"] : l["w_offset"] + l["out"] * l["in"]
                ].reshape(l["out"], l["in"])
                b = self.blob[l["b_offset"] : l["b_offset"] + l["out"]]
                x = w @ x + b
            elif op == "elu":
                x = x.copy()
                negative = x < 0
                x[negative] = l["alpha"] * np.expm1(x[negative])
            elif op == "relu":
                x = np.maximum(x, 0)
            elif op == "tanh":
                x = np.tanh(x)
            elif op == "sigmoid":
                x = 1 / (1 + np.exp(-x))
            else:
                raise ValueError(f"Unsupported upstream network op {op}")
        return x

    def act(self, data, command=None):
        if command is not None:
            self.command[:] = command
        R = data.xmat[self.body].reshape(3, 3)
        gravity = -R[2, :]
        obs = np.concatenate(
            [
                data.sensordata[self.gyro : self.gyro + 3],
                gravity,
                data.qpos[self.q] - self.default,
                data.qvel[self.v],
                self.last_action,
                self.command,
            ]
        ).astype(np.float32)
        if obs.shape != (75,):
            raise ValueError("Booster observation contract mismatch")
        action = self.network(obs)
        if action.shape != (22,) or not np.isfinite(action).all():
            raise ValueError("Invalid policy output")
        self.last_action = action.copy()
        self.last_obs = obs
        # Native affine actuators implement upstream PD (not a second external PD loop).
        data.ctrl[self.a] = self.default + action * self.scale
        return action


def sonic_module():
    import importlib.util

    path = POLICIES / "sonic_k1/gear_sonic/scripts/deploy_k1_booster.py"
    spec = importlib.util.spec_from_file_location("soccer_sonic_deploy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SonicMotionPolicy:
    """Reuse upstream ONNX observation/history/reference code; bind named joints in the combined field."""

    def __init__(self, model, prefix, motion, shared=None):
        import copy

        self.up = sonic_module() if shared is None else shared.up
        ck = POLICIES / "sonic_k1/checkpoints/sonic-k1"
        self.plant = (
            self.up.Plant(str(ck / "k1_plant.json")) if shared is None else shared.plant
        )
        if shared is None:
            self.policy = self.up.Policy(str(ck / "onnx/policy_g1.onnx"), self.plant)
        else:
            self.policy = copy.copy(shared.policy)
            self.policy.reset()
        self.ref = self.up.Reference(str(motion), self.plant)
        z = np.load(motion, allow_pickle=False)
        self.initial_qpos = z["qpos"][0].astype(float)
        self.initial_velocity = (z["qpos"][1, :3] - z["qpos"][0, :3]) * float(
            np.ravel(z["fps"])[0]
        )
        self.q = np.array([model.joint(prefix + j).qposadr[0] for j in self.plant.mj])
        self.v = np.array([model.joint(prefix + j).dofadr[0] for j in self.plant.mj])
        self.a = np.array([model.actuator(prefix + j).id for j in self.plant.mj])
        self.body = model.body(prefix + "trunk").id
        root = model.joint(prefix + "world_joint")
        self.rootq = int(root.qposadr[0])
        self.rootv = int(root.dofadr[0])
        self.control_dt = 0.02
        self.target = self.initial_qpos[7:].copy()
        self.kp = self.plant.kp[self.plant.i2m]
        self.kd = self.plant.kd[self.plant.i2m]
        self.effort = self.plant.effort[self.plant.i2m]
        model.dof_armature[self.v] = self.plant.armature[self.plant.i2m]
        # Upstream nominal evaluation explicitly clips motor torque to the released plant limits.
        model.actuator_ctrllimited[self.a] = 0
        model.actuator_forcelimited[self.a] = 0
        model.jnt_actfrclimited[model.actuator_trnid[self.a, 0]] = 0
        self.last_action = np.zeros(22)
        self.action_scale = 1.0

    def act(self, data, k):
        rq = data.qpos[self.rootq + 3 : self.rootq + 7]
        prop = self.policy.proprio(
            data.qvel[self.rootv + 3 : self.rootv + 6],
            data.qpos[self.q][self.plant.m2i],
            data.qvel[self.v][self.plant.m2i],
            self.up.quat_to_mat(rq).T @ np.array([0, 0, -1]),
        )
        cmd, ori = self.ref.command(k, rq)
        target, action = self.policy.act(prop, cmd, ori)
        default = self.plant.default[self.plant.i2m]
        self.target = default + (target[self.plant.i2m] - default) * self.action_scale
        self.last_action = action
        return action

    def torque(self, data):
        data.ctrl[self.a] = np.clip(
            self.kp * (self.target - data.qpos[self.q]) - self.kd * data.qvel[self.v],
            -self.effort,
            self.effort,
        )
