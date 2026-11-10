"""Two native Joga G1s in ONE MuJoCo world; ball is written only during reset.

Unlike the website teammate demo, no ball handoff or post-kick velocity correction.
"""

import json
import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco
import numpy as np

from .joga import ASSETS, JogaSkills


def pair_model(overrides=None):
    overrides = overrides or {}
    root = ET.parse(ASSETS / "scene.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSETS / "meshes"))
    root.find("compiler").set("texturedir", str(ASSETS / "meshes"))
    for el in root.iter():
        el.attrib.pop("colorspace", None)  # texture metadata newer than MuJoCo 3.3
    for name in ["keyframe", "sensor", "size"]:
        element = root.find(name)
        if element is not None:
            root.remove(element)
    world = root.find("worldbody")
    robot = world.find("body[@name='pelvis']")
    world.remove(robot)
    actuator = root.find("actuator")
    original_act = list(actuator)
    for el in original_act:
        actuator.remove(el)
    contacts = root.find("contact")
    original_contact = list(contacts)
    for el in original_contact:
        contacts.remove(el)
    for prefix in ["A1/", "A2/"]:
        copy = deepcopy(robot)
        for el in copy.iter():
            if "name" in el.attrib:
                el.set("name", prefix + el.get("name"))
        world.append(copy)
        for el in original_act:
            a = deepcopy(el)
            for key in ["name", "joint"]:
                a.set(key, prefix + a.get(key))
            actuator.append(a)
        for el in original_contact:
            c = deepcopy(el)
            for key in ["body1", "body2", "geom1", "geom2"]:
                if c.get(key) is not None and c.get(key) != "soccerball_geom":
                    c.set(key, prefix + c.get(key))
            contacts.append(c)
    # Visual-only HSL-sized field markings; native G1 floor/ball physics retained.
    for x in [-7.0, 7.0]:
        ET.SubElement(
            world,
            "geom",
            type="box",
            pos=f"{x} 0 .002",
            size=".025 4.5 .001",
            rgba="1 1 1 1",
            contype="0",
            conaffinity="0",
        )
    for y in [-4.5, 4.5]:
        ET.SubElement(
            world,
            "geom",
            type="box",
            pos=f"0 {y} .002",
            size="7 .025 .001",
            rgba="1 1 1 1",
            contype="0",
            conaffinity="0",
        )
    # Apply geometry/inertia changes before compilation so collision bounds,
    # inverse weights and other derived constants use the modified values.
    for kind in ["geom", "body", "joint"]:
        for name, values in overrides.get(kind, {}).items():
            obj = root.find(f".//{kind}[@name='{name}']")
            if obj is None:
                raise ValueError(f"Unknown {kind}: {name}")
            for key, value in values.items():
                target, attr = obj, key
                if kind == "body" and key in ["mass", "inertia", "ipos", "iquat"]:
                    target = obj.find("inertial")
                    if target is None:
                        raise ValueError(f"No explicit inertial element: {name}")
                    attr = {
                        "inertia": "diaginertia",
                        "ipos": "pos",
                        "iquat": "quat",
                    }.get(key, key)
                target.set(attr, " ".join(str(v) for v in np.asarray(value).ravel()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.tolerance = 1e-8
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name
        if body.startswith(("A1/", "A2/")):
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 3
        name = model.geom(g).name or ""
        if name.endswith("_collision") and ("foot" in name or "sole" in name):
            model.geom_friction[g, 0] = (
                overrides.get("geom", {}).get(name, {}).get("friction", [1.0])[0]
            )
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            model.dof_armature[model.jnt_dofadr[j]] = (
                overrides.get("joint", {})
                .get(model.joint(j).name, {})
                .get("armature", 0.01)
            )
    # Optional overrides preserve the downloaded native profile by default.
    # Explicit foot/ball pairs take precedence over geom contact parameters.
    for key, value in overrides.get("option", {}).items():
        current = getattr(model.opt, key)
        if isinstance(current, np.ndarray):
            current[:] = value
        else:
            setattr(model.opt, key, value)
    for key, value in overrides.get("foot_ball_pairs", {}).items():
        if key not in ["friction", "solref", "solimp", "margin", "gap"]:
            raise ValueError(f"Unsupported contact pair override: {key}")
        getattr(model, f"pair_{key}")[:] = value
    return model


class JogaPassEnv:
    action_dim = 20

    def __init__(self, settings):
        self.settings = settings
        self.spec = json.loads((ASSETS / "skills.json").read_text())
        self.m = pair_model(settings.get("physics_overrides"))
        self.substeps = round(settings["control_dt"] / self.m.opt.timestep)
        if self.substeps < 1 or not np.isclose(
            self.substeps * self.m.opt.timestep, settings["control_dt"]
        ):
            raise ValueError("control_dt must be an integer multiple of timestep")
        self.d = mujoco.MjData(self.m)
        self.skills = [
            JogaSkills(self.m, "A1/", ["locomotion", "passing"]),
            JogaSkills(self.m, "A2/", ["receiving"]),
        ]
        for p in self.skills:
            for name in ["kp", "kd", "limit", "maxvel"]:
                getattr(p, name)[:] *= settings.get("actuation_scale", {}).get(
                    name, 1.0
                )
        self.root_q = [p.root_q for p in self.skills]
        self.link_ids = [
            [i for i in range(self.m.nbody) if self.m.body(i).name.startswith(prefix)]
            for prefix in ["A1/", "A2/"]
        ]
        self.ball_body = self.m.body("soccerball").id
        self.ball_q = int(self.m.joint("soccerball_joint").qposadr[0])
        self.ball_v = int(self.m.joint("soccerball_joint").dofadr[0])
        self.radius = float(self.m.geom_size[self.m.geom("soccerball_geom").id, 0])
        floor = self.m.geom("floor").id
        self.physics_context = [
            float(self.m.geom_friction[floor, 0]),
            float(self.m.body_mass[self.ball_body]),
            self.radius,
        ]
        self.env = self
        self.config = {
            "embodiment": "Unitree G1 with 2-DoF neck (31 joints)",
            "ball": {"radius": self.radius, "mass": self.physics_context[1]},
            "physics": {"timestep": self.m.opt.timestep},
            "physics_overrides": settings.get("physics_overrides", {}),
            "actuation_scale": settings.get("actuation_scale", {}),
            "source": str(ASSETS / "scene.xml"),
        }
        self.layout = {}
        self.size = 0
        self.reset(0)

    def state(self):
        pieces = []

        def add(name, x):
            a = np.asarray(x).ravel()
            start = sum(len(v) for v in pieces)
            self.layout[name] = [start, start + len(a)]
            pieces.append(a)

        for i, p in enumerate(self.skills):
            add(f"r{i}_position", self.d.qpos[p.root_q : p.root_q + 3])
            add(f"r{i}_quaternion", self.d.qpos[p.root_q + 3 : p.root_q + 7])
            add(f"r{i}_velocity", self.d.qvel[p.root_v : p.root_v + 6])
            add(f"r{i}_q", self.d.qpos[p.q])
            add(f"r{i}_qd", self.d.qvel[p.v])
        add("ball_position", self.d.qpos[self.ball_q : self.ball_q + 3])
        add("ball_velocity", self.d.qvel[self.ball_v : self.ball_v + 6])
        add("time", [self.elapsed])
        for i, p in enumerate(self.skills):
            add(f"r{i}_controller", p.memory())
        self.size = sum(len(x) for x in pieces)
        return np.concatenate(pieces).astype(np.float32)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        self.elapsed = 0.0
        receiver = (
            np.array(self.settings["receiver_xy"])
            + rng.uniform(-1, 1, 2) * self.settings["receiver_xy_noise"]
        )
        sender = np.array(self.settings["sender_xy"])
        heading = np.arctan2(*(receiver - sender)[::-1])
        for i, (p, xy, yaw, skill) in enumerate(
            zip(
                self.skills,
                [sender, receiver],
                [heading, heading + np.pi],
                ["locomotion", "receiving"],
            )
        ):
            p.reset(skill)
            self.d.qpos[p.root_q : p.root_q + 3] = [*xy, self.spec["init_qpos"][2]]
            self.d.qpos[p.root_q + 3 : p.root_q + 7] = [
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            ]
            self.d.qpos[p.q] = self.spec["init_qpos"][7:38]
            self.d.qpos[p.q[p.posctrl]] = p.mods[skill]["default"][p.posctrl]
        self.d.qpos[self.ball_q : self.ball_q + 3] = [20, 20, self.radius]
        mujoco.mj_forward(self.m, self.d)
        for _ in range(
            round(self.settings["settle_time"] / self.settings["control_dt"])
        ):
            for p in self.skills:
                p.act(self.d, p.active, [0, 0, 0], self.ball_body)
            self._physics()
        local = (
            np.array(self.settings["ball_xy"])
            + rng.uniform(-1, 1, 2) * self.settings["ball_xy_noise"]
        )
        cs, sn = np.cos(heading), np.sin(heading)
        xy = (
            self.d.qpos[self.root_q[0] : self.root_q[0] + 2]
            + np.array([[cs, -sn], [sn, cs]]) @ local
        )
        self.d.qpos[self.ball_q : self.ball_q + 3] = [*xy, self.radius + 0.001]
        self.d.qvel[self.ball_v : self.ball_v + 6] = 0
        mujoco.mj_forward(self.m, self.d)
        self.initial_ball = xy
        self.capture_time = 0.0
        self.captured = False
        self.fallen = False
        self.first_contact_times = [None, None]
        self.max_ball_travel = 0.0
        self.min_distance = np.inf
        return self.state()

    def _physics(self):
        for _ in range(self.substeps):
            for p in self.skills:
                p.torque(self.d)
            mujoco.mj_step(self.m, self.d)

    def facing_valid(self):
        for i, p in enumerate(self.skills):
            delta = self.d.xpos[self.skills[1 - i].body, :2] - self.d.xpos[p.body, :2]
            f = self.d.xmat[p.body].reshape(3, 3)[:2, 0]
            if np.dot(f, delta) / max(
                np.linalg.norm(f) * np.linalg.norm(delta), 1e-8
            ) < np.cos(np.deg2rad(self.settings["facing_tolerance_deg"])):
                return False
        return True

    def step(self, action):
        a = np.clip(np.asarray(action), -1, 1)
        if a.shape != (20,) or not np.isfinite(a).all():
            raise ValueError("20 finite actions required")
        ball = self.d.xpos[self.ball_body]
        for i, p in enumerate(self.skills):
            if i == 0:
                kick = (
                    self.settings["kick_start"]
                    <= self.elapsed
                    < self.settings["kick_end"]
                )
                delta = self.d.xpos[self.skills[1].body, :2] - ball[:2]
                angle = (
                    np.arctan2(delta[1], delta[0])
                    - p.yaw(self.d)
                    + np.deg2rad(self.settings.get("pass_direction_bias_deg", 17))
                    + self.settings["kick_heading_scale"] * a[6]
                )
                command = (
                    np.r_[
                        self.settings["kick_speed"]
                        * np.array([np.cos(angle), np.sin(angle)]),
                        0,
                    ]
                    if kick
                    else np.zeros(3)
                )
                p.act(
                    self.d, "passing" if kick else "locomotion", command, self.ball_body
                )
            else:
                p.act(
                    self.d,
                    "receiving",
                    [0, 0, 0],
                    self.ball_body,
                    np.array(self.settings["receiver_joint_residual_scale"]) * a[8:],
                )
        for _ in range(self.substeps):
            for p in self.skills:
                p.torque(self.d)
            mujoco.mj_step(self.m, self.d)
            self.elapsed += self.m.opt.timestep
            for contact_index, con in enumerate(self.d.contact):
                ids = [
                    int(self.m.geom_bodyid[con.geom1]),
                    int(self.m.geom_bodyid[con.geom2]),
                ]
                if self.ball_body not in ids:
                    continue
                other = ids[1] if ids[0] == self.ball_body else ids[0]
                name = self.m.body(other).name
                if name.startswith(("A1/", "A2/")):
                    idx = 0 if name.startswith("A1/") else 1
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.m, self.d, contact_index, force)
                    if force[0] > 0.1 and self.first_contact_times[idx] is None:
                        self.first_contact_times[idx] = self.elapsed
            ball = self.d.qpos[self.ball_q : self.ball_q + 3]
            dist = np.linalg.norm(ball[:2] - self.d.xpos[self.skills[1].body, :2])
            speed = np.linalg.norm(self.d.qvel[self.ball_v : self.ball_v + 2])
            self.min_distance = min(self.min_distance, dist)
            self.max_ball_travel = max(
                self.max_ball_travel, np.linalg.norm(ball[:2] - self.initial_ball)
            )
            valid = (
                self.first_contact_times[1] is not None
                and dist < self.settings["capture_radius"]
                and speed < self.settings["capture_speed"]
                and self.facing_valid()
            )
            self.capture_time = (
                self.capture_time + self.m.opt.timestep if valid else 0.0
            )
            self.captured |= self.capture_time >= self.settings["capture_hold"]
            for p in self.skills:
                self.fallen |= (
                    self.d.xpos[p.body, 2] < self.settings["fall_height"]
                    or self.d.xmat[p.body].reshape(3, 3)[2, 2]
                    < self.settings["fall_upright"]
                )
        return self.state(), self.elapsed >= self.settings[
            "duration"
        ] - 1e-8 or self.fallen

    def metrics(self):
        ball = self.d.qpos[self.ball_q : self.ball_q + 3]
        p = self.skills[1]
        rel = self.d.xmat[p.body].reshape(3, 3).T @ (ball - self.d.xpos[p.body])
        ordered = (
            all(t is not None for t in self.first_contact_times)
            and self.first_contact_times[0] < self.first_contact_times[1]
        )
        success = (
            ordered
            and self.max_ball_travel >= self.settings["minimum_pass_travel"]
            and self.capture_time >= self.settings["capture_hold"]
            and not self.fallen
        )
        front = 0.03 < rel[0] < 0.55 and abs(rel[1]) < 0.3
        return {
            "pass_success": bool(success),
            "front_pass_success": bool(success and front),
            "captured": bool(self.captured),
            "fallen": bool(self.fallen),
            "final_receiver_distance": float(np.linalg.norm(rel[:2])),
            "final_ball_speed": float(
                np.linalg.norm(self.d.qvel[self.ball_v : self.ball_v + 2])
            ),
            "first_contact_times": self.first_contact_times,
            "max_ball_travel": float(self.max_ball_travel),
            "facing_valid_at_end": self.facing_valid(),
            "ball_in_front_at_end": bool(front),
            "receiver_ball_local": rel.tolist(),
            "min_receiver_distance": float(self.min_distance),
        }
