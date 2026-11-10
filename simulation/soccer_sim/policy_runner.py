"""Test pretrained locomotion/motion policies on the same six-robot HSL field."""

import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .config import DEFAULT_CONFIG, load_config
from .policy_scene import MODEL_PATHS, build_policy_scene
from .pretrained import POLICIES, BoosterVelocityPolicy, SonicMotionPolicy


def run(
    kind,
    config,
    out,
    *,
    scenario="stand",
    seconds=8.0,
    command=None,
    motion=None,
    video=False,
    native_contacts=False,
):
    if kind == "sonic_k1" and scenario not in ("motion", "ball_contact"):
        raise ValueError(
            "SONIC follows a reference motion; choose motion or ball_contact, not a velocity command"
        )
    if kind == "booster_mjlab" and scenario == "motion":
        raise ValueError(
            "The released Booster policy follows velocity commands, not reference clips"
        )
    c = deepcopy(config)
    c["robot"]["base_mode"] = "free"
    pc = c["pretrained"]
    command = list(command if command is not None else pc["command"])
    if len(command) != 3 or not np.isfinite(command).all():
        raise ValueError("command needs three finite values")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    pc["command"] = command
    roles = [r["name"] for r in c["robot"]["instances"]]
    sender = roles.index(c["passing"]["sender"])
    receiver = roles.index(c["passing"]["receiver"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # mj_saveLastXML uses MuJoCo's last parsed XML, so load the source before the field.
    source = mujoco.MjModel.from_xml_path(str(MODEL_PATHS[kind]))
    xml = build_policy_scene(c, kind, native_contacts=native_contacts)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    policies = []
    roots = []
    initial = []
    if kind == "booster_mjlab":
        mjd = mujoco.MjData(source)
        mujoco.mj_resetDataKeyframe(source, mjd, 0)
        initial_pose = mjd.qpos.copy()
    else:
        motion = Path(motion or POLICIES / "sonic_k1/examples/k1/motions/squat.npz")
    for spec in c["robot"]["instances"]:
        prefix = spec["name"] + "/"
        if kind == "booster_mjlab":
            policy = BoosterVelocityPolicy(m, prefix)
            free = m.joint(prefix + "floating_base_joint")
            rootq = int(free.qposadr[0])
            rootv = int(free.dofadr[0])
            pose = initial_pose.copy()
        else:
            policy = SonicMotionPolicy(
                m, prefix, motion, shared=policies[0] if policies else None
            )
            rootq, rootv = policy.rootq, policy.rootv
            pose = policy.initial_qpos.copy()
            d.qvel[policy.v] = policy.ref.jv[0][policy.plant.i2m]
            angle = spec["yaw"]
            R = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            d.qvel[rootv : rootv + 3] = R @ policy.initial_velocity
        pose[:2] = spec["position"]
        angle = spec["yaw"]
        qz = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        if kind == "booster_mjlab":
            pose[3:7] = qz
        else:
            pose[3:7] = policy.up.quat_mul(qz, pose[3:7])
        d.qpos[rootq : rootq + 7] = pose[:7]
        d.qpos[policy.q] = pose[7:]
        if kind == "booster_mjlab":
            d.ctrl[policy.a] = policy.default
            policy.scale *= pc["action_scale"]
            m.actuator_gainprm[policy.a, 0] *= pc["kp_scale"]
            m.actuator_biasprm[policy.a, 1] *= pc["kp_scale"]
            m.actuator_biasprm[policy.a, 2] *= pc["kd_scale"]
            m.actuator_forcerange[policy.a] *= c["robot"]["torque_scale"]
            m.jnt_actfrcrange[m.actuator_trnid[policy.a, 0]] *= c["robot"][
                "torque_scale"
            ]
        else:
            policy.kp *= pc["kp_scale"]
            policy.kd *= pc["kd_scale"]
            policy.effort *= c["robot"]["torque_scale"]
            policy.action_scale = pc["action_scale"]
        policies.append(policy)
        roots.append((rootq, rootv))
        initial.append(pose[:3].copy())
    mujoco.mj_forward(m, d)
    if kind == "sonic_k1":
        for p, (q, v) in zip(policies, roots):
            p.ref.align_heading(p.up.yaw_of(d.qpos[q + 3 : q + 7]))
        seconds = min(seconds, (policies[0].ref.n - 1) * 0.02)
    ballq = int(m.joint("ball_free").qposadr[0])
    ballv = int(m.joint("ball_free").dofadr[0])
    ballbody = m.body("ball").id
    ballgeom = m.geom("ball_geom").id
    if scenario not in ("ball_contact", "pass"):
        # Keep the ball away from the controlled path for locomotion qualification.
        d.qpos[ballq : ballq + 3] = [0, -3, c["ball"]["radius"] + 0.001]
        mujoco.mj_forward(m, d)
    c["robot"]["base_height"] = float(initial[0][2])
    np.savez(out / "initial_state.npz", qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl)
    # Save runtime gain/armature/limit edits and an explicit initial-state keyframe.
    mujoco.mj_saveLastXML(str(out / "scene.xml"), m)
    saved = ET.parse(out / "scene.xml")
    keys = saved.getroot().find("keyframe")
    if keys is None:
        keys = ET.SubElement(saved.getroot(), "keyframe")
    ET.SubElement(
        keys,
        "key",
        name="policy_initial",
        qpos=" ".join(map(str, d.qpos)),
        qvel=" ".join(map(str, d.qvel)),
        ctrl=" ".join(map(str, d.ctrl)),
    )
    saved.write(out / "scene.xml", encoding="unicode")
    parameters = []
    for spec, p in zip(c["robot"]["instances"], policies):
        prefix = spec["name"] + "/"
        parameters.append(
            {
                "name": spec["name"],
                "links": [
                    {
                        "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                        "mass_kg": float(m.body_mass[b]),
                        "diaginertia_kg_m2": m.body_inertia[b].tolist(),
                    }
                    for b in range(m.nbody)
                    if (
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                    ).startswith(prefix)
                ],
                "joints": [
                    {
                        "name": mujoco.mj_id2name(
                            m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])
                        ),
                        "kp": float(m.actuator_gainprm[a, 0])
                        if kind == "booster_mjlab"
                        else float(p.kp[j]),
                        "kd": float(-m.actuator_biasprm[a, 2])
                        if kind == "booster_mjlab"
                        else float(p.kd[j]),
                        "action_scale_rad": float(p.scale[j])
                        if kind == "booster_mjlab"
                        else float(p.plant.scale[p.plant.i2m][j] * p.action_scale),
                        "torque_limit_Nm": float(max(abs(m.actuator_forcerange[a])))
                        if kind == "booster_mjlab"
                        else float(p.effort[j]),
                        "range_rad": m.jnt_range[m.actuator_trnid[a, 0]].tolist(),
                        "armature_kg_m2": float(m.dof_armature[v]),
                        "damping_Nm_s_per_rad": float(m.dof_damping[v]),
                    }
                    for j, (a, v) in enumerate(zip(p.a, p.v))
                ],
            }
        )
    (out / "robot_parameters.json").write_text(json.dumps(parameters, indent=2))
    dt = m.opt.timestep
    decimation = round(0.02 / dt)
    if abs(decimation * dt - 0.02) > 1e-10:
        raise ValueError("Physics timestep must divide policy period 0.02s")
    minheight = np.full(len(policies), 10.0)
    minupright = np.ones(len(policies))
    fell = np.zeros(len(policies), dtype=bool)
    tilt_duration = np.zeros(len(policies))
    contacted = set()
    contact_first_times = {}
    peak_action = 0.0
    inference_times = []
    trace = []
    initial_ball = d.qpos[ballq : ballq + 3].copy()
    sender_stopped = False
    receive_hold = 0.0
    pass_success = False
    pass_success_time = None
    renderer = writer = camera = None
    render_options = mujoco.MjvOption()
    if kind == "booster_mjlab":
        # This web-export MJCF renders with external GLB; its own shapes are in group 3.
        render_options.geomgroup[3] = 1
    render_options.sitegroup[:] = 0
    if video:
        import imageio.v2 as imageio

        renderer = mujoco.Renderer(m, 720, 1280)
        writer = imageio.get_writer(str(out / "policy.mp4"), fps=30, macro_block_size=1)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, camera)
        camera.distance = 11
        camera.azimuth = 95
        camera.elevation = -38
        camera.lookat[:] = [0, 0, 0.2]
    next_frame = 0.0
    start = time.monotonic()
    try:
        for i in range(round(seconds / dt)):
            if i % decimation == 0:
                t0 = time.monotonic()
                for j, p in enumerate(policies):
                    if kind == "booster_mjlab":
                        if (
                            scenario == "pass"
                            and np.linalg.norm(
                                d.qpos[ballq : ballq + 2] - initial_ball[:2]
                            )
                            > pc["release_distance"]
                        ):
                            sender_stopped = True
                        a = p.act(
                            d,
                            command
                            if j == sender
                            and scenario != "stand"
                            and not sender_stopped
                            else (0, 0, 0),
                        )
                    else:
                        a = p.act(d, i // decimation)
                    peak_action = max(peak_action, float(np.max(np.abs(a))))
                inference_times.append(time.monotonic() - t0)
            if kind == "sonic_k1":
                for p in policies:
                    p.torque(d)
            vel = d.qvel[ballv : ballv + 3]
            d.xfrc_applied[ballbody, :3] = (
                -0.5
                * c["physics"]["air_density"]
                * c["physics"]["air_drag_coefficient"]
                * math.pi
                * c["ball"]["radius"] ** 2
                * np.linalg.norm(vel)
                * vel
            )
            mujoco.mj_step(m, d)
            for j, (p, (q, v)) in enumerate(zip(policies, roots)):
                h = d.qpos[q + 2]
                upright = d.xmat[p.body].reshape(3, 3)[2, 2]
                minheight[j] = min(minheight[j], h)
                minupright[j] = min(minupright[j], upright)
                tilt_duration[j] = tilt_duration[j] + dt if upright < 0.5 else 0
                fell[j] |= h < 0.25 or tilt_duration[j] > 0.4
            for contact_index, contact in enumerate(d.contact):
                if ballgeom in (contact.geom1, contact.geom2):
                    other = (
                        contact.geom2 if contact.geom1 == ballgeom else contact.geom1
                    )
                    name = (
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(other)) or ""
                    )
                    if "/" in name:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(m, d, contact_index, force)
                        if force[0] > 0.01:
                            name = name.split("/")[0]
                            contacted.add(name)
                            contact_first_times.setdefault(name, float(d.time))
            if scenario == "pass":
                dist = np.linalg.norm(
                    d.qpos[ballq : ballq + 2]
                    - d.qpos[roots[receiver][0] : roots[receiver][0] + 2]
                )
                good = (
                    c["passing"]["sender"] in contacted
                    and c["passing"]["receiver"] in contacted
                    and contact_first_times[c["passing"]["sender"]]
                    < contact_first_times[c["passing"]["receiver"]]
                    and np.linalg.norm(d.qpos[ballq : ballq + 2] - initial_ball[:2])
                    > 0.5
                    and dist < c["passing"]["receive_radius"]
                    and np.linalg.norm(d.qvel[ballv : ballv + 3])
                    < c["passing"]["receive_speed"]
                    and not any(fell)
                )
                receive_hold = receive_hold + dt if good else 0
                if receive_hold >= c["passing"]["receive_hold"] and not pass_success:
                    pass_success = True
                    pass_success_time = float(d.time)
            if i % decimation == 0:
                trace.append(
                    [
                        d.time,
                        *d.qpos[ballq : ballq + 3],
                        *[float(d.qpos[q + k]) for q, v in roots for k in range(3)],
                        *minupright,
                    ]
                )
            if renderer and d.time + 1e-9 >= next_frame:
                from PIL import Image, ImageDraw

                next_frame += 1 / 30
                renderer.update_scene(d, camera=camera, scene_option=render_options)
                im = Image.fromarray(renderer.render())
                dr = ImageDraw.Draw(im)
                dr.rectangle((12, 12, 1050, 75), fill=(12, 23, 34))
                dr.text(
                    (24, 20),
                    f"{kind} pretrained | 6 free-base K1s | {scenario} | t={d.time:.2f}s | fallen={int(sum(fell))}",
                    fill="white",
                )
                dr.text(
                    (24, 44),
                    f"50 Hz closed-loop policy / {1 / dt:.0f} Hz physics | uncalibrated turf | no soccer strategy in pretrained weights",
                    fill=(245, 198, 95),
                )
                writer.append_data(np.asarray(im))
                if i < decimation:
                    im.save(out / "preview.png")
        if not np.isfinite(d.qpos).all():
            raise FloatingPointError("Non-finite state")
    finally:
        if writer:
            writer.close()
            renderer.close()
    robots = []
    for spec, qv, h, u, f, ini in zip(
        c["robot"]["instances"], roots, minheight, minupright, fell, initial
    ):
        q, _v = qv
        robots.append(
            {
                "name": spec["name"],
                "fell": bool(f),
                "minimum_root_height_m": float(h),
                "minimum_upright_cosine": float(u),
                "displacement_xy_m": (d.qpos[q : q + 2] - ini[:2]).tolist(),
                "final_xyz_m": d.qpos[q : q + 3].tolist(),
            }
        )
    result = {
        "policy": kind,
        "scenario": scenario,
        "simulated_seconds": float(d.time),
        "wall_seconds": time.monotonic() - start,
        "all_robots_no_fall": not bool(any(fell)),
        "robots": robots,
        "ball_contact_robots": sorted(contacted),
        "ball_first_contact_time_s": contact_first_times,
        "final_ball_position_m": d.qpos[ballq : ballq + 3].tolist(),
        "final_ball_speed_mps": float(np.linalg.norm(d.qvel[ballv : ballv + 3])),
        "ball_displacement_m": float(
            np.linalg.norm(d.qpos[ballq : ballq + 3] - initial_ball)
        ),
        "peak_action_abs": peak_action,
        "mean_six_robot_inference_ms": float(np.mean(inference_times) * 1000),
        "p95_six_robot_inference_ms": float(np.percentile(inference_times, 95) * 1000),
        "mujoco_warning_count": int(sum(w.number for w in d.warning)),
        "mujoco_version": mujoco.__version__,
        "native_ground_contacts": native_contacts,
        "command_body_frame": list(command),
        "motion": str(motion) if motion else None,
        "is_soccer_decision_policy": False,
        "is_pretrained_pass_policy": False,
        "base_mode": "free",
        "receive_event_success": pass_success,
        "receive_event_time_s": pass_success_time,
        "stable_receive_at_end": bool(
            scenario == "pass" and receive_hold >= c["passing"]["receive_hold"]
        ),
        "pass_success": bool(
            pass_success
            and receive_hold >= c["passing"]["receive_hold"]
            and not any(fell)
        ),
        "sender_stopped": sender_stopped,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    (out / "source.json").write_text((POLICIES / kind / "SOURCE.json").read_text())
    provenance = {
        "rules": c["rules"],
        "policy_source": json.loads((POLICIES / kind / "SOURCE.json").read_text()),
        "physics_calibrated": False,
        "base_mode": "free",
        "control_dt_s": 0.02,
        "initialization": "upstream keyframe"
        if kind == "booster_mjlab"
        else "first frame of reference clip",
        "mujoco_version": mujoco.__version__,
    }
    if kind == "sonic_k1":
        provenance["weights"] = json.loads(
            (POLICIES / kind / "checkpoints/sonic-k1/WEIGHTS_SOURCE.json").read_text()
        )
        provenance["motion_sha256"] = hashlib.sha256(motion.read_bytes()).hexdigest()
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    np.savetxt(
        out / "trajectory.csv",
        np.array(trace),
        delimiter=",",
        header="time,ball_x,ball_y,ball_z,"
        + ",".join(
            f"{r['name']}_{k}" for r in c["robot"]["instances"] for k in ["x", "y", "z"]
        )
        + ","
        + ",".join(r["name"] + "_min_upright" for r in c["robot"]["instances"]),
        comments="",
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["booster_mjlab", "sonic_k1"], required=True)
    p.add_argument(
        "--scenario",
        choices=["stand", "walk", "ball_contact", "motion", "pass"],
        default="stand",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument(
        "--command",
        type=float,
        nargs=3,
        help="override pretrained.command: vx vy yaw_rate",
    )
    p.add_argument("--motion", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", action="store_true")
    p.add_argument("--native-contacts", action="store_true")
    a = p.parse_args()
    run(
        a.kind,
        load_config(a.config, a.set),
        a.out,
        scenario=a.scenario,
        seconds=a.seconds,
        command=a.command,
        motion=a.motion,
        video=a.video,
        native_contacts=a.native_contacts,
    )


if __name__ == "__main__":
    main()
