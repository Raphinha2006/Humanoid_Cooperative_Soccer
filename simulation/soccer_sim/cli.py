"""Build, run, render and probe the parameterised environment."""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .config import DEFAULT_CONFIG, load_config
from .env import SoccerEnv
from .passing import PassingController


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def snapshot(env, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(env.xml)
    (out / "resolved_config.yaml").write_text(
        yaml.safe_dump(env.config, sort_keys=False, allow_unicode=True)
    )
    write_json(
        out / "provenance.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "robot_repository": "https://github.com/BoosterRobotics/booster_assets",
            "robot_revision": "12a97516f57469078707afb587a5f84c1cb110d1",
            "rules": env.config["rules"],
            "robot_base_mode": env.config["robot"]["base_mode"],
            "contact_parameters_calibrated": False,
        },
    )
    # A single robot's complete inertial and actuation parameters, without six copies.
    m = env.model
    prefix = env.config["robot"]["instances"][0]["name"] + "/"
    bodies = []
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name.startswith(prefix):
            bodies.append(
                {
                    "name": name[len(prefix) :],
                    "mass_kg": float(m.body_mass[i]),
                    "inertial_position_m": m.body_ipos[i].tolist(),
                    "inertial_quaternion_wxyz": m.body_iquat[i].tolist(),
                    "principal_inertia_kg_m2": m.body_inertia[i].tolist(),
                }
            )
    joints = []
    for aid, name in enumerate(env.joint_names):
        if not name.startswith(prefix):
            continue
        jid = m.actuator_trnid[aid, 0]
        joints.append(
            {
                "name": name[len(prefix) :],
                "range_rad": m.jnt_range[jid].tolist(),
                "motor_torque_range_Nm": m.actuator_forcerange[aid].tolist(),
                "armature_kg_m2": float(m.dof_armature[m.jnt_dofadr[jid]]),
                "damping_Nm_s_per_rad": float(m.dof_damping[m.jnt_dofadr[jid]]),
            }
        )
    write_json(
        out / "robot_parameters.json",
        {
            "robot": "K1_22dof",
            "total_mass_kg": sum(b["mass_kg"] for b in bodies),
            "bodies": bodies,
            "joints": joints,
            "note": "Serial-joint MJCF approximation; model values, not a measurement of your robot.",
        },
    )


class Video:
    def __init__(self, env, out):
        import imageio.v2 as imageio

        self.env = env
        self.c = env.config["render"]
        self.out = out
        self.renderer = mujoco.Renderer(env.model, self.c["height"], self.c["width"])
        self.writer = imageio.get_writer(
            str(out / "passing.mp4"), fps=self.c["fps"], macro_block_size=1
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, self.camera)
        self.camera.distance = self.c["camera_distance"]
        self.camera.azimuth = self.c["camera_azimuth"]
        self.camera.elevation = self.c["camera_elevation"]
        self.camera.lookat[:] = self.c["camera_lookat"]
        self.next_time = 0.0
        self.count = 0

    def frame(self, phase):
        from PIL import Image, ImageDraw

        e = self.env
        if e.data.time + 1e-9 < self.next_time:
            return
        self.next_time += 1 / self.c["fps"]
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        self.renderer.update_scene(e.data, camera=self.camera, scene_option=opt)
        im = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(im)
        speed = np.linalg.norm(e.observe()["ball_velocity"])
        draw.rectangle((12, 12, 890, 78), fill=(12, 23, 34))
        draw.text(
            (24, 20),
            f"HSL M-Field 14 x 9 m | K1 3v3 | t={e.data.time:5.2f}s | {phase} | ball={speed:.2f}m/s",
            fill="white",
        )
        draw.text(
            (24, 44),
            f"Articulated torque control | {e.config['robot']['base_mode']} pelvis | uncalibrated contacts",
            fill=(245, 198, 95),
        )
        self.writer.append_data(np.asarray(im))
        if self.count == 0:
            im.save(self.out / "preview.png")
        self.count += 1

    def close(self):
        self.writer.close()
        self.renderer.close()


def run_passing(c, out, video=False, viewer=False):
    env = SoccerEnv(c)
    snapshot(env, out)
    controller = PassingController(env)
    p = c["passing"]
    receiver = next(r for r in c["robot"]["instances"] if r["name"] == p["receiver"])
    receiver_xy = np.array(receiver["position"])
    initial = env.observe()["ball_position"].copy()
    contacts = []
    sender_touch = False
    receiver_touch = False
    hold = 0.0
    success = False
    success_time = None
    min_receiver_distance = float("inf")
    max_speed = 0.0
    last_contact = {}
    recorder = Video(env, out) if video else None
    handle = None
    if viewer:
        import mujoco.viewer

        handle = mujoco.viewer.launch_passive(env.model, env.data)
    start = time.monotonic()
    next_record = 0.0
    dt = env.model.opt.timestep
    try:
        with (out / "trajectory.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "vx_mps",
                    "vy_mps",
                    "vz_mps",
                    "wx_radps",
                    "wy_radps",
                    "wz_radps",
                    "receiver_distance_m",
                    "phase",
                ]
            )
            for _ in range(round(p["timeout"] / dt)):
                controller.update()
                obs = env.step()
                pos, vel = obs["ball_position"], obs["ball_velocity"]
                speed = float(np.linalg.norm(vel))
                distance = float(np.linalg.norm(pos[:2] - receiver_xy))
                max_speed = max(max_speed, speed)
                min_receiver_distance = min(min_receiver_distance, distance)
                for contact in env.ball_contacts():
                    name = contact["geom"]
                    if contact["normal_force"] < 0.01:
                        continue
                    if name.startswith(p["sender"] + "/"):
                        sender_touch = True
                    if sender_touch and name.startswith(p["receiver"] + "/"):
                        receiver_touch = True
                    if (
                        name != "turf"
                        and env.data.time - last_contact.get(name, -1) > 0.1
                    ):
                        contacts.append({"time": float(env.data.time), **contact})
                        last_contact[name] = env.data.time
                inside = (
                    sender_touch
                    and receiver_touch
                    and distance < p["receive_radius"]
                    and speed < p["receive_speed"]
                    and np.linalg.norm(pos[:2] - initial[:2]) > 0.5
                )
                hold = hold + dt if inside else 0.0
                if hold >= p["receive_hold"] and not success:
                    success = True
                    success_time = float(env.data.time)
                if env.data.time + 1e-9 >= next_record:
                    writer.writerow(
                        [
                            obs["time"],
                            *pos,
                            *vel,
                            *obs["ball_angular_velocity"],
                            distance,
                            controller.phase,
                        ]
                    )
                    next_record += 0.01
                if recorder:
                    recorder.frame(controller.phase)
                if handle:
                    if not handle.is_running():
                        break
                    handle.sync()
                    delay = env.data.time - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(min(delay, 0.02))
    finally:
        if recorder:
            recorder.close()
        if handle:
            handle.close()
    result = {
        "success": bool(success),
        "success_time_s": success_time,
        "sender_contact": sender_touch,
        "receiver_contact": receiver_touch,
        "receiver_min_distance_m": min_receiver_distance,
        "final_ball_position_m": env.observe()["ball_position"].tolist(),
        "final_ball_speed_mps": float(np.linalg.norm(env.observe()["ball_velocity"])),
        "peak_ball_speed_mps": max_speed,
        "ball_status": env.ball_status(),
        "simulated_seconds": float(env.data.time),
        "wall_seconds": time.monotonic() - start,
        "mujoco_warning_count": int(sum(w.number for w in env.data.warning)),
        "base_mode": c["robot"]["base_mode"],
        "dynamics_calibrated": False,
        "success_definition": "sender contact, then receiver contact, distance and speed thresholds for receive_hold; >0.5m travel",
        "contacts": contacts,
    }
    write_json(out / "result.json", result)
    print(json.dumps(result, indent=2))
    return result


def probe(c, out):
    """Independent ball-only roll and drop experiments for contact calibration."""
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for mode in ("roll", "drop"):
        env = SoccerEnv(c, robots=False)
        r = c["ball"]["radius"]
        # Lane at y=2 avoids the goal frame/net when rolling beyond the end line.
        env.reset(
            ball_position=[0, 2, r + 0.001] if mode == "roll" else [0, 2, 1 + r],
            ball_velocity=[2, 0, 0] if mode == "roll" else [0, 0, 0],
        )
        if mode == "roll":
            env.data.qvel[env.ball_v + 4] = 2 / r  # rolling about +y gives forward +x
        trace = []
        for k in range(round(8 / env.model.opt.timestep)):
            obs = env.step()
            if k % 10 == 0:
                trace.append(
                    [obs["time"], *obs["ball_position"], *obs["ball_velocity"]]
                )
        np.savetxt(
            out / f"{mode}.csv",
            np.array(trace),
            delimiter=",",
            header="time,x,y,z,vx,vy,vz",
            comments="",
        )
        trace = np.array(trace)
        post_impact = trace[trace[:, 0] > 0.5]
        reports.append(
            {
                "mode": mode,
                "distance_m": float(trace[-1, 1]),
                "final_speed_mps": float(np.linalg.norm(trace[-1, 4:7])),
                "peak_height_after_0_5s_m": float(post_impact[:, 3].max()),
            }
        )
    write_json(
        out / "calibration_probe.json",
        {"calibrated": False, "experiments": reports, "config": c["contacts"]},
    )
    print(json.dumps(reports, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "pass", "probe"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", action="append", default=[], metavar="DOTTED.KEY=YAML")
    parser.add_argument("--out", type=Path, default=Path("runs/default"))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    c = load_config(args.config, args.set)
    out = args.out.resolve()
    if args.command == "build":
        env = SoccerEnv(c)
        snapshot(env, out)
        print(
            f"{out}/scene.xml: nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}"
        )
    elif args.command == "pass":
        run_passing(c, out, args.video, args.viewer)
    else:
        probe(c, out)


if __name__ == "__main__":
    main()
