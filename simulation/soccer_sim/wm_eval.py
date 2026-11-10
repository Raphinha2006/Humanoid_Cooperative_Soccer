"""Independent paired evaluation of all trained students, with strict final capture."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .wm_env import SkillPassEnv
from .wm_train import collect
from .world_model import Student


def record(env, student, seed, path):
    import imageio.v2 as imageio

    env.reset(seed)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.4, 0.3, 0.35]
    camera.distance, camera.azimuth, camera.elevation = 2.8, 120, -25
    if env.settings.get("facing_mode") == "mutual":
        camera.lookat[:] = [0.6, 0, 0.35]
        camera.distance, camera.azimuth, camera.elevation = 3.2, 90, -20
    scene = mujoco.MjvOption()
    scene.geomgroup[2] = 1
    renderer = mujoco.Renderer(env.m, height=720, width=960)
    writer = imageio.get_writer(str(path), fps=10, codec="libx264")
    rows, done = [], False
    while not done:
        state = env.state()
        with torch.no_grad():
            action = (
                np.zeros(8)
                if student is None
                else student(torch.from_numpy(state)[None])[0].numpy()
            )
        renderer.update_scene(env.d, camera=camera, scene_option=scene)
        frame = renderer.render()
        if not rows:
            imageio.imwrite(path.with_suffix(".png"), frame)
        writer.append_data(frame)
        rows.append(
            [
                env.elapsed,
                *env.d.qpos[env.env.ball_q : env.env.ball_q + 3],
                *env.d.xpos[env.skills[0].body],
                *env.d.xpos[env.skills[1].body],
                *action,
            ]
        )
        _, done = env.step(action)
    writer.close()
    renderer.close()
    path.with_suffix(".json").write_text(
        json.dumps({"seed": int(seed), "metrics": env.metrics()}, indent=2)
    )
    with path.with_suffix(".csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "ball_x",
                "ball_y",
                "ball_z",
                "sender_x",
                "sender_y",
                "sender_z",
                "receiver_x",
                "receiver_y",
                "receiver_z",
            ]
            + [f"command_{i}" for i in range(8)]
        )
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--record-only",
        type=Path,
        help="Record one saved checkpoint without repeating evaluation",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load((args.run / "settings.yaml").read_text())
    torch.set_num_threads(2)
    env = SkillPassEnv(cfg)
    if args.record_only:
        saved = torch.load(args.record_only, map_location="cpu", weights_only=True)
        student = Student(
            saved["mean"],
            saved["std"],
            saved["settings"].get("student_action_limit", 1.0),
            saved["settings"].get("student_action_mask"),
        )
        student.load_state_dict(saved["student"])
        student.eval()
        record(
            env,
            student,
            args.seed,
            args.run / f"{args.record_only.stem}_seed_{args.seed}.mp4",
        )
        return
    mujoco.mj_saveLastXML(str(args.run / "runtime_scene.xml"), env.m)
    limits = []
    for skill in env.skills:
        limits.append(
            {
                "kp": skill.kp.tolist(),
                "kd": skill.kd.tolist(),
                "action_scale": skill.scale.tolist(),
                "torque_limit_Nm": skill.limit.tolist(),
                "dof_damping": env.m.dof_damping[skill.v].tolist(),
            }
        )
    (args.run / "runtime_actuation.json").write_text(json.dumps(limits, indent=2))
    results = []
    checkpoints = [None] + sorted(args.run.glob("round_*.pt"))
    for path in checkpoints:
        student = None
        if path:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            student = Student(
                saved["mean"],
                saved["std"],
                saved["settings"].get("student_action_limit", 1.0),
                saved["settings"].get("student_action_mask"),
            )
            student.load_state_dict(saved["student"])
            student.eval()
        episodes = collect(
            env,
            student,
            range(args.seed, args.seed + args.episodes),
            np.random.default_rng(0),
        )
        metrics = [e["metrics"] for e in episodes]
        keys = [
            "sender_contact",
            "receiver_contact",
            "captured",
            "stable_receive_at_end",
            "pass_success",
            "fallen",
            "min_receiver_distance",
            "final_target_error",
        ]
        item = {
            "checkpoint": path.name if path else "baseline",
            "seed_start": args.seed,
            "episodes": args.episodes,
            "mean": {key: float(np.mean([m[key] for m in metrics])) for key in keys},
            "episodes_detail": [dict(seed=e["seed"], **e["metrics"]) for e in episodes],
        }
        results.append(item)
        print(
            json.dumps({k: v for k, v in item.items() if k != "episodes_detail"}),
            flush=True,
        )
        (args.run / "paired_evaluation.json").write_text(json.dumps(results, indent=2))
        (args.run / f"paired_evaluation_{args.seed}.json").write_text(
            json.dumps(results, indent=2)
        )
        if args.video and (path is None or path == checkpoints[-1]):
            record(
                env,
                student,
                args.seed,
                args.run / (path.stem + ".mp4" if path else "baseline.mp4"),
            )


if __name__ == "__main__":
    main()
