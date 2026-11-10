"""Paired real-physics evaluation; native G1, selected/final actor, joint ablation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .direct_env import DirectPassEnv
from .direct_model import act, make_actor


def load(run, checkpoint):
    cfg = yaml.safe_load((run / "settings.yaml").read_text())
    if cfg.get("embodiment") == "joga_g1":
        from .joga_env import JogaPassEnv

        env = JogaPassEnv(cfg)
    else:
        env = DirectPassEnv(cfg)
    actor = None
    if checkpoint != "baseline":
        saved = torch.load(run / checkpoint, map_location="cpu", weights_only=True)
        actor = make_actor(saved["size"], cfg["actor_hidden"])
        actor.load_state_dict(saved["actor"])
        actor.eval()
    return env, actor, cfg


def trial(env, actor, cfg, seed, ablate=False, video=None):
    s = env.reset(seed)
    done = False
    hold = 0.0
    handoff = False
    rows = []
    tick = 0
    writer = renderer = None
    if video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw

        writer = imageio.get_writer(str(video), fps=25, codec="libx264")
        env.m.vis.global_.offwidth = 960
        env.m.vis.global_.offheight = 720
        renderer = mujoco.Renderer(env.m, height=720, width=960)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (
            [2.5, 0, 0.65] if cfg.get("embodiment") == "joga_g1" else [0.6, 0, 0.35]
        )
        cam.distance = 8.5 if cfg.get("embodiment") == "joga_g1" else 3.2
        if cfg.get("video_closeup"):
            cam.distance = 2.8
        cam.azimuth = 90
        cam.elevation = -22
        opt = mujoco.MjvOption()
        opt.geomgroup[2] = 1
    while not done:
        with torch.no_grad():
            a = (
                np.zeros(20)
                if actor is None
                else act(actor, torch.from_numpy(s)[None], env.layout, cfg)[0].numpy()
            )
        if ablate:
            a[8:] = 0
        s, done = env.step(a)
        p = env.skills[1]
        ball = env.d.qpos[env.env.ball_q : env.env.ball_q + 3]
        dist = np.linalg.norm(ball[:2] - env.d.xpos[p.body, :2])
        speed = np.linalg.norm(env.d.qvel[env.env.ball_v : env.env.ball_v + 2])
        contact = (
            all(t is not None for t in env.first_contact_times)
            and env.first_contact_times[0] < env.first_contact_times[1]
        )
        valid = contact and dist <= 1.2 and speed < 0.6 and env.facing_valid()
        hold = hold + cfg["control_dt"] if valid else 0.0
        handoff |= hold >= 0.8 - 1e-8
        residual = np.array(cfg["receiver_joint_residual_scale"]) * a[8:]
        leg_target = (
            p.target[:12] if cfg.get("embodiment") == "joga_g1" else p.target[-12:]
        )
        rows.append([env.elapsed, *ball, speed, dist, *a, *leg_target, *residual])
        if renderer is not None and tick % 2 == 0:
            if cfg.get("video_closeup"):
                cam.lookat[:] = env.d.xpos[p.body] + np.array([0.1, 0, -0.25])
            renderer.update_scene(env.d, camera=cam, scene_option=opt)
            frame = renderer.render()
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            draw.rectangle([0, 0, 960, 62], fill=(18, 25, 35))
            draw.text(
                (16, 10),
                f"G1 pass -> receive | t={env.elapsed:.2f}s | speed={speed:.2f}m/s | distance={dist:.2f}m",
                fill=(255, 255, 255),
            )
            draw.text(
                (16, 35),
                f"receiver: native receiving policy + {'zero' if actor is None else 'learned'} leg residual | max delta={np.max(np.abs(residual)):.3f}rad",
                fill=(225, 230, 240),
            )
            frame = np.array(im)
            writer.append_data(frame)
            if tick == 0:
                imageio.imwrite(video.with_suffix(".png"), frame)
        tick += 1
    metrics = env.metrics()
    # Native demo threshold, strengthened with ordered real contacts and facing.
    metrics["handoff_event"] = bool(handoff and not metrics["fallen"])
    metrics["handoff_at_end"] = bool(hold >= 0.8 - 1e-8 and not metrics["fallen"])
    metrics["max_leg_residual_rad"] = float(np.max(np.abs(np.array(rows)[:, -12:])))
    if renderer is not None:
        writer.close()
        renderer.close()
        with video.with_suffix(".csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "time",
                    "ball_x",
                    "ball_y",
                    "ball_z",
                    "ball_speed",
                    "receiver_distance",
                ]
                + [f"action_{i}" for i in range(20)]
                + [f"leg_target_{i}" for i in range(12)]
                + [f"leg_residual_{i}" for i in range(12)]
            )
            w.writerows(rows)
        video.with_suffix(".json").write_text(
            json.dumps({"seed": seed, "metrics": metrics}, indent=2)
        )
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--seed", type=int, default=130000)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--checkpoints", nargs="+")
    p.add_argument("--video", action="store_true")
    p.add_argument("--record-only")
    p.add_argument("--ablate-legs", action="store_true")
    p.add_argument("--closeup", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    selected = (
        json.loads((args.run / "selected_checkpoint.json").read_text())["checkpoint"]
        if (args.run / "selected_checkpoint.json").exists()
        else "baseline"
    )
    final = "round_30.pt"
    names = list(dict.fromkeys(args.checkpoints or ["baseline", selected, final]))
    if args.record_only:
        names = [args.record_only]
    output = []
    for name in names:
        env, actor, cfg = load(args.run, name)
        cfg["video_closeup"] = args.closeup
        per = []
        seeds = (
            [args.seed]
            if args.record_only
            else range(args.seed, args.seed + args.episodes)
        )
        for seed in seeds:
            path = (
                args.run
                / f"{Path(name).stem}_{seed}{'_no_legs' if args.ablate_legs else ''}{'_closeup' if args.closeup else ''}.mp4"
                if args.video and seed == args.seed
                else None
            )
            per.append(
                dict(seed=seed, **trial(env, actor, cfg, seed, args.ablate_legs, path))
            )
        keys = [
            "pass_success",
            "front_pass_success",
            "captured",
            "fallen",
            "final_receiver_distance",
            "final_ball_speed",
            "handoff_event",
            "handoff_at_end",
            "max_leg_residual_rad",
        ]
        means = {k: float(np.mean([m[k] for m in per])) for k in keys}
        item = {"checkpoint": name, "mean": means, "episodes": per}
        output.append(item)
        print(json.dumps({"checkpoint": name, "mean": means}), flush=True)
    if not args.record_only:
        (
            args.run
            / f"paired_{args.seed}{'_no_legs' if args.ablate_legs else ''}.json"
        ).write_text(json.dumps(output, indent=2))
        (
            args.run
            / f"paired_{args.seed}{'_no_legs' if args.ablate_legs else ''}_runtime.json"
        ).write_text(
            json.dumps(
                {
                    "mujoco": mujoco.__version__,
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "checkpoints": names,
                    "seed_start": args.seed,
                    "episodes_per_checkpoint": args.episodes,
                    "ablate_legs": args.ablate_legs,
                    "code_sha256": {
                        name: hashlib.sha256(
                            (Path(__file__).parent / name).read_bytes()
                        ).hexdigest()
                        for name in [
                            "direct_eval.py",
                            "joga.py",
                            "joga_env.py",
                            "direct_model.py",
                        ]
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
