"""CPU MuJoCo confirmation from the saved independent GPU physical initial states."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from ..joga_env import JogaPassEnv
from .model import make_actor, residual
from .receipt import write_receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--seed", type=int, default=230000)
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(2)
    name = a.checkpoint if (a.run / a.checkpoint).exists() else "baseline.pt"
    initial = np.load(a.run / f"test_initial_{a.seed}_{Path(name).stem}.npz")
    saved = torch.load(a.run / name, map_location="cpu", weights_only=False)
    actor = make_actor()
    actor.load_state_dict(saved["actor"])
    actor.eval()
    cfg = yaml.safe_load(Path("configs/supermimic_g1_receive.yaml").read_text())
    cfg["receiver_joint_residual_scale"] = [0.15] * 12
    env = JogaPassEnv(cfg)
    results = []
    for baseline in [True, False]:
        per = []
        for i in range(min(a.episodes, len(initial["qpos"]))):
            env.reset(a.seed)
            env.d.qpos[:] = initial["qpos"][i]
            env.d.qvel[:] = initial["qvel"][i]
            env.d.qacc_warmstart[:] = 0
            mujoco.mj_forward(env.m, env.d)
            env.initial_ball = env.d.qpos[env.ball_q : env.ball_q + 2].copy()
            s = env.state()
            done = False
            hold = 0.0
            while not done:
                with torch.no_grad():
                    d = (
                        torch.zeros(12)
                        if baseline
                        else residual(actor, torch.tensor(s[:160])[None])[0]
                    )
                a20 = np.r_[np.zeros(8), d.numpy() / 0.15]
                s, done = env.step(a20)
                ball = env.d.qpos[env.ball_q : env.ball_q + 3]
                dist = np.linalg.norm(ball[:2] - env.d.xpos[env.skills[1].body, :2])
                speed = np.linalg.norm(env.d.qvel[env.ball_v : env.ball_v + 2])
                contacts = env.first_contact_times
                ordered = (
                    all(t is not None for t in contacts) and contacts[0] < contacts[1]
                )
                hold = (
                    hold + 0.02
                    if ordered
                    and env.facing_valid()
                    and dist <= 1.2
                    and speed < 0.6
                    and not env.fallen
                    else 0.0
                )
            per.append(
                dict(world=i, **env.metrics(), handoff_at_end=hold >= 0.8 - 1e-8)
            )
        item = {
            "checkpoint": "baseline" if baseline else name,
            "episodes": per,
            "mean": {
                k: float(np.mean([m[k] for m in per]))
                for k in [
                    "front_pass_success",
                    "fallen",
                    "handoff_at_end",
                    "final_receiver_distance",
                ]
            },
        }
        results.append(item)
        print(
            json.dumps({"checkpoint": item["checkpoint"], "mean": item["mean"]}),
            flush=True,
        )
    (a.run / f"cpu_confirmation_{a.seed}_{Path(name).stem}.json").write_text(
        json.dumps(results, indent=2)
    )
    write_receipt(
        a.run,
        name,
        a.seed,
        "cpu",
        len(results[0]["episodes"]),
        [
            a.run / f"cpu_confirmation_{a.seed}_{Path(name).stem}.json",
            a.run / f"test_initial_{a.seed}_{Path(name).stem}.npz",
        ],
    )


if __name__ == "__main__":
    main()
