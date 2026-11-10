"""Fixed native-policy trajectories for open-loop WM validation, separate from control tests.

The recorded, executed joint commands are supplied to every predictor. This measures
dynamics prediction under known future actions, not an online ball forecast with
unknown future actions. Never feed this dataset back into training or replay.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml

from ..joga_env import JogaPassEnv
from ..supermimic_core.kinematics import DifferentiableKinematics, build_kinematic_tree
from ..supermimic_core.math import quat_mul
from ..supermimic_core.networks import _rotvec_to_quat
from .model import PhysicalWM


def collect(a):
    from .env import BatchEnv

    if a.out.exists():
        raise FileExistsError(a.out)
    cfg = yaml.safe_load(Path("configs/supermimic_g1_receive.yaml").read_text())
    env = BatchEnv(cfg, a.num_envs)
    s = env.reset(a.seed)
    initial_qpos = env.qpos.cpu().numpy().copy()
    initial_qvel = env.qvel.cpu().numpy().copy()
    states = [s.cpu().numpy()]
    actions = []
    with torch.no_grad():
        for _ in range(300):
            s = env.step(env.prepare().clone())
            states.append(s.cpu().numpy())
            actions.append(env.qcmd.cpu().numpy().copy())
    overflow = env.wd.overflow.numpy()
    if (overflow & 511).any():
        raise RuntimeError(
            "Prediction dataset has contact/constraint capacity overflow"
        )
    meta = {
        "purpose": "fixed native-policy WM validation, never training data",
        "seed": a.seed,
        "num_envs": a.num_envs,
        "cfg": cfg,
        "mujoco": __import__("mujoco").__version__,
        "mujoco_warp": version("mujoco-warp"),
        "solver_limit_world_fraction": float(np.mean((overflow & 1536) != 0)),
        "metrics": env.metrics(),
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["env.py", "native.py", "model.py"]
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        states=np.stack(states),
        actions=np.stack(actions),
        first_contacts=env.contacts_t.cpu().numpy(),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        metadata=json.dumps(meta),
    )
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


def constant_velocity(s):
    p = s.clone()
    for off in [0, 75]:
        p[:, off : off + 3] += s[:, off + 7 : off + 10] * 0.02
        # A free joint's angular velocity is local to its child frame.
        p[:, off + 3 : off + 7] = quat_mul(
            s[:, off + 3 : off + 7],
            _rotvec_to_quat(s[:, off + 10 : off + 13] * 0.02),
        )
        p[:, off + 13 : off + 44] += s[:, off + 44 : off + 75] * 0.02
    p[:, 150:153] += s[:, 153:156] * 0.02
    p[:, 159] += 0.02
    return p


@torch.no_grad()
def validate(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    data = np.load(a.dataset)
    saved = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    wm = PhysicalWM(saved["settings"]["wm_width"]).to(a.device)
    wm.load_state_dict(saved["wm"])
    wm.eval()
    cfg = json.loads(str(data["metadata"]))["cfg"]
    cpu = JogaPassEnv(cfg)
    names = [
        cpu.m.body(i).name
        for i in range(cpu.m.nbody)
        if cpu.m.body(i).name.startswith("A2/")
    ]
    joints = [cpu.m.joint(i).name.rsplit("/", 1)[-1] for i in cpu.skills[1].j]
    fk = DifferentiableKinematics(build_kinematic_tree(cpu.m, names, joints)).to(
        a.device
    )
    feet = [i for i, name in enumerate(names) if "ankle_roll" in name]
    if len(feet) != 2:
        raise ValueError(
            f"Expected two ankle-roll bodies, found {[names[i] for i in feet]}"
        )
    ss = torch.tensor(data["states"], device=a.device)
    aa = torch.tensor(data["actions"], device=a.device)
    contacts = torch.tensor(data["first_contacts"][:, 1], device=a.device)
    rows = []
    for start in range(25, 251, 25):
        predictions = {"wm": ss[start].clone(), "constant_velocity": ss[start].clone()}
        for horizon in range(1, 51):
            predictions["wm"] = wm(predictions["wm"], aa[start + horizon - 1])
            predictions["constant_velocity"] = constant_velocity(
                predictions["constant_velocity"]
            )
            if horizon not in [1, 5, 25, 32, 50]:
                continue
            true = ss[start + horizon]
            body_true, _ = fk(true[:, 75:78], true[:, 78:82], true[:, 88:119])
            masks = {
                "all": torch.ones(len(true), device=a.device, dtype=torch.bool),
                "receiver_contact_in_window": (contacts > start * 0.02)
                & (contacts <= (start + horizon) * 0.02),
                "ball_moving_at_start": ss[start, :, 153:155].norm(dim=-1) > 0.25,
            }
            for method, p in predictions.items():
                body_pred, _ = fk(p[:, 75:78], p[:, 78:82], p[:, 88:119])
                errors = {
                    "ball_position_m": (p[:, 150:153] - true[:, 150:153])
                    .square()
                    .sum(-1),
                    "ball_velocity_m_s": (p[:, 153:156] - true[:, 153:156])
                    .square()
                    .sum(-1),
                    "receiver_root_m": (p[:, 75:78] - true[:, 75:78]).square().sum(-1),
                    "receiver_body_m": (body_pred - body_true)
                    .square()
                    .sum(-1)
                    .mean(-1),
                    "receiver_feet_m": (body_pred[:, feet] - body_true[:, feet])
                    .square()
                    .sum(-1)
                    .mean(-1),
                }
                for subset, mask in masks.items():
                    n = int(mask.sum())
                    rows.append(
                        {
                            "method": method,
                            "start_tick": start,
                            "horizon_ticks": horizon,
                            "horizon_s": horizon * 0.02,
                            "subset": subset,
                            "count": n,
                            "mean_square_error": {
                                key: float(err[mask].mean()) if n else None
                                for key, err in errors.items()
                            },
                        }
                    )
    aggregates = []
    for method in ["wm", "constant_velocity"]:
        for horizon in [1, 5, 25, 32, 50]:
            for subset in ["all", "receiver_contact_in_window", "ball_moving_at_start"]:
                selected = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["horizon_ticks"] == horizon
                    and r["subset"] == subset
                ]
                n = sum(r["count"] for r in selected)
                aggregates.append(
                    {
                        "method": method,
                        "horizon_s": horizon * 0.02,
                        "subset": subset,
                        "window_world_count": n,
                        "rmse": {
                            k: (
                                sum(
                                    r["mean_square_error"][k] * r["count"]
                                    for r in selected
                                    if r["count"]
                                )
                                / n
                            )
                            ** 0.5
                            if n
                            else None
                            for k in selected[0]["mean_square_error"]
                        },
                    }
                )
    result = {
        "checkpoint": str(a.checkpoint),
        "iteration": saved["iteration"],
        "transitions": saved["transitions"],
        "dataset": str(a.dataset),
        "dataset_sha256": hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
        "known_future_executed_actions": True,
        "note": "Overlapping windows are correlated; counts are not independent trials. Not used for checkpoint selection.",
        "aggregates": aggregates,
        "by_anchor": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({"checkpoint": str(a.checkpoint), "aggregates": aggregates}),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--seed", type=int, default=225000)
    c.add_argument("--num-envs", type=int, default=256)
    v = sub.add_parser("validate")
    v.add_argument("--dataset", type=Path, required=True)
    v.add_argument("--checkpoint", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--device", default="cpu")
    a = p.parse_args()
    torch.set_num_threads(2)
    (collect if a.command == "collect" else validate)(a)


if __name__ == "__main__":
    main()
