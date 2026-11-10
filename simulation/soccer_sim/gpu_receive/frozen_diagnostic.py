"""Fixed-trajectory WM diagnosis with sender controls isolated from receiver inputs.

Recorded receiver controls are used in every forecast mode. This is a diagnostic
of model/controller prediction, not a closed-loop task evaluation or a planner.
"""

import argparse
import json
from pathlib import Path

import torch

from .audit import digest
from .controller import SenderController
from .env import BatchEnv
from .model import PhysicalWM, clip_command


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=225300)
    a = p.parse_args()
    if a.seed == 240000:
        p.error("Final heldout seed is reserved")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    paths = dict(item.split("=", 1) for item in a.checkpoint)
    first = torch.load(next(iter(paths.values())), map_location="cpu", weights_only=False)
    cfg = dict(first["settings"]["cfg"], gpu_njmax=1024, controller_kinematics="current")
    del first
    env = BatchEnv(cfg, a.num_envs)
    sender = SenderController(env)
    state = env.reset(a.seed)
    initial = {"qpos": env.qpos.cpu().clone(), "qvel": env.qvel.cpu().clone()}
    states, controls, memories, contacts = [state.cpu().clone()], [], [], [env.contacts_t.cpu().clone()]
    for _ in range(300):
        target = env.prepare().clone()
        memories.append(sender.snapshot(env).cpu())
        state = env.step(target)
        controls.append(env.qcmd.cpu().clone())
        states.append(state.cpu().clone())
        contacts.append(env.contacts_t.cpu().clone())
    env.check_capacity()
    trace = dict(states=torch.stack(states), controls=torch.stack(controls),
                 sender_memory=torch.stack(memories), contacts=torch.stack(contacts),
                 initial=initial, seed=a.seed, cfg=cfg, policy="native_frozen")
    torch.save(trace, a.out / "fixed_native_trajectories.pt")
    truth = trace["states"].to(env.device)
    actions = trace["controls"].to(env.device)
    memory = trace["sender_memory"].to(env.device)
    contact_flags = trace["contacts"][:, :, 1] < 1e5
    has_contact = contact_flags.any(0).to(env.device)
    first_contact = contact_flags.to(torch.int32).argmax(0).to(env.device)
    windows = {
        "pre_kick": (torch.full((env.n,), 15, device=env.device), torch.ones_like(has_contact)),
        "flight": (torch.full((env.n,), 80, device=env.device), torch.ones_like(has_contact)),
        "pre_contact": ((first_contact - 8).clamp(0, 250), has_contact & (first_contact >= 8) & (first_contact <= 250)),
        "post_contact": ((first_contact + 4).clamp(0, 250), has_contact & (first_contact <= 246)),
    }
    ids = torch.arange(env.n, device=env.device)
    results = []
    for label, path in paths.items():
        saved = torch.load(path, map_location=env.device, weights_only=False)
        wm = PhysicalWM(saved["settings"].get("wm_width", 2048)).to(env.device).eval()
        wm.load_state_dict(saved["wm"])
        for phase, (starts, valid) in windows.items():
            if not valid.any():
                continue
            for mode in ["held", "recorded", "closed_loop"]:
                predicted = truth[starts, ids].clone()
                mem = memory[starts, ids].clone()
                held = actions[starts, ids, :31].clone()
                for tick in range(50):
                    actual_control = actions[starts + tick, ids]
                    if mode == "recorded":
                        command = actual_control
                    else:
                        send = held
                        if mode == "closed_loop" and tick:
                            send, mem = sender(predicted, mem)
                        proposed = torch.cat([send, actual_control[:, 31:]], -1)
                        command = torch.cat([clip_command(predicted, proposed, env)[:, :31],
                                             actual_control[:, 31:]], -1)
                    predicted = wm(predicted, command)
                    if not torch.isfinite(predicted).all():
                        raise FloatingPointError(f"Nonfinite fixed forecast: {label}/{phase}/{mode}")
                    if tick + 1 in [8, 16, 32, 50]:
                        actual = truth[starts + tick + 1, ids]
                        def rmse(lo, hi):
                            return float((predicted[valid, lo:hi] - actual[valid, lo:hi]).square().sum(-1).mean().sqrt())
                        row = {"checkpoint": label, "iteration": saved["iteration"],
                               "phase": phase, "sender_control": mode, "horizon_steps": tick + 1,
                               "horizon_seconds": (tick + 1) * .02, "n": int(valid.sum()),
                               "ball_position_rmse_m": rmse(150, 153),
                               "receiver_root_rmse_m": rmse(75, 78),
                               "sender_root_rmse_m": rmse(0, 3)}
                        results.append(row)
                        print(json.dumps(row), flush=True)
        del wm, saved
        torch.cuda.empty_cache()
    report = {"scope": "frozen native trajectories, recorded receiver controls; no policy selection",
              "heldout_used": False, "seed": a.seed, "n": env.n,
              "controller_kinematics": "current",
              "trajectory_sha256": digest(a.out / "fixed_native_trajectories.pt"),
              "checkpoints": {k: {"path": v, "sha256": digest(v)} for k, v in paths.items()},
              "results": results}
    (a.out / "forecast_errors.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
