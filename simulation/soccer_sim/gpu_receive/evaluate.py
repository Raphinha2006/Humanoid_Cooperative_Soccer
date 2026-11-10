"""Held-out final evaluation. Run only after training and checkpoint selection finish."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .env import BatchEnv
from .model import make_actor, residual
from .receipt import write_receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--seed", type=int, default=230000)
    p.add_argument("--checkpoint", default="best.pt")
    a = p.parse_args()
    name = a.checkpoint
    if not (a.run / name).exists():
        if name != "best.pt":
            raise FileNotFoundError(a.run / name)
        name = "baseline.pt"  # No learned checkpoint beat the native development score.
    saved = torch.load(a.run / name, map_location="cpu", weights_only=False)
    cfg = saved.get("settings", {}).get("cfg") or yaml.safe_load(
        Path("configs/supermimic_g1_receive.yaml").read_text())
    e = BatchEnv(cfg, a.num_envs)
    actor = make_actor().to(e.device)
    actor.load_state_dict(saved["actor"])
    actor.eval()
    results = []
    initial = None
    for baseline in [True, False]:
        s = e.reset(a.seed)
        # Store and reapply identical initial physical state, not just a PRNG label.
        if initial is None:
            initial = (e.qpos.clone(), e.qvel.clone())
        else:
            import mujoco_warp as mjw

            e.qpos.copy_(initial[0])
            e.qvel.copy_(initial[1])
            mjw.forward(e.wm, e.wd)
            s = e.physical()
        incoming = torch.full((e.n,), float("nan"), device=e.device)
        outgoing = incoming.clone()
        first_contact_xy = torch.full((e.n, 2), float("nan"), device=e.device)
        for _ in range(300):
            with torch.no_grad():
                before = s
                q = e.prepare().clone()
                q[:, 31:43] += (
                    torch.zeros(e.n, 12, device=e.device)
                    if baseline
                    else residual(actor, s)
                )
                s = e.step(q)
                touched = e.contacts_t[:, 1] < 1e5
                first = touched & incoming.isnan()
                incoming[first] = before[first, 153:155].norm(dim=-1)
                first_contact_xy[first] = s[first, 150:152] - s[first, 75:77]
                after_100ms = (
                    touched & outgoing.isnan() & (e.elapsed >= e.contacts_t[:, 1] + 0.1)
                )
                outgoing[after_100ms] = s[after_100ms, 153:155].norm(dim=-1)
        metrics = e.metrics()
        e.check_capacity()
        moving_strict = e.strict & (incoming > 0.5)
        measured = incoming.isfinite() & outgoing.isfinite()
        metrics["moving_ball_strict_diagnostic"] = moving_strict.float().mean().item()
        metrics["contact_speed_measurement_fraction"] = measured.float().mean().item()
        metrics["incoming_speed_measured_mean"] = (
            float(incoming[measured].mean()) if measured.any() else None
        )
        metrics["speed_100ms_after_contact_mean"] = (
            float(outgoing[measured].mean()) if measured.any() else None
        )
        rel = s[:, 150:152] - s[:, 75:77]

        def nullable(v):
            return np.where(
                torch.isfinite(v).cpu().numpy(), v.cpu().numpy(), None
            ).tolist()

        results.append(
            {
                "checkpoint": "baseline" if baseline else name,
                "mean": metrics,
                "per_world": {
                    "strict": e.strict.cpu().tolist(),
                    "fallen": e.fallen.cpu().tolist(),
                    "hold": e.hold.cpu().tolist(),
                    "max_hold": e.max_hold.cpu().tolist(),
                    "handoff": (e.handoff_hold >= 0.8).cpu().tolist(),
                    "distance": rel.norm(dim=-1).cpu().tolist(),
                    "contacts": e.contacts_t.cpu().tolist(),
                    "max_travel": e.max_travel.cpu().tolist(),
                    "facing_at_end": e.facing(s).cpu().tolist(),
                    "speed_at_end": s[:, 153:155].norm(dim=-1).cpu().tolist(),
                    "moving_ball_strict_diagnostic": moving_strict.cpu().tolist(),
                    "speed_before_receiver_contact": nullable(incoming),
                    "speed_100ms_after_receiver_contact": nullable(outgoing),
                    "ball_relative_xy_at_receiver_contact": nullable(first_contact_xy),
                },
                "capacity_overflow_max": int((e.wd.overflow.numpy() & 511).max()),
            }
        )
        print(
            json.dumps({"checkpoint": results[-1]["checkpoint"], "metrics": metrics}),
            flush=True,
        )
    np.savez_compressed(
        a.run / f"test_initial_{a.seed}_{Path(name).stem}.npz",
        qpos=initial[0].cpu().numpy(),
        qvel=initial[1].cpu().numpy(),
    )
    (a.run / f"independent_{a.seed}_{Path(name).stem}.json").write_text(
        json.dumps(results, indent=2)
    )
    write_receipt(
        a.run,
        name,
        a.seed,
        "gpu",
        a.num_envs,
        [
            a.run / f"independent_{a.seed}_{Path(name).stem}.json",
            a.run / f"test_initial_{a.seed}_{Path(name).stem}.npz",
        ],
    )


if __name__ == "__main__":
    main()
