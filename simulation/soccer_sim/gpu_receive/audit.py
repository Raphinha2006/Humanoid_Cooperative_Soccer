"""Preselected checkpoint repeatability audit on development initials only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .env import BatchEnv
from .model import make_actor
from .train import evaluate


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=220000)
    p.add_argument("--probe-seed", type=int, default=250000)
    p.add_argument("--controller-kinematics", choices=["legacy", "current"], default="legacy")
    a = p.parse_args()
    if 240000 in [a.seed, a.probe_seed] or a.repeats < 1:
        p.error("Reserved final test cannot be used by this development audit")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    checkpoints = {}
    cfg = None
    for item in a.checkpoint:
        label, path = item.split("=", 1)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        def physics_config(c):
            return {k: v for k, v in c.items() if k not in {"gpu_njmax", "gpu_nconmax"}}
        if cfg is not None and physics_config(cfg) != physics_config(saved["settings"]["cfg"]):
            raise ValueError("Checkpoint physics/configurations differ")
        cfg = saved["settings"]["cfg"]
        checkpoints[label] = {"path": str(Path(path).resolve()), "sha256": digest(path),
                              "iteration": saved["iteration"], "actor": saved["actor"]}
        del saved
    manifest = {"created_epoch": time.time(), "seed": a.seed, "probe_seed": a.probe_seed,
                "n": a.num_envs, "repeats": a.repeats, "heldout_used": False,
                "checkpoints": {k: {f: v for f, v in d.items() if f != "actor"}
                                for k, d in checkpoints.items()},
                "reward_accounting": "previous_residual_v1", "strict_criterion_changed": False}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cfg = dict(cfg, gpu_njmax=1024, controller_kinematics=a.controller_kinematics)
    manifest["controller_kinematics"] = a.controller_kinematics
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = BatchEnv(cfg, a.num_envs)
    actor = make_actor().to(env.device).eval()
    actor.load_state_dict(next(iter(checkpoints.values()))["actor"])
    initial = {}
    results = []
    # Alternate checkpoint ordering across repeats to expose load/time confounding.
    labels = ["native", *checkpoints]
    for seed, repeats in [(a.seed, a.repeats), (a.probe_seed, 1)]:
        for repeat in range(repeats):
            for label in labels if repeat % 2 == 0 else list(reversed(labels)):
                if label != "native":
                    actor.load_state_dict(checkpoints[label]["actor"])
                # Confirm the exact physical initial arrays, not just the seed label.
                env.reset(seed)
                actual = (env.qpos.cpu().numpy().copy(), env.qvel.cpu().numpy().copy())
                if seed not in initial:
                    initial[seed] = actual
                    np.savez_compressed(a.out / f"initial_{seed}.npz", qpos=actual[0], qvel=actual[1])
                else:
                    for x, y in zip(actual, initial[seed]):
                        np.testing.assert_array_equal(x, y)
                score = evaluate(env, actor, seed, baseline=label == "native", details=True)
                per_world = score.pop("per_world")
                record = {"label": label, "seed": seed, "repeat": repeat,
                          "metrics": score, "per_world": per_world,
                          "epoch": time.time(), "initials_exactly_equal": True}
                results.append(record)
                with (a.out / "results.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps({k: v for k, v in record.items() if k != "per_world"}), flush=True)
    summary = {}
    for label in labels:
        rows = [r for r in results if r["label"] == label and r["seed"] == a.seed]
        rates = [r["metrics"]["front_pass_success"] for r in rows]
        summary[label] = {"strict_rates": rates, "mean": float(np.mean(rates)),
                          "std": float(np.std(rates)), "min": min(rates), "max": max(rates)}
    (a.out / "summary.json").write_text(json.dumps({"status": "complete", "summary": summary,
        "scope": "development repeatability; not independent training replications",
        "heldout_used": False}, indent=2))


if __name__ == "__main__":
    main()
