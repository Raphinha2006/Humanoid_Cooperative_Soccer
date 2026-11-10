"""Collect -> fit multi-step dynamics -> plan on-policy -> distil -> collect again."""

import argparse
import hashlib
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime
import torch
import yaml

from .skill_policies import source_manifest
from .wm_env import DEFAULT, SkillPassEnv
from .world_model import (
    Dynamics,
    Student,
    indices,
    plan_teacher,
    sample_windows,
    task_cost,
)


def collect(env, policy, seeds, rng, noise=0.0):
    episodes = []
    for seed in seeds:
        state = env.reset(int(seed))
        states, actions = [state], []
        done = False
        while not done:
            with torch.no_grad():
                action = (
                    np.zeros(8)
                    if policy is None
                    else policy(torch.from_numpy(state)[None])[0].numpy()
                )
            action = np.clip(action + rng.normal(0, noise, 8), -1, 1).astype(np.float32)
            action *= np.asarray(env.settings.get("student_action_mask", [1.0] * 8))
            state, done = env.step(action)
            states.append(state)
            actions.append(action)
        episodes.append(
            {
                "states": np.stack(states),
                "actions": np.stack(actions),
                "seed": int(seed),
                "metrics": env.metrics(),
            }
        )
    return episodes


def aggregate(episodes):
    metrics = [e["metrics"] for e in episodes]
    return {
        k: float(np.mean([m[k] for m in metrics]))
        for k in [
            "sender_contact",
            "receiver_contact",
            "captured",
            "fallen",
            "min_receiver_distance",
            "final_target_error",
        ]
    }


def save_data(path, episodes):
    lengths = np.array([len(e["actions"]) for e in episodes])
    np.savez_compressed(
        path,
        states=np.concatenate([e["states"] for e in episodes]),
        actions=np.concatenate([e["actions"] for e in episodes]),
        lengths=lengths,
        seeds=[e["seed"] for e in episodes],
    )


def fit_models(models, optimizers, episodes, rng, cfg, device):
    losses = []
    for model in models:
        model.requires_grad_(True)
    for _ in range(cfg["wm_updates"]):
        for model, optimizer in zip(models, optimizers):
            # Each ensemble member gets its own episode/window bootstrap samples.
            s, a = sample_windows(episodes, cfg["batch_size"], cfg["wm_window"], rng)
            s, a = torch.tensor(s, device=device), torch.tensor(a, device=device)
            pred, loss = s[:, 0], 0.0
            for t in range(cfg["wm_window"]):
                pred = model(pred, a[:, t])
                error = ((pred - s[:, t + 1]) / model.std) ** 2
                loss = loss + error.mean()
                bp = indices(model.layout, "ball_position")
                loss = loss + 3 * error[:, bp].mean()
            loss = loss / cfg["wm_window"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses[-100:]))


@torch.no_grad()
def validate_models(models, episodes, cfg, device, layout):
    # Fixed held-out episodes and window starts; never used for fitting or teacher queries.
    rng = np.random.default_rng(9147)
    s, a = sample_windows(episodes, 256, 10, rng)
    truth, actions = torch.tensor(s, device=device), torch.tensor(a, device=device)
    predictions = [truth[:, 0] for _ in models]
    metrics, snapshots = {}, []
    for t in range(10):
        predictions = [m(p, actions[:, t]) for m, p in zip(models, predictions)]
        mean = torch.stack(predictions).mean(0)
        if t + 1 in [1, 5, 10]:
            bp = indices(layout, "ball_position")
            metrics[f"ball_rmse_{t + 1}step_m"] = float(
                ((mean[:, bp] - truth[:, t + 1, bp]) ** 2).sum(-1).mean().sqrt()
            )
            errors = []
            for i in range(2):
                body = indices(layout, f"r{i}_body_positions")
                errors.append(
                    (mean[:, body] - truth[:, t + 1, body]).reshape(len(mean), -1, 3)
                )
            metrics[f"body_rmse_{t + 1}step_m"] = float(
                torch.cat(errors, 1).square().sum(-1).mean().sqrt()
            )
            snapshots.append(mean[:8].cpu().numpy())
    return metrics, {
        "predicted": np.stack(snapshots),
        "truth": truth[:8, [1, 5, 10]].transpose(0, 1).cpu().numpy(),
        "actions": a[:8],
    }


def improve_student(
    models, student, optimizer, episodes, labels, rng, cfg, device, layout
):
    own_states = np.concatenate([e["states"][:-1] for e in episodes])
    query = own_states[rng.integers(0, len(own_states), cfg["teacher_queries"])]
    targets = []
    for start in range(0, len(query), 64):
        s = torch.tensor(query[start : start + 64], device=device)
        targets.append(plan_teacher(models, s, student, layout, cfg).cpu().numpy())
    labels.append((query, np.concatenate(targets)))
    # Aggregate labels across visited student distributions, bounded to recent rounds.
    xs = torch.tensor(np.concatenate([x for x, _ in labels[-3:]]), device=device)
    ys = torch.tensor(np.concatenate([y for _, y in labels[-3:]]), device=device)
    for _ in range(cfg["student_updates"]):
        ix = torch.tensor(rng.integers(0, len(xs), cfg["batch_size"]), device=device)
        state, target = xs[ix], ys[ix]
        action = student(state)
        bc = (action - target).square().mean()
        # Short BPTT through a frozen WM; preserve state/action derivatives.
        imagined, imagined_cost = state, state.new_zeros(())
        if cfg["imagination_weight"]:
            for t in range(4):
                action = student(imagined)
                imagined = models[0](imagined, action)
                imagined_cost = (
                    imagined_cost + task_cost(imagined, action, layout, cfg).mean() / 4
                )
        loss = bc + cfg["imagination_weight"] * imagined_cost
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
    return {
        "distillation_mse": float(bc.detach()),
        "imagined_cost": float(imagined_cost.detach()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(
            initial_episodes=3,
            episodes_per_round=2,
            validation_episodes=2,
            evaluation_episodes=2,
            rounds=1,
            wm_updates=2,
            teacher_queries=8,
            teacher_updates=2,
            student_updates=2,
            batch_size=8,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    code_dir = args.output / "code"
    code_dir.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, code_dir / source.name)
    (code_dir / "sha256.json").write_text(
        json.dumps(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in code_dir.glob("*.py")
            },
            indent=2,
        )
    )
    (args.output / "settings.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(2)
    env = SkillPassEnv(cfg)
    (args.output / "state_layout.json").write_text(json.dumps(env.layout, indent=2))
    (args.output / "body_order.json").write_text(
        json.dumps(
            [[env.m.body(i).name for i in robot] for robot in env.link_ids], indent=2
        )
    )
    mujoco.mj_saveLastXML(str(args.output / "scene.xml"), env.m)
    (args.output / "environment.yaml").write_text(
        yaml.safe_dump(env.env.config, sort_keys=False)
    )
    (args.output / "runtime.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "mujoco": mujoco.__version__,
                "onnxruntime": onnxruntime.__version__,
                "device": args.device,
                "gpu": torch.cuda.get_device_name(args.device)
                if str(args.device).startswith("cuda")
                else None,
                "skills": source_manifest(cfg.get("kick_kind", "nubots")),
                "training_method": "skill-level WM planning + on-policy distillation + short BPTT",
            },
            indent=2,
        )
    )
    log_path = args.output / "metrics.jsonl"
    started = time.time()

    def log(record):
        record["wall_seconds"] = round(time.time() - started, 2)
        with log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    validation = collect(
        env, None, range(10000, 10000 + cfg["validation_episodes"]), rng, 0.35
    )
    baseline = collect(env, None, range(20000, 20000 + cfg["evaluation_episodes"]), rng)
    save_data(args.output / "validation.npz", validation)
    (args.output / "baseline_episodes.json").write_text(
        json.dumps([dict(seed=e["seed"], **e["metrics"]) for e in baseline], indent=2)
    )
    log({"stage": "baseline", "evaluation": aggregate(baseline)})
    episodes = collect(
        env,
        None,
        range(cfg["seed"], cfg["seed"] + cfg["initial_episodes"]),
        rng,
        cfg["exploration_std"],
    )
    states = np.concatenate([e["states"] for e in episodes])
    mean, std = states.mean(0), np.maximum(states.std(0), 0.1)
    models = [
        Dynamics(mean, std, env.layout, cfg).to(args.device)
        for _ in range(cfg["wm_ensemble"])
    ]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=cfg["wm_lr"]) for m in models]
    student = Student(
        mean, std, cfg.get("student_action_limit", 1.0), cfg.get("student_action_mask")
    ).to(args.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["student_lr"])
    before, _ = validate_models(models, validation, cfg, args.device, env.layout)
    log({"stage": "untrained_wm", "validation": before, "state_dim": env.size})
    labels = []
    for round_index in range(cfg["rounds"]):
        if round_index:
            current = deepcopy(student).cpu().eval()
            seed = (
                cfg["seed"]
                + cfg["initial_episodes"]
                + (round_index - 1) * cfg["episodes_per_round"]
            )
            new = collect(
                env,
                current,
                range(seed, seed + cfg["episodes_per_round"]),
                rng,
                cfg["exploration_std"] * 0.7,
            )
            episodes.extend(new)
        else:
            new = episodes
        log(
            {
                "stage": "collected",
                "round": round_index + 1,
                "episodes": len(episodes),
                "transitions": sum(len(e["actions"]) for e in episodes),
                "behavior": aggregate(new),
            }
        )
        loss = (
            None
            if cfg.get("skip_wm_fit", False)
            else fit_models(models, optimizers, episodes, rng, cfg, args.device)
        )
        wm_metrics, predictions = validate_models(
            models, validation, cfg, args.device, env.layout
        )
        np.savez_compressed(
            args.output / f"prediction_{round_index + 1}.npz", **predictions
        )
        learning = improve_student(
            models, student, optimizer, new, labels, rng, cfg, args.device, env.layout
        )
        evaluation = collect(
            env,
            deepcopy(student).cpu().eval(),
            range(20000, 20000 + cfg["evaluation_episodes"]),
            rng,
        )
        (args.output / f"evaluation_{round_index + 1}.json").write_text(
            json.dumps(
                [dict(seed=e["seed"], **e["metrics"]) for e in evaluation], indent=2
            )
        )
        record = {
            "stage": "round",
            "round": round_index + 1,
            "wm_train_loss": loss,
            "validation": wm_metrics,
            "learning": learning,
            "evaluation": aggregate(evaluation),
        }
        log(record)
        torch.save(
            {
                "settings": cfg,
                "layout": env.layout,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "models": [m.state_dict() for m in models],
                "student": student.state_dict(),
                "round": round_index + 1,
                "metrics": record,
            },
            args.output / f"round_{round_index + 1}.pt",
        )
        save_data(args.output / "replay.npz", episodes)
        np.savez_compressed(
            args.output / "teacher_labels.npz",
            states=np.concatenate([x for x, _ in labels]),
            actions=np.concatenate([y for _, y in labels]),
        )
    log(
        {
            "stage": "complete",
            "note": "First-stage one-pass experiment; cyclic possession is not implemented by this trainer.",
        }
    )


if __name__ == "__main__":
    main()
