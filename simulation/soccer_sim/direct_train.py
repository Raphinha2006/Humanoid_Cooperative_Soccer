"""G1/K1 SuperMimic adaptation: collect -> fit -> 32-step BPTT + lambda critic.

No planning teacher or distillation targets. Simulator transitions are authoritative.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .direct_env import DirectPassEnv
from .direct_model import (
    DirectDynamics,
    act,
    context,
    goal,
    make_actor,
    make_fk,
    reward,
    sl,
)
from .skill_policies import source_manifest
from .wm_train import sample_windows, save_data


def collect(env, actor, seeds, rng, cfg, noise=0.0):
    out = []
    for seed in seeds:
        s = env.reset(int(seed))
        states, actions = [s], []
        done = False
        while not done:
            with torch.no_grad():
                a = (
                    np.zeros(20)
                    if actor is None
                    else act(actor, torch.from_numpy(s)[None], env.layout, cfg)[
                        0
                    ].numpy()
                )
            a = np.clip(a + rng.normal(0, noise, 20), -1, 1) * np.asarray(
                cfg["action_mask"]
            )
            a = a.astype(np.float32)
            s, done = env.step(a)
            states.append(s)
            actions.append(a)
        out.append(
            {
                "states": np.stack(states),
                "actions": np.stack(actions),
                "seed": int(seed),
                "metrics": env.metrics(),
            }
        )
    return out


def summarize(episodes):
    keys = [
        "pass_success",
        "front_pass_success",
        "captured",
        "fallen",
        "final_receiver_distance",
        "final_ball_speed",
    ]
    return {k: float(np.mean([e["metrics"][k] for e in episodes])) for k in keys}


@torch.no_grad()
def validate(model, episodes, rng, cfg, device, layout, fk, feet):
    s, a = sample_windows(episodes, 64, 50, np.random.default_rng(9147))
    truth = torch.tensor(s, device=device)
    action = torch.tensor(a, device=device)
    pred = truth[:, 0]
    out = {}
    for t in range(50):
        pred = model(pred, action[:, t])
        if t + 1 in [5, 25, 50]:
            for name in ["ball_position", "r1_position", "r1_q"]:
                diff = pred[:, sl(layout, name)] - truth[:, t + 1, sl(layout, name)]
                out[f"{name}_rmse_{t + 1}"] = float(diff.square().sum(-1).mean().sqrt())
            pp, _ = fk(
                pred[:, sl(layout, "r1_position")],
                pred[:, sl(layout, "r1_quaternion")],
                pred[:, sl(layout, "r1_q")],
            )
            tt = truth[:, t + 1]
            tp, _ = fk(
                tt[:, sl(layout, "r1_position")],
                tt[:, sl(layout, "r1_quaternion")],
                tt[:, sl(layout, "r1_q")],
            )
            out[f"receiver_body_rmse_{t + 1}"] = float(
                (pp - tp).square().sum(-1).mean().sqrt()
            )
    return out


def update(model, actor, fk, feet, episodes, optim, rng, cfg, device, layout):
    wmopt, actoropt, criticopt = optim
    model.requires_grad_(True)
    for _ in range(cfg["wm_updates"]):
        ss, aa = sample_windows(episodes, cfg["batch_size"], 8, rng)
        ss = torch.tensor(ss, device=device)
        aa = torch.tensor(aa, device=device)
        pred = ss[:, 0]
        loss = ss.new_zeros(())
        for t in range(8):
            pred = model(pred, aa[:, t])
            loss = loss + model.loss(pred, ss[:, t + 1]) / 8
        # Soft fall head; time limits are treated as truncations, never falls.
        fall = []
        for i in range(2):
            p = ss[:, 1, sl(layout, f"r{i}_position")]
            q = ss[:, 1, sl(layout, f"r{i}_quaternion")]
            fall.append(
                (p[:, 2] < cfg["fall_height"])
                | (1 - 2 * q[:, 1:3].square().sum(-1) < cfg["fall_upright"])
            )
        label = (fall[0] | fall[1]).float()
        termloss = F.binary_cross_entropy(
            model.termination(ss[:, 1]).clamp(1e-6, 1 - 1e-6), label
        )
        loss = loss + termloss
        wmopt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        wmopt.step()
    wm_loss = float(loss.detach())
    model.requires_grad_(False)
    actor_params = list(actor.encoder.parameters()) + list(actor.decoder.parameters())
    for _ in range(cfg["actor_updates"]):
        ss, aa = sample_windows(episodes, cfg["imagination_batch"], 1, rng)
        state = torch.tensor(ss[:, 0], device=device)
        previous = torch.tensor(aa[:, 0], device=device)
        actor.critic.requires_grad_(False)
        states, rs, ds = [state], [], []
        survival = torch.ones(len(state), device=device)
        objective = state.new_zeros(len(state))
        for t in range(cfg["horizon"]):
            a = act(actor, state, layout, cfg)
            nxt = model(state, a)
            r = reward(nxt, a, previous, layout, cfg, fk, feet)
            d = model.termination(nxt)
            # End-of-episode time is deterministic and distinct from a learned fall.
            d = (
                1
                - (1 - d)
                * (nxt[:, sl(layout, "time")].squeeze(-1) < cfg["duration"]).float()
            )
            objective = objective + (cfg["gamma"] ** t) * survival * r
            survival = survival * (1 - d)
            states.append(nxt)
            rs.append(r)
            ds.append(d)
            state = nxt
            previous = a
        objective = objective + (
            cfg["gamma"] ** cfg["horizon"]
        ) * survival * actor.value(state, goal(state, layout), context(state, cfg))
        actor_loss = -objective.mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("non-finite imagined objective")
        actoropt.zero_grad()
        actor_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError("non-finite actor gradient")
        actoropt.step()
        # SuperMimic lambda-return critic: detach targets, preserve critic input
        # derivatives only in the actor phase above.
        actor.critic.requires_grad_(True)
        with torch.no_grad():
            states = [s.detach() for s in states]
            rs = [r.detach() for r in rs]
            ds = [d.detach() for d in ds]
            vals = [actor.value(s, goal(s, layout), context(s, cfg)) for s in states]
            running = vals[-1]
            targets = []
            for t in reversed(range(cfg["horizon"])):
                running = rs[t] + cfg["gamma"] * (1 - ds[t]) * (
                    (1 - cfg["lambda"]) * vals[t + 1] + cfg["lambda"] * running
                )
                targets.append(running)
            y = torch.stack(list(reversed(targets))).flatten()
            x = torch.cat(states[:-1])
        for _ in range(cfg["critic_epochs"]):
            value = actor.value(x, goal(x, layout), context(x, cfg))
            critic_loss = F.mse_loss(value, y)
            criticopt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.critic.parameters(), 1.0)
            criticopt.step()
    return {
        "wm_loss": wm_loss,
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "actor_grad_norm": float(grad),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/supermimic_k1_receive.yaml",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(
            rounds=1,
            initial_episodes=3,
            episodes_per_round=2,
            validation_episodes=2,
            evaluation_episodes=2,
            wm_updates=2,
            actor_updates=1,
            batch_size=8,
            imagination_batch=4,
            critic_epochs=1,
        )
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        Path(__file__).parent,
        args.out / "code",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (args.out / "code_sha256.json").write_text(
        json.dumps(
            {
                str(f.relative_to(args.out / "code")): hashlib.sha256(
                    f.read_bytes()
                ).hexdigest()
                for f in (args.out / "code").rglob("*.py")
            },
            indent=2,
        )
    )
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(2)
    rng = np.random.default_rng(cfg["seed"])
    if cfg.get("embodiment") == "joga_g1":
        from .joga_env import JogaPassEnv

        env = JogaPassEnv(cfg)
    else:
        env = DirectPassEnv(cfg)
    layout = env.layout
    cfg["physics_context"] = (
        env.physics_context
        if hasattr(env, "physics_context")
        else [
            float(env.m.pair_friction[0, 0]),
            float(env.m.body_mass[env.env.ball_body]),
            env.env.config["ball"]["radius"],
        ]
    )
    (args.out / "settings.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (args.out / "environment.yaml").write_text(
        yaml.safe_dump(env.env.config, sort_keys=False)
    )
    (args.out / "state_layout.json").write_text(json.dumps(layout, indent=2))
    (args.out / "runtime.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "mujoco": mujoco.__version__,
                "device": args.device,
                "skills": json.loads(
                    (
                        Path(__file__).resolve().parents[1]
                        / "configs/joga_sources.json"
                    ).read_text()
                )
                if cfg.get("embodiment") == "joga_g1"
                else source_manifest(cfg["kick_kind"]),
                "supermimic_core": json.loads(
                    (Path(__file__).parent / "supermimic_core/SOURCE.json").read_text()
                ),
                "method": "50 Hz collect-fit-imagine, 8-step fit, 32-step BPTT, lambda-return critic; humanoid receiving residual adaptation",
                "official_training_code": False,
            },
            indent=2,
        )
    )
    mujoco.mj_saveLastXML(str(args.out / "runtime_scene.xml"), env.m)
    (args.out / "runtime_actuation.json").write_text(
        json.dumps(
            [
                {
                    "joints": [env.m.joint(j).name for j in s.j],
                    "kp": s.kp.tolist(),
                    "kd": s.kd.tolist(),
                    "limits": s.limit.tolist(),
                }
                for s in env.skills
            ],
            indent=2,
        )
    )
    actor = make_actor(env.size, cfg["actor_hidden"])
    model = DirectDynamics(env.size, layout, cfg["control_dt"], cfg["wm_hidden"]).to(
        args.device
    )
    fk, feet = make_fk(env)
    fk = fk.to(args.device)
    started = time.time()
    path = args.out / "metrics.jsonl"

    def log(item):
        item["wall_seconds"] = round(time.time() - started, 2)
        with path.open("a") as f:
            f.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)

    validation = collect(
        env, None, range(110000, 110000 + cfg["validation_episodes"]), rng, cfg, 0.15
    )
    baseline = collect(
        env, None, range(120000, 120000 + cfg["evaluation_episodes"]), rng, cfg
    )
    log({"stage": "baseline", "evaluation": summarize(baseline)})
    (args.out / "baseline_episodes.json").write_text(
        json.dumps([e["metrics"] for e in baseline], indent=2)
    )
    episodes = collect(
        env,
        None,
        range(cfg["seed"], cfg["seed"] + cfg["initial_episodes"]),
        rng,
        cfg,
        cfg["exploration_std"],
    )
    real = torch.tensor(
        np.concatenate([e["states"][:-1] for e in episodes]), device=args.device
    )
    model.norm.update(real)
    model.norm.var.clamp_(min=0.0025)
    actor = actor.to(args.device)
    actor.obs_norm.update(real)
    actor.obs_norm.var.clamp_(min=0.0025)
    actor.goal_norm.update(goal(real, layout))
    actor.goal_norm.var.clamp_(min=0.0025)
    actor.critic_norm.update(
        torch.cat((real, goal(real, layout), context(real, cfg)), -1)
    )
    actor.critic_norm.var.clamp_(min=0.0025)
    log(
        {
            "stage": "initial",
            "validation": validate(
                model, validation, rng, cfg, args.device, layout, fk, feet
            ),
        }
    )
    optim = (
        torch.optim.Adam(model.parameters(), lr=cfg["wm_lr"]),
        torch.optim.Adam(
            list(actor.encoder.parameters()) + list(actor.decoder.parameters()),
            lr=cfg["actor_lr"],
        ),
        torch.optim.Adam(actor.critic.parameters(), lr=cfg["critic_lr"]),
    )
    next_seed = cfg["seed"] + cfg["initial_episodes"]
    best = summarize(baseline)["front_pass_success"]
    selected = "baseline"
    for round_ in range(1, cfg["rounds"] + 1):
        if round_ > 1:
            actor.cpu()
            new = collect(
                env,
                actor,
                range(next_seed, next_seed + cfg["episodes_per_round"]),
                rng,
                cfg,
                cfg["exploration_std"],
            )
            next_seed += cfg["episodes_per_round"]
            episodes.extend(new)
            actor.to(args.device)
        # Bounded replay keeps recent distributions; retain all raw episodes on disk at checkpoints.
        training = episodes[-cfg["replay_episodes"] :]
        learning = update(
            model, actor, fk, feet, training, optim, rng, cfg, args.device, layout
        )
        rec = {
            "stage": "round",
            "round": round_,
            "episodes": len(episodes),
            "transitions": sum(len(e["actions"]) for e in episodes),
            "learning": learning,
        }
        if round_ % cfg["eval_interval"] == 0 or round_ == 1 or round_ == cfg["rounds"]:
            rec["validation"] = validate(
                model, validation, rng, cfg, args.device, layout, fk, feet
            )
            actor.cpu()
            ev = collect(
                env, actor, range(120000, 120000 + cfg["evaluation_episodes"]), rng, cfg
            )
            rec["evaluation"] = summarize(ev)
            (args.out / f"evaluation_{round_}.json").write_text(
                json.dumps([e["metrics"] for e in ev], indent=2)
            )
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "model": model.state_dict(),
                    "settings": cfg,
                    "layout": layout,
                    "size": env.size,
                    "round": round_,
                },
                args.out / f"round_{round_}.pt",
            )
            score = (
                rec["evaluation"]["front_pass_success"]
                if rec["evaluation"]["fallen"] == 0
                else -1
            )
            if score > best:
                best = score
                selected = f"round_{round_}.pt"
            actor.to(args.device)
        log(rec)
    save_data(args.out / "replay.npz", episodes)
    save_data(args.out / "validation.npz", validation)
    (args.out / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint": selected,
                "rule": "monitor front_pass_success, zero falls; earliest tie; baseline allowed; only evaluated checkpoints",
            },
            indent=2,
        )
    )
    log({"stage": "complete", "rounds": cfg["rounds"], "selected": selected})


if __name__ == "__main__":
    main()
