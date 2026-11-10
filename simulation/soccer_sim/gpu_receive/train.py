"""Collect -> 8-step fit -> 32-step differentiable imagination, with lambda critic.

No planner, teacher, or OPD. Resume preserves optimizer/RNG and budget metadata.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .buffer import Replay
from .env import BatchEnv
from .model import (
    ImagineController,
    PhysicalWM,
    clip_command,
    context,
    goal,
    make_actor,
    receiver,
    residual,
    reward,
    value,
)


def wm_update(wm, replay, opt, batch):
    ss, aa, _ = replay.sample(batch, 8)
    pred = ss[:, 0]
    loss = 0.0
    for t in range(8):
        pred = wm(pred, aa[:, t])
        loss = loss + wm.loss(pred, ss[:, t + 1]) / 8
    label = (
        (ss[:, 1, 77] < 0.5) | (1 - 2 * ss[:, 1, 79:81].square().sum(-1) < 0.76484)
    ).float()
    loss = loss + F.binary_cross_entropy(
        wm.termination(ss[:, 1]).clamp(1e-6, 1 - 1e-6), label
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


def imagine(
    actor, wm, controller, env, replay, actoropt, criticopt, batch, micro, horizon,
    sender_control="held", matched_starts=False, sender_controller=None,
):
    actor_params = list(actor.encoder.parameters()) + list(actor.decoder.parameters())
    actoropt.zero_grad(set_to_none=True)
    wm.requires_grad_(False)
    actor.critic.requires_grad_(False)
    objectives = []
    critic_data = []
    for start in range(0, batch, micro):
        n = min(micro, batch - start)
        window = horizon if matched_starts or sender_control == "recorded" else 1
        sample = replay.sample(n, window, with_sender=sender_control == "closed_loop")
        ss, aa, h = sample[:3]
        sender_memory = sample[3] if sender_control == "closed_loop" else None
        s = ss[:, 0]
        sender = aa[:, 0, :31].detach()
        previous = s.new_zeros(n, 12)
        # h is the exact receiving observation after the real controller call.
        mod = controller.native
        raw = mod.net(h)
        last = mod.offsets[-2]
        ballhist = h[:, last:].reshape(n, 3, -1)
        held = ballhist[:, -1, :-1]
        states = []
        rs = []
        ds = []
        survival = s.new_ones(n)
        objective = s.new_zeros(n)
        for t in range(horizon):
            if sender_control == "recorded":
                sender = aa[:, t, :31].detach()
            elif sender_control == "closed_loop" and t:
                sender, sender_memory = sender_controller(s, sender_memory)
            delta = residual(actor, s)
            base = mod.default + mod.scale * raw
            recv = torch.cat([base[:, :12] + delta, base[:, 12:]], -1)
            qcmd = clip_command(s, torch.cat([sender, recv], -1), env)
            if sender_control == "recorded":
                # These are already executed, clipped controls. Re-clipping at a
                # predicted state would destroy this oracle-input diagnostic.
                qcmd = torch.cat([sender, qcmd[:, 31:]], -1)
            nxt = wm(s, qcmd)
            r = reward(nxt, delta, previous)
            d = wm.termination(nxt)
            d = 1 - (1 - d) * (nxt[:, 159] < 6).float()
            objective = objective + (0.99**t) * survival * r
            survival = survival * (1 - d)
            states.append(s.detach())
            rs.append(r.detach())
            ds.append(d.detach())
            s = nxt
            previous = delta
            if t + 1 < horizon:
                _, h, raw, held = controller(s, h, raw, held, s.new_zeros(n, 12))
        objective = objective + (0.99**horizon) * survival * value(actor, s)
        loss = -objective.mean() * (n / batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite actor objective")
        loss.backward()
        objectives.append(float(objective.detach().mean()))
        with torch.no_grad():
            boot = value(actor, s)
            targets = []
            for t in reversed(range(horizon)):
                vn = value(actor, states[t + 1]) if t + 1 < horizon else value(actor, s)
                boot = rs[t] + 0.99 * (1 - ds[t]) * (0.05 * vn + 0.95 * boot)
                targets.append(boot)
            critic_data.append(
                (torch.cat(states), torch.stack(list(reversed(targets))).flatten())
            )
    grad = torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
    actoropt.step()
    actor.critic.requires_grad_(True)
    for _ in range(3):
        criticopt.zero_grad(set_to_none=True)
        for states, targets in critic_data:
            loss = F.mse_loss(value(actor, states), targets) / len(critic_data)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.critic.parameters(), 1.0)
        criticopt.step()
    wm.requires_grad_(True)
    return float(np.mean(objectives)), float(grad)


@torch.no_grad()
def evaluate(env, actor, seed, baseline=False, wm=None, out=None, details=False):
    s = env.reset(seed)
    total = 0.0
    legacy_total = 0.0
    previous = s.new_zeros(env.n, 12)
    frames = []
    pred = None
    truth = []
    preds = []
    residual_square = s.new_zeros(())
    residual_saturated = s.new_zeros(())
    for tick in range(300):
        q = env.prepare().clone()
        d = (
            torch.zeros(env.n, 12, device=env.device)
            if baseline
            else residual(actor, s)
        )
        residual_square += d.square().mean()
        residual_saturated += (d.abs() > 0.1425).float().mean()
        q[:, 31:43] += d
        if tick == 100 and wm is not None:
            pred = s.clone()
        s = env.step(q)
        total += float(reward(s, d, previous).mean())
        legacy_total += float(reward(s, d, d).mean())
        previous = d
        if pred is not None and 100 <= tick < 150:
            pred = wm(pred, env.qcmd)
            truth.append(s.clone())
            preds.append(pred.clone())
        if out is not None and tick % 2 == 0:
            frames.append(env.qpos.clone())
    score = dict(**env.metrics(), return_mean=total,
                 legacy_return_mean=legacy_total,
                 residual_rms_rad=float((residual_square / 300).sqrt()),
                 residual_saturation_fraction=float(residual_saturated / 300))
    overflow = env.wd.overflow.numpy()
    score["capacity_overflow_max"] = int((overflow & 511).max())
    score["solver_limit_world_fraction"] = float(np.mean((overflow & 1536) != 0))
    if score["capacity_overflow_max"] != 0:
        raise RuntimeError("MuJoCo Warp contact/constraint capacity overflow")
    if preds:
        for t in [4, 24, 49]:
            score[f"ball_rmse_{t + 1}"] = float(
                (preds[t][:, 150:153] - truth[t][:, 150:153])
                .square()
                .sum(-1)
                .mean()
                .sqrt()
            )
            score[f"receiver_root_rmse_{t + 1}"] = float(
                (preds[t][:, 75:78] - truth[t][:, 75:78]).square().sum(-1).mean().sqrt()
            )
    if details:
        score["per_world"] = {
            "strict": env.strict.cpu().tolist(), "fallen": env.fallen.cpu().tolist(),
            "hold": env.hold.cpu().tolist(), "max_hold": env.max_hold.cpu().tolist(),
            "contacts": env.contacts_t.cpu().tolist(),
            "handoff": (env.handoff_hold >= .8).cpu().tolist(),
            "distance": (s[:, 150:152] - s[:, 75:77]).norm(dim=-1).cpu().tolist(),
            "speed": s[:, 153:155].norm(dim=-1).cpu().tolist(),
        }
    if out is not None:
        good = env.strict.nonzero().flatten()
        i = int(good[0]) if len(good) else 0
        path = Path(out)
        np.savez_compressed(
            path,
            qpos=torch.stack(frames)[:, i].cpu().numpy(),
            times=np.arange(len(frames)) * 0.04 + 0.02,
            world=i,
            seed=seed,
            strict_success=bool(env.strict[i]),
            fps=25,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "world": i,
                    "seed": seed,
                    "strict_success": bool(env.strict[i]),
                    "metrics": score,
                },
                indent=2,
            )
        )
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--eval-num-envs", type=int, default=512)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--micro", type=int, default=128)
    p.add_argument("--wm-width", type=int, default=2048)
    p.add_argument("--hours", type=float, default=2.5)
    p.add_argument(
        "--max-transitions",
        type=int,
        help="Optional exact real-transition cap, including warmup; must be a multiple of num-envs",
    )
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--eval-seconds", type=int, default=300)
    p.add_argument("--checkpoint-seconds", type=int, default=1800)
    p.add_argument("--njmax", type=int, default=512)
    p.add_argument("--actor-lr", type=float, help="Override WM actor LR after restoring optimizer state")
    p.add_argument("--seed", type=int, default=104)
    p.add_argument("--sender-control", choices=["held", "recorded", "closed_loop"], default="held")
    p.add_argument("--controller-kinematics", choices=["legacy", "current"], default="legacy")
    p.add_argument("--matched-imagination-starts", action="store_true")
    p.add_argument("--fresh-training-rng", action="store_true",
                   help="Use --seed after resume; continuation replications must not share saved RNG")
    p.add_argument("--new-updates", type=int, help="Exact number of additional updates after resume")
    p.add_argument("--eval-iterations", type=int, default=0,
                   help="Evaluate every N new updates instead of using a wall-clock interval")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke-horizon", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument(
        "--collect-steps",
        type=int,
        default=32,
        help="Real vector steps collected before each WM update round",
    )
    p.add_argument("--algorithm", choices=["wm", "ppo"], default="wm")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()
    if a.max_transitions is not None and (
        a.max_transitions % a.num_envs
        or a.max_transitions < a.warmup_steps * a.num_envs
    ):
        p.error("max-transitions must be a multiple of num-envs and cover warmup")
    if a.collect_steps < 1:
        p.error("collect-steps must be positive")
    if a.checkpoint_seconds < 1:
        p.error("checkpoint-seconds must be positive")
    if a.actor_lr is not None and a.actor_lr <= 0:
        p.error("actor-lr must be positive")
    if a.new_updates is not None and a.new_updates < 1:
        p.error("new-updates must be positive")
    if a.eval_iterations < 0:
        p.error("eval-iterations must be nonnegative")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision("high")
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path("configs/supermimic_g1_receive.yaml").read_text())
    cfg["gpu_njmax"] = a.njmax
    cfg["controller_kinematics"] = a.controller_kinematics
    env = BatchEnv(cfg, a.num_envs)
    # Evaluation must never reset the training world's trajectory. In particular,
    # slow model updates must not prevent the ball from reaching the receiver.
    eval_env = BatchEnv(cfg, a.eval_num_envs)
    actor = make_actor().to(env.device)
    wm = PhysicalWM(a.wm_width).to(env.device)
    controller = ImagineController(env)
    from .controller import SenderController
    sender_controller = SenderController(env) if a.sender_control == "closed_loop" else None
    if a.compile:
        controller = torch.compile(controller)
    opts = [
        torch.optim.Adam(wm.parameters(), lr=2e-4),
        torch.optim.Adam(
            list(actor.encoder.parameters()) + list(actor.decoder.parameters()), lr=2e-4
        ),
        torch.optim.Adam(actor.critic.parameters(), lr=2e-4),
    ]
    replay = Replay(a.num_envs, 256, env.device,
                    sender_memory_dim=sender_controller.memory_dim if sender_controller else 0)
    ppoopt = torch.optim.Adam(actor.parameters(), lr=2e-4)
    started = time.monotonic()
    transitions = 0
    iteration = 0
    phase_origin = {k: 0 for k in env.phase_transitions}
    episode_origin = 0
    meta = {
        **vars(a),
        "out": str(a.out),
        "resume": str(a.resume) if a.resume else None,
        "physical_state_dim": 160,
        "wm_action_dim": 62,
        "teacher": False,
        "opd": False,
        "separate_evaluation_worlds": True,
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco": __import__("mujoco").__version__,
        "warp": __import__("warp").__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cfg": cfg,
        "reward_accounting": "previous_residual_v1; legacy_return_mean retained",
        "sender_memory_dim": sender_controller.memory_dim if sender_controller else 0,
    }
    (a.out / "settings.json").write_text(json.dumps(meta, indent=2))
    shutil.copytree(
        Path(__file__).parent,
        a.out / "code",
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    def log(x):
        free, total = torch.cuda.mem_get_info()
        x.update(
            torch_allocated_gib=torch.cuda.memory_allocated() / 2**30,
            torch_reserved_gib=torch.cuda.memory_reserved() / 2**30,
            device_free_gib=free / 2**30,
            device_total_gib=total / 2**30,
            seconds=time.monotonic() - started,
            iteration=iteration,
            transitions=transitions,
            training_tick=env.tick,
            training_reset_seed=env.reset_seed,
            onpolicy_phase_transitions={
                k: env.phase_transitions[k] - phase_origin[k] for k in phase_origin
            },
            completed_onpolicy_world_episodes=max(
                0, env.completed_world_episodes - episode_origin
            ),
        )
        with (a.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(x) + "\n")
        print(json.dumps(x), flush=True)

    def render(path):
        if a.no_render:
            return
        with (a.out / "render.log").open("a") as f:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "soccer_sim.gpu_receive.render",
                    str(path),
                    "--closeup",
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "MUJOCO_GL": "egl"},
            )

    def save(name):
        target = a.out / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "actor": actor.state_dict(),
                "wm": wm.state_dict(),
                "optimizers": [o.state_dict() for o in opts],
                "ppo_optimizer": ppoopt.state_dict(),
                "iteration": iteration,
                "transitions": transitions,
                "run_seconds": time.monotonic() - started,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "settings": meta,
            },
            tmp,
        )
        tmp.replace(target)

    if a.resume:
        r = torch.load(a.resume, map_location=env.device, weights_only=False)
        actor.load_state_dict(r["actor"])
        wm.load_state_dict(r["wm"])
        for o, v in zip(opts, r["optimizers"]):
            o.load_state_dict(v)
        if "ppo_optimizer" in r:
            ppoopt.load_state_dict(r["ppo_optimizer"])
        iteration = r["iteration"]
        transitions = r["transitions"]
        torch.set_rng_state(r["torch_rng"].cpu())
        torch.cuda.set_rng_state(r["cuda_rng"].cpu())
    if a.fresh_training_rng:
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
    if a.actor_lr is not None:
        for group in opts[1].param_groups:
            group["lr"] = a.actor_lr
    meta["effective_actor_lr"] = opts[1].param_groups[0]["lr"]
    (a.out / "settings.json").write_text(json.dumps(meta, indent=2))
    if (
        a.max_transitions is not None
        and a.max_transitions < transitions + a.warmup_steps * env.n
    ):
        raise ValueError(
            "Transition budget does not cover resumed count plus fresh replay warmup"
        )
    baseline_score = evaluate(
        eval_env, actor, 220000, True, out=a.out / "baseline_demo.npz"
    )
    log({"stage": "baseline", "evaluation": baseline_score})
    save("baseline.pt")
    if a.resume:
        save("resume_start.pt")
        resume_score = evaluate(eval_env, actor, 220000, wm=wm if a.algorithm == "wm" else None)
        log({"stage": "resume_evaluation", "evaluation": resume_score})
    render(a.out / "baseline_demo.npz")
    s = env.reset(a.seed)
    for tick in range(a.warmup_steps):
        with torch.no_grad():
            q = env.prepare().clone()
            sender_memory = sender_controller.snapshot(env) if sender_controller else None
            q[:, 31:43] += (torch.randn(env.n, 12, device=env.device) * 0.015).clamp(
                -0.05, 0.05
            )
            old = s.clone()
            s = env.step(q)
            replay.add(
                old,
                env.qcmd,
                env.memory[(1, "receiving")]["history"],
                env.fallen | (env.tick >= 300),
                sender=sender_memory,
            )
            transitions += env.n
    samples = replay.s[: replay.count].reshape(-1, 160)[::8]
    if not a.resume:
        wm.norm.update(samples)
        wm.norm.var.clamp_(min=0.0025)
        actor.obs_norm.update(receiver(samples))
        actor.obs_norm.var.clamp_(min=0.0025)
        actor.goal_norm.update(goal(samples))
        actor.goal_norm.var.clamp_(min=0.0025)
        actor.critic_norm.update(
            torch.cat([receiver(samples), goal(samples), context(samples)], -1)
        )
        actor.critic_norm.var.clamp_(min=0.0025)
    del samples
    if a.algorithm == "ppo":
        # PPO never samples the dynamics replay after identical warmup/norm fitting.
        # Release its ~2.3 GB allocation without changing observations or updates.
        del replay
        torch.cuda.empty_cache()
    last_eval = 0.0
    best = resume_score["front_pass_success"] if a.resume else baseline_score["front_pass_success"]
    if a.resume:
        save("best.pt")
        (a.out / "best.json").write_text(json.dumps({"iteration": iteration,
            "evaluation": resume_score, "selection": "resumed policy before new updates"}, indent=2))
    last_save = time.monotonic()
    next_checkpoint = a.checkpoint_seconds
    phase_origin = env.phase_transitions.copy()
    episode_origin = env.completed_world_episodes + (env.n if env.tick >= 300 else 0)
    log({"stage": "warmup", "memory_gb": torch.cuda.memory_allocated() / 1e9})
    start_iteration = iteration + 1
    stop_iteration = (iteration + a.new_updates if a.new_updates is not None else a.iterations)
    start_transitions = transitions
    for next_iteration in range(start_iteration, stop_iteration + 1):
        if time.monotonic() - started >= a.hours * 3600:
            break
        remaining_steps = (
            (a.max_transitions - transitions) // env.n
            if a.max_transitions is not None
            else None
        )
        if remaining_steps is not None and remaining_steps <= 0:
            break
        iteration = next_iteration
        if a.algorithm == "ppo":
            from .ppo import update as ppo_update

            rollout_steps = (
                min(32, remaining_steps) if remaining_steps is not None else 32
            )
            s, learning = ppo_update(
                actor, env, s, ppoopt, a.batch, steps=rollout_steps
            )
            transitions += env.n * rollout_steps
        else:
            with torch.no_grad():
                collect_steps = a.collect_steps
                if remaining_steps is not None:
                    collect_steps = min(collect_steps, remaining_steps)
                for _ in range(collect_steps):
                    if env.tick >= 300:
                        s = env.reset(a.seed + iteration)
                    q = env.prepare().clone()
                    sender_memory = sender_controller.snapshot(env) if sender_controller else None
                    d = residual(actor, s)
                    q[:, 31:43] += d + torch.randn_like(d) * 0.01
                    old = s.clone()
                    s = env.step(q)
                    replay.add(
                        old,
                        env.qcmd,
                        env.memory[(1, "receiving")]["history"],
                        env.fallen | (env.tick >= 300),
                        sender=sender_memory,
                    )
                    transitions += env.n
            env.check_capacity()
            if iteration % 10 == 0:
                norm_s, _, _ = replay.sample(4096, 1)
                norm_s = norm_s[:, 0]
                wm.norm.update(norm_s)
                wm.norm.var.clamp_(min=0.0025)
                actor.obs_norm.update(receiver(norm_s))
                actor.obs_norm.var.clamp_(min=0.0025)
                actor.goal_norm.update(goal(norm_s))
                actor.goal_norm.var.clamp_(min=0.0025)
                actor.critic_norm.update(
                    torch.cat([receiver(norm_s), goal(norm_s), context(norm_s)], -1)
                )
                actor.critic_norm.var.clamp_(min=0.0025)
            wm.requires_grad_(True)
            losses = [
                wm_update(wm, replay, opts[0], a.batch)
                for _ in range(10 if not a.smoke else 1)
            ]
            objective = grad = 0.0
            if iteration > 10 or a.smoke:
                objective, grad = imagine(
                    actor,
                    wm,
                    controller,
                    env,
                    replay,
                    opts[1],
                    opts[2],
                    a.batch,
                    a.micro,
                    32 if not a.smoke else a.smoke_horizon,
                    sender_control=a.sender_control,
                    matched_starts=a.matched_imagination_starts,
                    sender_controller=sender_controller,
                )
        if a.algorithm == "ppo":
            losses = [learning["ppo_loss"]]
            objective = learning["rollout_reward"]
            grad = 0.0
        if iteration % 5 == 0 or a.smoke:
            log(
                {
                    "stage": "train",
                    "wm_loss": float(np.mean(losses)),
                    "objective": objective,
                    "actor_grad_norm": grad,
                    "memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "training_capacity": env.check_capacity(),
                }
            )
        new_updates = iteration - start_iteration + 1
        evaluate_now = (new_updates % a.eval_iterations == 0 if a.eval_iterations
                        else time.monotonic() - started - last_eval >= a.eval_seconds)
        if evaluate_now:
            save("latest.pt")
            score = evaluate(
                eval_env,
                actor,
                220000,
                wm=wm if a.algorithm == "wm" else None,
                out=a.out / f"monitor_{iteration}.npz",
            )
            log({"stage": "evaluation", "evaluation": score})
            last_eval = time.monotonic() - started
            if score["front_pass_success"] > best:
                best = score["front_pass_success"]
                save("best.pt")
                render(a.out / f"monitor_{iteration}.npz")
                (a.out / "best.json").write_text(
                    json.dumps({"iteration": iteration, "evaluation": score}, indent=2)
                )
        elapsed = time.monotonic() - started
        if elapsed >= next_checkpoint:
            name = f"checkpoint_{next_checkpoint:06d}s_iter_{iteration:07d}.pt"
            save(name)
            log({"stage": "checkpoint", "checkpoint": name,
                 "scheduled_seconds": next_checkpoint})
            next_checkpoint = (int(elapsed // a.checkpoint_seconds) + 1) * a.checkpoint_seconds
        if time.monotonic() - last_save > 300:
            save("latest.pt")
            last_save = time.monotonic()
        if a.smoke:
            break
    save("final.pt")
    log(
        {
            "stage": "complete",
            "completed_new_updates": max(0, iteration - start_iteration + 1),
            "new_training_transitions": transitions - start_transitions,
            "matched_budget_completed": a.new_updates is None or iteration >= stop_iteration,
            "evaluation": evaluate(
                eval_env,
                actor,
                220000,
                wm=wm if a.algorithm == "wm" else None,
                out=a.out / "final_demo.npz",
            ),
        }
    )
    render(a.out / "final_demo.npz")


if __name__ == "__main__":
    main()
