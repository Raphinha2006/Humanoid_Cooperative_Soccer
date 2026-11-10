"""Matched residual-policy baseline, PPO with no world model or teacher."""

import torch

from .model import goal, receiver, reward, value


def update(actor, env, s, opt, batch=2048, steps=32, epochs=4):
    states = []
    raws = []
    logps = []
    rewards = []
    values = []
    dones = []
    std = 0.15
    previous = s.new_zeros(env.n, 12)
    for _ in range(steps):
        if env.tick >= 300:
            s = env.reset(104 + env.total_steps)
        with torch.no_grad():
            mean = actor.act(receiver(s), goal(s))
            dist = torch.distributions.Normal(mean, std)
            raw = dist.sample()
            delta = 0.15 * raw.tanh()
            q = env.prepare().clone()
            q[:, 31:43] += delta
            states.append(s)
            values.append(value(actor, s))
            raws.append(raw)
            logps.append(dist.log_prob(raw).sum(-1))
            s = env.step(q)
            rewards.append(reward(s, delta, previous))
            dones.append(env.fallen | (env.tick >= 300))
            previous = delta
    with torch.no_grad():
        v = value(actor, s)
        adv = torch.zeros_like(v)
        advantages = []
        for t in reversed(range(steps)):
            nextv = v if t == steps - 1 else values[t + 1]
            td = rewards[t] + 0.99 * (~dones[t]) * nextv - values[t]
            adv = td + 0.99 * 0.95 * (~dones[t]) * adv
            advantages.append(adv)
        advantages = torch.stack(list(reversed(advantages)))
        returns = (advantages + torch.stack(values)).flatten()
        adv = advantages.flatten()
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
        ss = torch.cat(states)
        rr = torch.cat(raws)
        lp = torch.cat(logps)
    for _ in range(epochs):
        for ids in torch.randperm(len(ss), device=ss.device).split(batch):
            mean = actor.act(receiver(ss[ids]), goal(ss[ids]))
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(rr[ids]).sum(-1)
            ratio = (logp - lp[ids]).exp()
            loss = (
                -torch.minimum(
                    ratio * adv[ids], ratio.clamp(0.8, 1.2) * adv[ids]
                ).mean()
                + 0.5 * (value(actor, ss[ids]) - returns[ids]).square().mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
    return s, {
        "ppo_loss": float(loss.detach()),
        "rollout_reward": float(torch.stack(rewards).mean()),
    }
