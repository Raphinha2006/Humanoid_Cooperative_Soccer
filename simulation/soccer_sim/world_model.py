"""Differentiable skill-conditioned dynamics and an on-policy planning teacher.

This is a small task-specific adaptation, not a reproduction of SuperMimic's G1
motion tracker. Gradients cross this learned model, never MuJoCo or ONNX Runtime.
"""

import numpy as np
import torch
from torch import nn


def mlp(nin, nout, hidden):
    return nn.Sequential(
        nn.Linear(nin, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, nout),
    )


def indices(layout, key):
    return slice(*layout[key])


class Dynamics(nn.Module):
    def __init__(self, mean, std, layout, settings):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = mlp(len(mean) + 8, len(mean), settings["wm_hidden"])
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.layout, self.settings = layout, settings

    def forward(self, state, action):
        delta = (
            self.net(torch.cat(((state - self.mean) / self.std, action), -1)) * self.std
        )
        result = state + delta
        dt = self.settings["control_dt"]
        # A transparent kinematic prior, corrected by learned contact dynamics.
        for prefix in ["r0", "r1", "ball"]:
            p, v = (
                indices(self.layout, prefix + "_position"),
                indices(self.layout, prefix + "_velocity"),
            )
            result[..., p] = result[..., p] + dt * state[..., v][..., :3]
        result[..., indices(self.layout, "context")] = state[
            ..., indices(self.layout, "context")
        ]
        result[..., indices(self.layout, "time")] = (
            state[..., indices(self.layout, "time")] + dt
        )
        # The phase clock belongs to the frozen skill controller, not the physics.
        for prefix in ["r0", "r1"]:
            if prefix + "_kick_memory" in self.layout:
                # B-Human controls its frequency; learn the aggregate phase change.
                continue
            ph = state[..., indices(self.layout, prefix + "_phase")]
            angle = torch.as_tensor(2 * np.pi * 1.5 * dt, device=state.device)
            result[..., indices(self.layout, prefix + "_phase")] = torch.cat(
                (
                    ph[..., :2] * angle.cos() - ph[..., 2:] * angle.sin(),
                    ph[..., 2:] * angle.cos() + ph[..., :2] * angle.sin(),
                ),
                -1,
            )
        return result


class Student(nn.Module):
    def __init__(self, mean, std, action_limit=1.0, action_mask=None):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = mlp(len(mean), 8, 128)
        self.action_limit = float(action_limit)
        self.action_mask = [1.0] * 8 if action_mask is None else action_mask
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return (
            self.action_limit
            * self.net((state - self.mean) / self.std).tanh()
            * state.new_tensor(self.action_mask)
        )


def task_cost(state, action, layout, settings):
    ball = state[..., indices(layout, "ball_position")]
    receiver = state[..., indices(layout, "r1_position")]
    # A receiving pocket in front of the receiver's feet (both robots face +x).
    pocket = receiver[..., :2] + state.new_tensor([0.16, 0.0])
    distance = ((ball[..., :2] - pocket) ** 2).sum(-1)
    cost = distance
    for i in range(2):
        pos = state[..., indices(layout, f"r{i}_position")]
        rot = state[..., indices(layout, f"r{i}_rotation")]
        cost = cost + 10 * (0.45 - pos[..., 2]).relu() ** 2 + 5 * (1 - rot[..., 8]) ** 2
    # Do not let the planner gain by walking the receiver arbitrarily into the ball.
    receiver0 = state.new_tensor(settings["receiver_xy"])
    cost = cost + 0.15 * ((receiver[..., :2] - receiver0) ** 2).sum(-1)
    return cost + settings["action_penalty"] * (action**2).mean(-1)


def plan_teacher(models, state, student, layout, settings):
    """Optimize bounded commands on the student's visited states; no expert rollouts."""
    horizon = settings["teacher_horizon"]
    action_limit = student.action_limit
    for model in models:
        model.requires_grad_(False)
    if settings.get("teacher_mode") in ["intercept", "current_intercept"]:
        # Learned future states followed by a structured interception controller.
        with torch.no_grad():
            predicted = [state for _ in models]
            steps = (
                0
                if settings.get("teacher_mode") == "current_intercept"
                else settings.get("intercept_steps", 3)
            )
            for _ in range(steps):
                predicted = [
                    model(s, student(s)) for model, s in zip(models, predicted)
                ]
            future = torch.stack(predicted).mean(0)
            ball = 0.5 * (
                future[..., indices(layout, "ball_position")]
                + state[..., indices(layout, "ball_position")]
            )
            receiver = 0.5 * (
                future[..., indices(layout, "r1_position")]
                + state[..., indices(layout, "r1_position")]
            )
            rotation = state[..., indices(layout, "r1_rotation")].reshape(-1, 3, 3)
            relative = torch.bmm(
                rotation.transpose(1, 2), (ball - receiver).unsqueeze(-1)
            ).squeeze(-1)
            relative[..., 0] -= settings.get("intercept_pocket_x", 0.16)
            deadband = settings.get("intercept_deadband", 0.0)
            relative = torch.where(
                relative.abs() < deadband, torch.zeros_like(relative), relative
            )
            result = torch.zeros((len(state), 8), device=state.device)
            limit = settings.get("intercept_command_limit", action_limit)
            result[..., 3:5] = (
                settings.get("intercept_gain", 1.5) * relative[..., :2]
            ).clamp(-limit, limit)
            if settings.get("intercept_stop_distance", 0.0):
                current_ball = state[..., indices(layout, "ball_position")]
                current_receiver = state[..., indices(layout, "r1_position")]
                speed = state[..., indices(layout, "ball_velocity")][..., :2].norm(
                    dim=-1
                )
                close = (current_ball[..., :2] - current_receiver[..., :2]).norm(
                    dim=-1
                ) < settings["intercept_stop_distance"]
                stop = close & (speed < settings.get("intercept_stop_speed", 0.5))
                result[stop, 3:5] = 0
            return result
    with torch.no_grad():
        initial = (student(state) / action_limit).clamp(-0.95, 0.95).atanh()
    raw = initial[:, None, :].repeat(1, horizon, 1).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=settings["teacher_lr"])
    for model in models:
        model.requires_grad_(False)
    for _ in range(settings["teacher_updates"]):
        trajectories = [state for _ in models]
        loss = 0.0
        for t in range(horizon):
            action = action_limit * raw[:, t].tanh()
            trajectories = [model(s, action) for model, s in zip(models, trajectories)]
            costs = torch.stack(
                [task_cost(s, action, layout, settings) for s in trajectories]
            )
            ball = torch.stack(
                [s[..., indices(layout, "ball_position")] for s in trajectories]
            )
            uncertainty = ball.var(0, unbiased=False).sum(-1)
            for i in range(2):
                body = torch.stack(
                    [s[..., indices(layout, f"r{i}_position")] for s in trajectories]
                )
                uncertainty = uncertainty + body.var(0, unbiased=False).sum(-1)
            uncertainty = uncertainty / settings.get("uncertainty_scale", 1.0) ** 2
            loss = (
                loss
                + (0.97**t)
                * (costs.mean(0) + settings["uncertainty_penalty"] * uncertainty).mean()
            )
        loss = loss + 0.1 * ((raw.tanh() - initial[:, None, :].tanh()) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 5.0)
        optimizer.step()
    return (action_limit * raw[:, 0].tanh()).detach()


def sample_windows(episodes, count, horizon, rng):
    valid = [e for e in episodes if len(e["actions"]) >= horizon]
    if not valid:
        raise ValueError("No episode long enough for the world-model window")
    chosen = rng.integers(0, len(valid), count)
    states, actions = [], []
    for i in chosen:
        episode = valid[i]
        start = rng.integers(0, len(episode["actions"]) - horizon + 1)
        states.append(episode["states"][start : start + horizon + 1])
        actions.append(episode["actions"][start : start + horizon])
    return np.stack(states), np.stack(actions)
