"""Tests for differentiable planning and episode-safe training windows."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soccer_sim.world_model import Dynamics, Student, plan_teacher, sample_windows


def test_windows_never_cross_episode_boundary():
    episodes = [
        {"states": np.full((13, 2), i), "actions": np.full((12, 8), i)}
        for i in range(3)
    ]
    states, actions = sample_windows(episodes, 50, 8, np.random.default_rng(14))
    assert states.shape == (50, 9, 2)
    assert np.all(states == states[:, :1])
    assert np.all(actions[:, :, 0] == states[:, :8, 0])


def test_frozen_dynamics_still_provide_action_gradients():
    layout, cursor = {}, 0
    for key, size in [
        ("r0_position", 3),
        ("r1_position", 3),
        ("ball_position", 3),
        ("r0_velocity", 6),
        ("r1_velocity", 6),
        ("ball_velocity", 6),
        ("r0_rotation", 9),
        ("r1_rotation", 9),
        ("context", 3),
        ("time", 1),
        ("r0_phase", 4),
        ("r1_phase", 4),
    ]:
        layout[key] = [cursor, cursor + size]
        cursor += size
    settings = {
        "wm_hidden": 32,
        "control_dt": 0.1,
        "teacher_horizon": 3,
        "teacher_updates": 2,
        "teacher_lr": 0.1,
        "receiver_xy": [0.7, 0.7],
        "action_penalty": 0.02,
        "uncertainty_penalty": 0.5,
    }
    model = Dynamics(np.zeros(cursor), np.ones(cursor), layout, settings)
    torch.nn.init.normal_(model.net[-1].weight, std=0.01)
    model.requires_grad_(False)
    state = torch.zeros(4, cursor)
    action = torch.zeros(4, 8, requires_grad=True)
    prediction = model(state, action)
    prediction.square().mean().backward()
    assert action.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    student = Student(np.zeros(cursor), np.ones(cursor))
    teacher = plan_teacher([model, model], state, student, layout, settings)
    assert teacher.shape == (4, 8)
    assert torch.isfinite(teacher).all() and teacher.abs().max() <= 1

    # The matched ablation must not obtain future states from a dynamics model.
    class NoDynamics(torch.nn.Module):
        def forward(self, *args):
            raise AssertionError("Current-state teacher must not invoke WM")

    settings["teacher_mode"] = "current_intercept"
    state[:, slice(*layout["r1_position"])] = torch.tensor([0.7, 0.7, 0.54])
    state[:, slice(*layout["ball_position"])] = torch.tensor([0.4, 0.3, 0.095])
    state[:, slice(*layout["r1_rotation"])] = torch.eye(3).flatten()
    restricted = Student(
        np.zeros(cursor), np.ones(cursor), 0.7, [0, 0, 0, 1, 1, 0, 0, 0]
    )
    command = plan_teacher([NoDynamics()], state, restricted, layout, settings)
    assert torch.all(command[:, 3:5] < 0)
    assert torch.all(command[:, [0, 1, 2, 5, 6, 7]] == 0)
    settings.update(intercept_stop_distance=0.34, intercept_stop_speed=0.5)
    state[:, slice(*layout["ball_position"])] = torch.tensor([0.92, 0.7, 0.095])
    held_command = plan_teacher([NoDynamics()], state, restricted, layout, settings)
    assert torch.all(held_command == 0)
