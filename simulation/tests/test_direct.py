"""Check K1 physics/FK and policy gradients of the direct SuperMimic adapter."""

from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
from soccer_sim.direct_env import DirectPassEnv
from soccer_sim.direct_model import DirectDynamics, act, make_actor, make_fk, reward, sl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def task():
    cfg = yaml.safe_load((ROOT / "configs/supermimic_k1_receive.yaml").read_text())
    env = DirectPassEnv(cfg)
    return env, cfg


def test_reused_fk_matches_real_k1_bodies(task):
    env, _cfg = task
    state = torch.tensor(env.reset(104))[None]
    fk, feet = make_fk(env)
    pos, _ = fk(
        state[:, sl(env.layout, "r1_position")],
        state[:, sl(env.layout, "r1_quaternion")],
        state[:, sl(env.layout, "r1_q")],
    )
    np.testing.assert_allclose(
        pos.detach().numpy()[0], env.d.xpos[env.link_ids[1]], atol=2e-6
    )
    assert len(feet) == 2 and env.size == 914
    assert len(env.skills[1].j) == 22
    assert env.skills[1].bhuman_walk.get_inputs()[0].shape[-1] == 344
    assert env.skills[1].bhuman_walk.get_outputs()[0].shape[-1] == 13


def test_receiving_has_real_joint_authority_and_no_support_force(task):
    env, cfg = task
    env.reset(104)
    env.step(np.zeros(20))
    zero = env.skills[1].target.copy()
    env.reset(104)
    a = np.zeros(20)
    a[8:] = 0.5
    env.step(a)
    np.testing.assert_allclose(
        env.skills[1].target[10:] - zero[10:],
        np.array(cfg["receiver_joint_residual_scale"]) * 0.5,
        atol=1e-6,
    )
    robots = [j for ids in env.link_ids for j in ids]
    assert np.all(env.d.xfrc_applied[robots] == 0)
    assert np.all(env.d.qfrc_applied == 0)
    assert env.facing_valid()


def test_wm_quaternions_and_frozen_model_keep_action_gradients(task):
    env, cfg = task
    state = torch.tensor(env.reset(104))[None].repeat(2, 1)
    model = DirectDynamics(env.size, env.layout, 0.02, 32)
    actor = make_actor(env.size, 32)
    fk, feet = make_fk(env)
    model.requires_grad_(False)
    action = act(actor, state, env.layout, cfg)
    future = model(state, action)
    grad = torch.autograd.grad(
        future[:, sl(env.layout, "ball_position")].sum(), action, retain_graph=True
    )[0]
    assert grad.abs().sum() > 0
    loss = -reward(
        future, action, torch.zeros_like(action), env.layout, cfg, fk, feet
    ).mean()
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert (
        sum(
            p.grad.abs().sum() for p in actor.decoder.parameters() if p.grad is not None
        )
        > 0
    )
    for i in range(2):
        torch.testing.assert_close(
            future[:, sl(env.layout, f"r{i}_quaternion")].norm(dim=-1), torch.ones(2)
        )
