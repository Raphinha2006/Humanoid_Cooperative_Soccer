"""Validate native G1 model/inference and common WM interfaces before training."""

from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
from soccer_sim.direct_model import DirectDynamics, make_fk, sl
from soccer_sim.joga_env import JogaPassEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def env():
    return JogaPassEnv(
        yaml.safe_load((ROOT / "configs/supermimic_g1_receive.yaml").read_text())
    )


def test_g1_is_native_31_joint_and_policies_match_interface(env):
    assert env.m.nq == 83 and env.m.nv == 80
    for p in env.skills:
        assert len(p.j) == 31
        for mod in p.mods.values():
            assert mod["net"].dim == mod["spec"]["obs_dim"]
            a = mod["net"](np.zeros(mod["net"].dim))
            assert a.shape == (31,) and np.isfinite(a).all()
    assert len(env.skills[1].mods) == 1 and "receiving" in env.skills[1].mods


def test_g1_fk_and_dynamic_quaternion_layout(env):
    s = torch.tensor(env.reset(0))[None]
    fk, _feet = make_fk(env)
    pos, _ = fk(
        s[:, sl(env.layout, "r1_position")],
        s[:, sl(env.layout, "r1_quaternion")],
        s[:, sl(env.layout, "r1_q")],
    )
    np.testing.assert_allclose(pos.numpy()[0], env.d.xpos[env.link_ids[1]], atol=2e-6)
    model = DirectDynamics(env.size, env.layout, 0.02, 32)
    future = model(s, torch.zeros((1, 20)))
    assert future.shape == s.shape
    for i in range(2):
        torch.testing.assert_close(
            future[:, sl(env.layout, f"r{i}_quaternion")].norm(dim=-1), torch.ones(1)
        )


def test_g1_shared_ball_is_untouched_outside_reset(env):
    results = []
    for _ in range(2):
        env.reset(0)
        start = env.d.qpos[env.ball_q : env.ball_q + 3].copy()
        for _ in range(100):
            env.step(np.zeros(20))
        assert np.linalg.norm(env.d.qpos[env.ball_q : env.ball_q + 2] - start[:2]) > 0.5
        assert env.first_contact_times[0] is not None
        assert np.all(env.d.xfrc_applied == 0) and np.all(env.d.qfrc_applied == 0)
        assert env.facing_valid()
        results.append(env.state())
    np.testing.assert_array_equal(results[0], results[1])


def test_g1_residual_reaches_leg_targets_and_physics(env):
    env.reset(0)
    baseline, _ = env.step(np.zeros(20))
    target = env.skills[1].target.copy()
    env.reset(0)
    a = np.r_[np.zeros(8), np.full(12, 0.5)]
    changed, _ = env.step(a)
    np.testing.assert_allclose(
        env.skills[1].target[:12] - target[:12],
        np.array(env.settings["receiver_joint_residual_scale"]) * 0.5,
        atol=1e-7,
    )
    assert np.linalg.norm(changed[:160] - baseline[:160]) > 1e-5
    assert np.all(env.d.xfrc_applied == 0) and np.all(env.d.qfrc_applied == 0)
