"""Physics-state/gradient correctness; GPU throughput is verified separately."""

from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
from soccer_sim.gpu_receive.model import (
    PhysicalWM,
    context,
    goal,
    make_actor,
    receiver,
    residual,
)
from soccer_sim.gpu_receive.native import Native
from soccer_sim.joga_env import JogaPassEnv
from soccer_sim.supermimic_core.kinematics import (
    DifferentiableKinematics,
    build_kinematic_tree,
)

ROOT = Path(__file__).resolve().parents[1]


def test_training_capacity_guard_rejects_overflow_but_reports_solver_limit():
    from types import SimpleNamespace
    from soccer_sim.gpu_receive.env import BatchEnv

    env = BatchEnv.__new__(BatchEnv)
    flags = np.array([0, 512], dtype=np.int32)
    env.wd = SimpleNamespace(overflow=SimpleNamespace(numpy=lambda: flags))
    assert env.check_capacity() == {
        "capacity_overflow_max": 0, "solver_limit_world_fraction": 0.5}
    flags[0] = 2
    with pytest.raises(RuntimeError, match="capacity overflow"):
        env.check_capacity()


def test_native_observation_matches_cpu_public_interface():
    cfg = yaml.safe_load((ROOT / "configs/supermimic_g1_receive.yaml").read_text())
    env = JogaPassEnv(cfg)
    s = env.reset(0)
    p = env.skills[1]
    mem = p.mods["receiving"]
    net = Native(env.spec, "receiving")
    history = torch.tensor(
        np.concatenate([h.ravel() for h in mem["history"]]), dtype=torch.float32
    )[None]
    previous = torch.tensor(mem["previous"], dtype=torch.float32)[None]
    held = torch.tensor(mem["box"], dtype=torch.float32)[None]
    state = torch.tensor(np.r_[s[75:150], s[150:159]])[None]
    q, h, _, _ = net.observation(
        state,
        torch.tensor(env.d.xpos[p.cam], dtype=torch.float32)[None],
        torch.tensor(env.d.xquat[p.cam], dtype=torch.float32)[None],
        history,
        previous,
        torch.zeros(1, 3),
        torch.zeros(1),
        held,
    )
    expected = p.infer(env.d, "receiving", [0, 0, 0], env.ball_body)
    actual_hist = np.concatenate([h.ravel() for h in mem["history"]])
    np.testing.assert_allclose(h.detach().numpy()[0], actual_hist, atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(q.detach().numpy()[0], expected, atol=1e-5, rtol=1e-5)


def test_wm_compact_state_and_action_gradients():
    wm = PhysicalWM(32)
    s = torch.randn(4, 160) * 0.01
    s[:, 3] = 1
    s[:, 78] = 1
    s[:, 77] = 0.76
    a = torch.zeros(4, 62, requires_grad=True)
    wm.requires_grad_(False)
    p = wm(s, a)
    assert p.shape == (4, 160)
    for off in [0, 75]:
        torch.testing.assert_close(p[:, off + 3 : off + 7].norm(dim=-1), torch.ones(4))
    p[:, 150:153].sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
    assert wm.norm.mean.shape == (160,)
    target = p.detach().clone()
    target[:, 3:7] *= -1
    target[:, 78:82] *= -1
    assert abs(float(wm.loss(p.detach(), target))) < 1e-5


def test_actor_and_critic_shapes_and_native_initialization():
    actor = make_actor()
    s = torch.zeros(3, 160)
    s[:, 3] = 1
    s[:, 78] = 1
    s[:, 77] = 0.75
    torch.testing.assert_close(residual(actor, s), torch.zeros(3, 12))
    assert actor.value(receiver(s), goal(s), context(s)).shape == (3,)


def test_g1_full_body_and_camera_chain_fk_match_mujoco():
    import mujoco

    cfg = yaml.safe_load((ROOT / "configs/supermimic_g1_receive.yaml").read_text())
    env = JogaPassEnv(cfg)
    env.reset(225103)
    skill = env.skills[1]
    env.d.qpos[skill.q] += np.linspace(-0.07, 0.07, 31)
    mujoco.mj_forward(env.m, env.d)
    s = torch.tensor(env.state())[None]
    joints = [env.m.joint(i).name.rsplit("/", 1)[-1] for i in skill.j]
    full = [i for i in range(env.m.nbody) if env.m.body(i).name.startswith("A2/")]
    chain = []
    i = skill.cam
    while i:
        chain.append(i)
        i = int(env.m.body_parentid[i])
    for ids in [full, sorted(chain)]:
        names = [env.m.body(i).name for i in ids]
        fk = DifferentiableKinematics(build_kinematic_tree(env.m, names, joints))
        pos, quat = fk(s[:, 75:78], s[:, 78:82], s[:, 88:119])
        np.testing.assert_allclose(pos[0].numpy(), env.d.xpos[ids], atol=3e-6)
        # Quaternions q and -q represent the same orientation.
        dot = (quat[0].numpy() * env.d.xquat[ids]).sum(-1)
        np.testing.assert_allclose(np.abs(dot), 1, atol=2e-6)
