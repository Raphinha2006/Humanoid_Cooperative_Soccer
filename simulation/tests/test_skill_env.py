"""Integration invariants for the actual released walk/kick control path."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from soccer_sim.wm_env import SkillPassEnv

ASSETS = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ASSETS / "policies/nubots/LICENCE").exists()
    or not (ASSETS / "policies/bhuman/Config/NeuralNets/Walk/K1_walk.onnx").exists(),
    reason="Fetch pinned skill assets first",
)


def test_floating_skill_episode_is_deterministic_and_has_no_support_forces():
    env = SkillPassEnv()
    assert sum(env.m.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 3  # two roots plus ball
    recorded = []
    for _ in range(2):
        state = env.reset(104)
        for _ in range(15):
            state, _ = env.step(np.zeros(8))
        recorded.append(state)
        # Only the ball may receive aerodynamic drag; robot support forces stay zero.
        robots = [i for robot in env.link_ids for i in robot]
        assert np.all(env.d.xfrc_applied[robots] == 0)
        assert np.all(env.d.qfrc_applied == 0)
        assert env.skills[0].previous_skill == "kick"
        assert env.skills[1].previous_skill == "walk"
        assert env.skills[0].last_action is not env.skills[1].last_action
        for i, bodies in enumerate(env.link_ids):
            start, stop = env.layout[f"r{i}_body_positions"]
            np.testing.assert_allclose(
                state[start:stop].reshape(-1, 3), env.d.xpos[bodies], atol=1e-6
            )
    np.testing.assert_array_equal(*recorded)
    assert env.touched[0]


def test_rejects_control_period_incompatible_with_native_policies():
    import yaml
    from soccer_sim.wm_env import DEFAULT

    settings = yaml.safe_load(DEFAULT.read_text())
    settings["control_dt"] = 0.03
    with pytest.raises(ValueError, match="multiple of 0.02"):
        SkillPassEnv(settings)


def test_mutual_facing_is_measured_after_physics_and_rejects_reversed_receiver():
    import yaml

    if not (ASSETS / "policies/bhuman/Config/NeuralNets/Walk/K1_walk.onnx").exists():
        pytest.skip("Fetch B-Human exports first")
    settings = yaml.safe_load((ASSETS / "configs/wm_pass_facing.yaml").read_text())
    env = SkillPassEnv(settings)
    env.reset(104)
    assert env.facing_valid()
    p, q = env.skills[1], env.root_q[1]
    assert env.d.xmat[env.skills[0].body].reshape(3, 3)[0, 0] > 0.98
    assert env.d.xmat[p.body].reshape(3, 3)[0, 0] < -0.98
    for _ in range(10):
        _, done = env.step(np.zeros(8))
        assert not done
    assert env.facing_valid()
    # A receiver turned away cannot qualify for facing reception even nearby.
    env.d.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(env.m, env.d)
    assert not env.facing_valid()
