"""Regression: development evaluation must not interrupt a training ball trajectory."""

from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

from soccer_sim.gpu_receive.env import BatchEnv
from soccer_sim.gpu_receive.train import evaluate


def test_development_evaluation_preserves_training_episode():
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "configs/supermimic_g1_receive.yaml"
        ).read_text()
    )
    training = BatchEnv(cfg, 32)
    development = BatchEnv(cfg, 32)
    training.reset(225100)
    for _ in range(180):
        training.step(training.prepare().clone())
    qpos, qvel = training.qpos.clone(), training.qvel.clone()
    contacts = training.contacts_t.clone()
    history = training.memory[(1, "receiving")]["history"].clone()
    phases = training.phase_transitions.copy()
    evaluate(development, None, 225101, baseline=True)
    assert training.tick == 180
    assert training.phase_transitions == phases
    for actual, expected in [
        (training.qpos, qpos),
        (training.qvel, qvel),
        (training.contacts_t, contacts),
        (training.memory[(1, "receiving")]["history"], history),
    ]:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for _ in range(120):
        training.step(training.prepare().clone())
    assert training.tick == 300
    assert training.phase_transitions["post_kick"] > phases["post_kick"] > 0
    training.reset(225102)
    assert training.completed_world_episodes == 32
