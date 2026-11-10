"""Native-controller state and replay-boundary regression checks."""

from pathlib import Path
import json

import pytest
import yaml

torch = pytest.importorskip("torch")

from soccer_sim.gpu_receive.buffer import Replay
from soccer_sim.gpu_receive.model import reward


def test_sender_memory_uses_the_same_start_as_actions_and_histories():
    torch.manual_seed(7)
    replay = Replay(4, 64, "cpu", sender_memory_dim=5)
    for tick in range(80):
        replay.add(torch.full((4, 160), float(tick)),
                   torch.full((4, 62), float(tick)),
                   torch.full((4, 324), float(tick)),
                   torch.zeros(4, dtype=torch.bool),
                   sender=torch.full((4, 5), float(tick)))
    s, a, h, sender = replay.sample(16, 32, with_sender=True)
    torch.testing.assert_close(s[:, :-1, :62], a)
    torch.testing.assert_close(s[:, 0, 0], h[:, 0])
    torch.testing.assert_close(s[:, 0, 0], sender[:, 0])
    torch.testing.assert_close(s[:, 1:, 0] - s[:, :-1, 0], torch.ones(16, 32))


def test_sender_memory_is_required_and_reset_windows_are_rejected():
    replay = Replay(2, 40, "cpu", sender_memory_dim=3)
    with pytest.raises(ValueError, match="memory required"):
        replay.add(torch.zeros(2, 160), torch.zeros(2, 62),
                   torch.zeros(2, 324), torch.ones(2, dtype=torch.bool))
    for _ in range(40):
        replay.add(torch.zeros(2, 160), torch.zeros(2, 62),
                   torch.zeros(2, 324), torch.ones(2, dtype=torch.bool),
                   sender=torch.zeros(2, 3))
    with pytest.raises(RuntimeError, match="valid replay windows"):
        replay.sample(8, 32, with_sender=True)


def test_previous_action_accounting_is_observable_in_reward():
    s = torch.zeros(2, 160)
    s[:, 3] = s[:, 78] = 1.
    s[:, 77] = .76
    s[:, 150] = .35
    d = torch.full((2, 12), .1)
    corrected = reward(s, d, torch.zeros_like(d))
    legacy = reward(s, d, d)
    torch.testing.assert_close(legacy - corrected, torch.full((2,), .02 * .01 / .0225))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_closed_loop_sender_matches_every_real_phase_and_preserves_gradients():
    from soccer_sim.gpu_receive.controller import SenderController
    from soccer_sim.gpu_receive.env import BatchEnv
    from soccer_sim.gpu_receive.model import ImagineController

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          "configs/supermimic_g1_receive.yaml").read_text())
    cfg["gpu_njmax"] = 1024
    cfg["controller_kinematics"] = "current"
    env = BatchEnv(cfg, 16)
    env.reset(225200)
    sender = SenderController(env)
    receiving = ImagineController(env)
    target = env.prepare().clone()
    memory = sender.snapshot(env)
    rm = env.memory[(1, "receiving")]
    rh, raw, held = rm["history"].clone(), rm["previous"].clone(), rm["held"].clone()
    commands, memories = [], []
    receiver_errors = []
    for _ in range(299):
        with torch.no_grad():
            state = env.step(target)
            target = env.prepare().clone()
            predicted, advanced = sender(state, memory)
            rq, rh, raw, held = receiving(state, rh, raw, held, state.new_zeros(env.n, 12))
            receiver_errors.append((rq - target[:, 31:]).abs().max())
            expected = sender.snapshot(env)
            commands.append((predicted - target[:, :31]).abs().max())
            memories.append((advanced - expected).abs().max())
            # No teacher forcing: advance the reconstructed memory for the full episode.
            memory = advanced
    assert float(torch.stack(commands).max()) <= 1e-4
    assert float(torch.stack(memories).max()) <= 1e-3
    assert float(torch.stack(receiver_errors).max()) <= 1e-4
    env.check_capacity()
    state = env.physical().detach().requires_grad_(True)
    predicted, _ = sender(state, memory.detach())
    predicted.square().mean().backward()
    assert torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    print(json.dumps({"conformance": "passed", "steps": 299, "worlds": env.n,
                      "sender_command_max_error_rad": float(torch.stack(commands).max()),
                      "sender_memory_max_error": float(torch.stack(memories).max()),
                      "receiver_command_max_error_rad": float(torch.stack(receiver_errors).max()),
                      "finite_state_gradients": True}))
