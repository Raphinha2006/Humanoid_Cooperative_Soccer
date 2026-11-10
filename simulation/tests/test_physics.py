import numpy as np
import pytest
from soccer_sim.cli import run_passing
from soccer_sim.config import load_config
from soccer_sim.env import SoccerEnv


def test_full_scene_has_six_articulated_robots_and_shell_ball():
    e = SoccerEnv(load_config())
    assert e.model.nu == 6 * 22
    assert e.model.nq == 6 * 22 + 7
    c = e.config
    b = e.ball_body
    assert e.model.body_mass[b] == pytest.approx(c["ball"]["mass"])
    expected = (
        c["ball"]["inertia_factor"] * c["ball"]["mass"] * c["ball"]["radius"] ** 2
    )
    np.testing.assert_allclose(e.model.body_inertia[b], [expected] * 3)
    assert sum(e.model.body_mass) > 100


def test_pass_requires_real_sender_and_receiver_contact(tmp_path):
    r = run_passing(load_config(), tmp_path)
    assert r["success"] and r["sender_contact"] and r["receiver_contact"]
    assert r["mujoco_warning_count"] == 0
    assert r["final_ball_position_m"][0] > 1.5
    assert r["final_ball_speed_mps"] < 0.05


def test_no_kick_does_not_count_as_pass(tmp_path):
    c = load_config(overrides=["passing.kick_start_time=20", "passing.timeout=4"])
    r = run_passing(c, tmp_path)
    assert not r["success"] and not r["sender_contact"]
    assert abs(r["final_ball_position_m"][0] - c["ball"]["initial_position"][0]) < 0.005


def test_rolling_friction_is_effective_and_hot_change_preserves_state():
    distances = []
    for roll in [0.0, 0.006]:
        e = SoccerEnv(load_config(), robots=False)
        q = e.data.qpos.copy()
        v = e.data.qvel.copy()
        e.set_contact("ball_turf", friction=[0.65, 0.65, 0.002, roll, roll])
        np.testing.assert_array_equal(q, e.data.qpos)
        np.testing.assert_array_equal(v, e.data.qvel)
        radius = e.config["ball"]["radius"]
        e.reset(ball_position=[0, 2, radius + 0.001], ball_velocity=[1, 0, 0])
        e.data.qvel[e.ball_v + 4] = 1 / radius
        for _ in range(2000):
            e.step()
        distances.append(e.observe()["ball_position"][0])
    assert distances[0] > distances[1] + 0.2


def test_parameter_override_and_invalid_values():
    c = load_config(overrides=["ball.mass=0.35", "field.length=15.0"])
    assert c["ball"]["mass"] == 0.35 and c["field"]["length"] == 15.0
    for item in [
        "ball.mass=-1",
        "physics.timestep=0.1",
        "contacts.ball_turf.friction=[0.5,0.1]",
        "ball.radius=.nan",
    ]:
        with pytest.raises(ValueError):
            load_config(overrides=[item])
    with pytest.raises(KeyError):
        load_config(overrides=["ball.typo=1"])


def test_full_ball_boundary_and_goal_opening():
    e = SoccerEnv(load_config(), robots=False)
    r = e.config["ball"]["radius"]
    e.reset(ball_position=[7 + r - 0.001, 3, r])
    assert e.ball_status() == "in"
    e.reset(ball_position=[7 + r + 0.001, 3, r])
    assert e.ball_status() == "out"
    e.reset(ball_position=[7 + r + 0.001, 0, r])
    assert e.ball_status() == "goal_positive_x"
    e.reset(ball_position=[7 + r + 0.001, 0, 1.7])
    assert e.ball_status() == "out"


def test_free_base_configuration_exposes_floating_dofs():
    e = SoccerEnv(load_config(overrides=["robot.base_mode=free"]))
    assert e.model.nq == 6 * 29 + 7
    assert e.model.nv == 6 * 28 + 6
    assert e.model.nu == 6 * 22
