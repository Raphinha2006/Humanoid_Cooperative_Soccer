import mujoco
import numpy as np
import pytest
from soccer_sim.config import DEFAULT_CONFIG, load_config
from soccer_sim.policy_runner import run
from soccer_sim.policy_scene import MODEL_PATHS, build_policy_scene
from soccer_sim.pretrained import POLICIES, BoosterVelocityPolicy, SonicMotionPolicy


def test_config_inheritance_preserves_physics_and_rejects_cycles(tmp_path):
    base = load_config()
    aligned = load_config(DEFAULT_CONFIG.with_name("hsl_m_pretrained_pass.yaml"))
    assert aligned["contacts"] == base["contacts"]
    assert aligned["field"] == base["field"]
    assert aligned["robot"]["instances"][1]["position"][1] == pytest.approx(-0.0262)
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.skipif(
    not MODEL_PATHS["booster_mjlab"].exists(), reason="fetch_policies.py required"
)
def test_pretrained_scene_keeps_free_roots_and_applies_physical_overrides():
    base = load_config()
    m = mujoco.MjModel.from_xml_string(build_policy_scene(base, "booster_mjlab"))
    changed = load_config(
        overrides=["robot.mass_scale=1.2", "pretrained.joint_damping=0.1"]
    )
    n = mujoco.MjModel.from_xml_string(build_policy_scene(changed, "booster_mjlab"))
    assert m.nu == 132 and m.nq == 6 * 29 + 7 and m.nv == 6 * 28 + 6
    p = BoosterVelocityPolicy(m, "A1/")
    assert np.isfinite(p.network(np.zeros(75))).all()
    np.testing.assert_allclose(
        n.body_mass[m.body("A1/Trunk").id], m.body_mass[m.body("A1/Trunk").id] * 1.2
    )
    ids = [
        b
        for b in range(m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("A1/")
    ]
    np.testing.assert_allclose(n.body_mass[ids], m.body_mass[ids] * 1.2)
    np.testing.assert_allclose(
        n.body_inertia[ids], m.body_inertia[ids] * 1.2, rtol=1e-7
    )
    np.testing.assert_allclose(n.dof_damping[p.v], 0.1)
    assert m.pair_friction[m.pair("ball_turf").id, 3] == pytest.approx(
        base["contacts"]["ball_turf"]["friction"][3]
    )


@pytest.mark.skipif(
    not MODEL_PATHS["booster_mjlab"].exists(), reason="fetch_policies.py required"
)
def test_six_free_robot_policy_contact_trial_and_reloadable_initial_state(tmp_path):
    c = load_config(DEFAULT_CONFIG.with_name("hsl_m_pretrained_pass.yaml"))
    r = run("booster_mjlab", c, tmp_path, scenario="pass", seconds=12)
    assert r["all_robots_no_fall"] and r["mujoco_warning_count"] == 0
    assert r["ball_contact_robots"] == ["A1", "A2"]
    assert r["ball_first_contact_time_s"]["A1"] < r["ball_first_contact_time_s"]["A2"]
    assert not r["is_pretrained_pass_policy"]
    assert r["pass_success"] == (
        r["receive_event_success"] and r["stable_receive_at_end"]
    )
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("policy_initial").id)
    initial = np.load(tmp_path / "initial_state.npz", allow_pickle=False)
    np.testing.assert_allclose(d.qpos, initial["qpos"], atol=1e-6)
    np.testing.assert_allclose(d.ctrl, initial["ctrl"], atol=1e-6)


@pytest.mark.skipif(
    not (POLICIES / "sonic_k1/checkpoints/sonic-k1/onnx/policy_g1.onnx").exists(),
    reason="SONIC weights required",
)
def test_sonic_shares_network_but_keeps_each_robots_history_separate():
    pytest.importorskip("onnxruntime")
    m = mujoco.MjModel.from_xml_string(build_policy_scene(load_config(), "sonic_k1"))
    clip = POLICIES / "sonic_k1/examples/k1/motions/squat.npz"
    a = SonicMotionPolicy(m, "A1/", clip)
    b = SonicMotionPolicy(m, "A2/", clip, shared=a)
    assert a.policy.s is b.policy.s
    args = [np.zeros(3), a.plant.default, np.zeros(22), np.array([0, 0, -1])]
    assert a.policy.proprio(*args).shape == (720,)
    b.policy.proprio(*args)
    assert all(not np.shares_memory(x, y) for x, y in zip(a.policy.hist, b.policy.hist))
    cmd, ori = a.ref.command(0, a.initial_qpos[3:7])
    assert len(cmd) + len(ori) + 720 == 1220
