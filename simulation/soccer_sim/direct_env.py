"""50 Hz facing K1 task with explicit receiving joint residuals and compact state."""

import numpy as np

from .wm_env import SkillPassEnv


class DirectPassEnv(SkillPassEnv):
    action_dim = 20

    def state(self):
        pieces = []

        def add(name, value):
            a = np.asarray(value).ravel()
            start = sum(len(x) for x in pieces)
            self.layout[name] = [start, start + len(a)]
            pieces.append(a)

        for i, p in enumerate(self.skills):
            q = self.root_q[i]
            add(f"r{i}_position", self.d.qpos[q : q + 3])
            add(f"r{i}_quaternion", self.d.qpos[q + 3 : q + 7])
            add(f"r{i}_velocity", self.d.qvel[p.root_v : p.root_v + 6])
            add(f"r{i}_q", self.d.qpos[p.q])
            add(f"r{i}_qd", self.d.qvel[p.v])
        add("ball_position", self.d.qpos[self.env.ball_q : self.env.ball_q + 3])
        add("ball_velocity", self.d.qvel[self.env.ball_v : self.env.ball_v + 6])
        add("time", [self.elapsed])
        for i, p in enumerate(self.skills):
            for name, value in [
                ("target", p.target),
                ("phase", np.r_[np.cos(p.phase), np.sin(p.phase)]),
                ("history", p.walk_history),
                ("blend_from", p.blend_from),
                ("blend_time", [p.blend_time]),
                ("memory", [p.kick_frequency, *p.old_ball]),
                ("last_action", p.last_action),
                ("skill", [p.previous_skill == "kick"]),
            ]:
                add(f"r{i}_{name}", value)
        self.size = sum(len(x) for x in pieces)
        return np.concatenate(pieces).astype(np.float32)

    def metrics(self):
        m = super().metrics()
        p = self.skills[1]
        rot = self.d.xmat[p.body].reshape(3, 3)
        ball = self.d.qpos[self.env.ball_q : self.env.ball_q + 3]
        rel = rot.T @ (ball - self.d.xpos[p.body])
        m["ball_in_front_at_end"] = bool(0.03 < rel[0] < 0.4 and abs(rel[1]) < 0.25)
        m["front_pass_success"] = m["pass_success"] and m["ball_in_front_at_end"]
        m["receiver_ball_local"] = rel.tolist()
        return m
