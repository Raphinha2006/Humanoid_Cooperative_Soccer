"""Open-loop joint-space kick; receiver stays in a compliant standing pose."""

import numpy as np


def blend(a, b, u):
    u = np.clip(u, 0.0, 1.0)
    u = u * u * (3 - 2 * u)
    return {key: float(a[key] + u * (b[key] - a[key])) for key in a}


class PassingController:
    def __init__(self, env):
        self.env = env
        self.c = env.config["passing"]
        keys = set(self.c["backswing"]) | set(self.c["strike"])
        self.home = {k: env.config["robot"]["home"].get(k, 0.0) for k in keys}
        self.back = {k: self.c["backswing"].get(k, self.home[k]) for k in keys}
        self.strike = {k: self.c["strike"].get(k, self.home[k]) for k in keys}
        self.phase = "settle"

    def update(self):
        self.env.set_joint_targets(self.c["receiver"], self.c["receiver_pose"])
        p = self.c
        t = self.env.data.time - p["kick_start_time"]
        if t < 0:
            targets = self.home
            self.phase = "settle"
        elif t < p["backswing_duration"]:
            targets = blend(self.home, self.back, t / p["backswing_duration"])
            self.phase = "backswing"
        elif t < p["backswing_duration"] + p["strike_duration"]:
            targets = blend(
                self.back,
                self.strike,
                (t - p["backswing_duration"]) / p["strike_duration"],
            )
            self.phase = "strike"
        else:
            targets = blend(
                self.strike,
                self.home,
                (t - p["backswing_duration"] - p["strike_duration"])
                / p["recover_duration"],
            )
            self.phase = "receive"
        self.env.set_joint_targets(p["sender"], targets)
