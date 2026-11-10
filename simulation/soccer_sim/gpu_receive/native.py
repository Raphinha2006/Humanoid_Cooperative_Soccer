"""Batched, differentiable public Joga inference. History is explicit, not learned dynamics."""

import numpy as np
import torch
from torch import nn

from ..joga import ASSETS, JMLP
from ..supermimic_core.math import quat_apply, quat_inv


class NativeMLP(nn.Module):
    def __init__(self, spec):
        super().__init__()
        src = JMLP(ASSETS / spec["weights"])
        self.register_buffer("mean", torch.tensor(src.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(src.std, dtype=torch.float32))
        self.eps = src.eps
        self.net = nn.Sequential()
        for i, (w, b) in enumerate(src.layers):
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data.copy_(torch.tensor(w))
            layer.bias.data.copy_(torch.tensor(b))
            self.net.append(layer)
            if i + 1 < len(src.layers):
                self.net.append(nn.ELU())
        self.requires_grad_(False)

    def forward(self, x):
        return self.net((x - self.mean) / (self.std + self.eps))


class Native(nn.Module):
    def __init__(self, spec, name):
        super().__init__()
        self.spec = spec
        self.s = spec["modules"][name]
        self.net = NativeMLP(self.s)
        for n, v in [("default", self.s["default_q"]), ("scale", spec["action_scale"])]:
            self.register_buffer(n, torch.tensor(v, dtype=torch.float32))
        self.widths = [3, 3, 31, 31, 31, 3, 1, 4 if self.s["ball"] == "point" else 5]
        self.offsets = np.cumsum([0] + [3 * w for w in self.widths]).tolist()

    def observation(
        self, s, cam_pos, cam_quat, history, previous, command, yaw_offset, held,
        history_valid=None,
    ):
        q = s[:, 3:7]
        gravity = quat_apply(
            quat_inv(q),
            torch.tensor([0.0, 0.0, -1.0], device=s.device).expand(len(s), -1),
        )
        yaw = (
            torch.atan2(
                2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
            )
            - yaw_offset
        )
        yaw = torch.atan2(yaw.sin(), yaw.cos())[:, None]
        point = quat_apply(quat_inv(cam_quat), s[:, 75:78] - cam_pos)
        x, y, z = point.unbind(-1)
        k = self.spec["intrinsics"]
        r = self.s.get("bbox_radius", k["ball_radius"])
        hu = torch.asin(
            (r / torch.sqrt(x * x + y * y).clamp_min(r + 1e-6)).clamp(max=1 - 1e-6)
        )
        hv = torch.asin(
            (r / torch.sqrt(x * x + z * z).clamp_min(r + 1e-6)).clamp(max=1 - 1e-6)
        )
        pu, pv = torch.atan2(y, x), torch.atan2(z, x)
        lim = np.deg2rad(89)
        us = (
            torch.stack(
                [
                    k["cx"] - k["fx"] * torch.tan((pu - hu).clamp(-lim, lim)),
                    k["cx"] - k["fx"] * torch.tan((pu + hu).clamp(-lim, lim)),
                ],
                -1,
            )
            .sort(-1)
            .values
        )
        vs = (
            torch.stack(
                [
                    k["cy"] - k["fy"] * torch.tan((pv - hv).clamp(-lim, lim)),
                    k["cy"] - k["fy"] * torch.tan((pv + hv).clamp(-lim, lim)),
                ],
                -1,
            )
            .sort(-1)
            .values
        )
        vis = (
            (x > 0.05)
            & (us[:, 1] >= 0)
            & (us[:, 0] < k["width"])
            & (vs[:, 1] >= 0)
            & (vs[:, 0] < k["height"])
        )
        if self.s["ball"] == "bbox":
            vis = vis & (point.norm(dim=-1) <= self.s["max_detect_dist"])
        box = torch.stack(
            [
                us[:, 0].clamp(0, k["width"] - 1),
                vs[:, 0].clamp(0, k["height"] - 1),
                us[:, 1].clamp(0, k["width"] - 1),
                vs[:, 1].clamp(0, k["height"] - 1),
            ],
            -1,
        )
        current = point if self.s["ball"] == "point" else box
        held = torch.where(vis[:, None], current, held)
        terms = [
            s[:, 10:13],
            gravity,
            s[:, 13:44] - self.default,
            s[:, 44:75],
            previous,
            command,
            yaw,
            torch.cat([held, vis[:, None]], -1),
        ]
        initial = None
        if history is None or history_valid is not None:
            initial = torch.cat(
                [t[:, None].expand(-1, 3, -1).reshape(len(s), -1) for t in terms], -1
            )
        if history is None:
            history = initial
        else:
            history = torch.cat(
                [
                    torch.cat(
                        [
                            history[:, self.offsets[i] : self.offsets[i + 1]].reshape(
                                len(s), 3, -1
                            )[:, 1:],
                            t[:, None],
                        ],
                        1,
                    ).flatten(1)
                    for i, t in enumerate(terms)
                ],
                -1,
            )
            if history_valid is not None:
                history = torch.where(history_valid[:, None], history, initial)
        raw = self.net(history)
        return self.default + self.scale * raw, history, raw, held
