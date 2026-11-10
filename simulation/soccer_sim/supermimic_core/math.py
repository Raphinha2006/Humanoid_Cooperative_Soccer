"""Standalone differentiable wxyz quaternion utilities (no mjlab dependency)."""

import torch


def quat_mul(a, b):
    w, v = a[..., :1], a[..., 1:]
    u, z = b[..., :1], b[..., 1:]
    return torch.cat(
        (
            w * u - (v * z).sum(-1, keepdim=True),
            w * z + u * v + torch.linalg.cross(v, z),
        ),
        -1,
    )


def quat_inv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1) / q.square().sum(
        -1, keepdim=True
    ).clamp_min(1e-8)


def quat_apply(q, v):
    uv = torch.linalg.cross(q[..., 1:], v)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(q[..., 1:], uv))


def quat_from_angle_axis(angle, axis):
    half = angle.unsqueeze(-1) / 2
    return torch.cat((half.cos(), half.sin() * axis), -1)
