"""Render saved exact GPU qpos sequence; no re-simulation or cherry-picked motion edits."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw

from ..joga_env import JogaPassEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory", type=Path)
    p.add_argument("--closeup", action="store_true")
    a = p.parse_args()
    cfg = yaml.safe_load(Path("configs/supermimic_g1_receive.yaml").read_text())
    e = JogaPassEnv(cfg)
    z = np.load(a.trajectory)
    out = a.trajectory.with_name(
        a.trajectory.stem + ("_closeup" if a.closeup else "") + ".mp4"
    )
    e.m.vis.global_.offwidth = 960
    e.m.vis.global_.offheight = 720
    renderer = mujoco.Renderer(e.m, 720, 960)
    cam = mujoco.MjvCamera()
    cam.azimuth = 90
    cam.elevation = -22
    cam.distance = 2.8 if a.closeup else 8.5
    with imageio.get_writer(str(out), fps=25, codec="libx264") as w:
        for t, q in enumerate(z["qpos"]):
            e.d.qpos[:] = q
            mujoco.mj_forward(e.m, e.d)
            cam.lookat[:] = (
                e.d.xpos[e.skills[1].body] + [0.1, 0, -0.25]
                if a.closeup
                else [2.5, 0, 0.65]
            )
            renderer.update_scene(e.d, camera=cam)
            im = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 960, 44], fill=(18, 25, 35))
            d.text(
                (12, 12),
                f"G1 GPU physical trajectory | t={float(z['times'][t]) if 'times' in z else t / 25 + 0.02:.2f}s | world={int(z['world'])} | episode strict success={bool(z['strict_success'])}",
                fill=(255, 255, 255),
            )
            w.append_data(np.array(im))
    renderer.close()
    print(json.dumps({"video": str(out), "frames": len(z["qpos"])}), flush=True)


if __name__ == "__main__":
    main()
