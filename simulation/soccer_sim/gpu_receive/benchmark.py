import argparse
import json
import time
from pathlib import Path

import torch
import warp as wp
import yaml

from .env import BatchEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(Path("configs/supermimic_g1_receive.yaml").read_text())
    e = BatchEnv(cfg, a.num_envs)
    begin = time.time()
    for k in range(a.steps):
        e.step(e.prepare())
        if (k + 1) % 50 == 0:
            print(json.dumps({"step": k + 1, **e.metrics()}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - begin
    overflow = e.wd.overflow.numpy()
    result = {
        "nworld": a.num_envs,
        "steps": a.steps,
        "seconds": elapsed,
        "transitions_per_second": a.num_envs * a.steps / elapsed,
        "overflow_max": int(overflow.max()),
        "metrics": e.metrics(),
        "mujoco": "3.12.0",
        "warp": wp.__version__,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
