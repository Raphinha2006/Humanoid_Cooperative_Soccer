"""Measure total GPU process memory through one full-size training update."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--njmax", type=int, default=512)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, "-u", "-m", "soccer_sim.gpu_receive.train",
           "--out", str(a.out / "train"), "--resume", "runs/g1_wm_11h_gpu0_20261110/wm/best.pt",
           "--iterations", "142", "--hours", "0.1", "--num-envs", "4096",
           "--eval-num-envs", "512", "--batch", "2048", "--micro", "128",
           "--wm-width", "2048", "--collect-steps", "32", "--eval-seconds", "999999",
           "--njmax", str(a.njmax)]
    peak = 0
    with (a.out / "train.log").open("w") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu), "PYTHONPATH": ".",
                 "OPENBLAS_NUM_THREADS": "1", "MUJOCO_GL": "egl"})
        while True:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10)
            values = [int(line.split(",")[1]) for line in result.stdout.splitlines()
                      if line.split(",")[0].strip() == str(child.pid)]
            used = sum(values)
            peak = max(peak, used)
            with (a.out / "memory.jsonl").open("a") as f:
                f.write(json.dumps({"time": time.time(), "pid": child.pid, "process_mib": used}) + "\n")
            try:
                rc = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                pass
    (a.out / "status.json").write_text(json.dumps({"returncode": rc, "finished": time.time(),
        "process_peak_mib": peak, "njmax": a.njmax, "command": cmd}, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
