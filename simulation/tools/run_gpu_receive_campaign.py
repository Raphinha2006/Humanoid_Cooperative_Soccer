"""Finite GPU1 experiment campaign, process-backed status, no job restarts."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hours", type=float, default=2.5)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": ".",
        "MUJOCO_GL": "egl",
    }
    status = {
        "pid": os.getpid(),
        "started": time.time(),
        "gpu_physical_index": 1,
        "runs": [],
    }
    for algo in ["wm", "ppo"]:
        out = a.out / algo
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "soccer_sim.gpu_receive.train",
            "--algorithm",
            algo,
            "--out",
            str(out),
            "--hours",
            str(a.hours),
            "--num-envs",
            "4096",
            "--batch",
            "2048",
            "--micro",
            "128",
            "--wm-width",
            "2048",
            "--collect-steps",
            "32",
            "--eval-seconds",
            "300",
        ]
        with (a.out / f"{algo}.log").open("w") as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            rec = {
                "algorithm": algo,
                "pid": child.pid,
                "started": time.time(),
                "command": cmd,
                "status": "running",
            }
            status["runs"].append(rec)
            (a.out / "status.json").write_text(json.dumps(status, indent=2))
            rc = child.wait()
            rec.update(
                returncode=rc,
                finished=time.time(),
                status="complete" if rc == 0 else "failed",
            )
            (a.out / "status.json").write_text(json.dumps(status, indent=2))
            if rc:
                sys.exit(rc)
    status["finished"] = time.time()
    status["status"] = "training_complete_pending_independent_evaluation"
    (a.out / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
