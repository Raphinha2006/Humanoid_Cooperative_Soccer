"""Concurrent same-GPU single-variable8h experiment, followed by held-out tests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_gpu_receive_longrun import gpu_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--hours", type=float, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    resource = gpu_snapshot(1)
    if resource["free_mib"] < 14 * 1024:
        raise RuntimeError("Need14GiB free on GPU1 before starting both arms")
    state = {"pid": os.getpid(), "started": time.time(), "status": "starting",
             "hours_per_arm": a.hours, "gpu": 1, "num_envs_per_arm": 2048,
             "batch_per_arm": 1024, "resume": str(a.resume),
             "checkpoint_sha256": hashlib.sha256(a.resume.read_bytes()).hexdigest(),
             "heldout_seed": 240000, "heldout_worlds": 1024, "arms": {}}
    children = {}

    def save():
        state["heartbeat"] = time.time()
        tmp = a.out / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(a.out / "status.json")

    def read_status(name):
        path = a.out / name / "status.json"
        return json.loads(path.read_text()) if path.exists() else {}

    save()
    try:
        for name, lr in [("control", 2e-4), ("low_actor_lr", 5e-5)]:
            cmd = [sys.executable, "-u", "tools/run_gpu_receive_longrun.py",
                   "--out", str(a.out / name), "--resume", str(a.resume),
                   "--hours", str(a.hours), "--gpu", "1", "--njmax", "1024",
                   "--num-envs", "2048", "--batch", "1024", "--actor-lr", str(lr),
                   "--min-free-gib", "7"]
            with (a.out / f"{name}_supervisor.log").open("w") as log:
                child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            children[name] = child
            state["arms"][name] = {"supervisor_pid": child.pid, "actor_lr": lr, "command": cmd}
        state["status"] = "training"
        save()
        while True:
            done = True
            for name, child in children.items():
                state["arms"][name]["state"] = read_status(name)
                rc = child.poll()
                state["arms"][name]["returncode"] = rc
                if rc is None:
                    done = False
                elif rc != 0:
                    raise RuntimeError(f"{name} supervisor failed: code{rc}")
            save()
            if done:
                break
            time.sleep(30)
        state["status"] = "training_complete_independent_validation"
        save()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONPATH": ".", "MUJOCO_GL": "egl"}
        summary = {}
        acceptance = {}
        initial_reference = None
        for name in children:
            run = Path(read_status(name)["active_run"])
            rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            dev = [r for r in rows if r["stage"] == "evaluation"]
            recent = [r for r in dev if r["seconds"] >= dev[-1]["seconds"] - 1800] if dev else []
            rates = [r["evaluation"]["front_pass_success"] for r in recent]
            summary[name] = {"active_run": str(run), "last30min_evaluations": len(rates),
                "last30min_strict_range": [min(rates), max(rates)] if rates else None,
                "last30min_strict_mean": sum(rates) / len(rates) if rates else None,
                "note": "Development monitor, one training seed. Not proof of convergence."}
            for checkpoint in ["best.pt", "final.pt"]:
                cmd = [sys.executable, "-u", "-m", "soccer_sim.gpu_receive.evaluate", str(run),
                       "--checkpoint", checkpoint, "--seed", "240000", "--num-envs", "1024"]
                with (a.out / f"{name}_{Path(checkpoint).stem}_acceptance.log").open("w") as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=1800)
                result = json.loads((run / f"independent_240000_{Path(checkpoint).stem}.json").read_text())
                summary[name][checkpoint] = result[-1]["mean"]
                acceptance[(name, checkpoint)] = result[-1]["per_world"]["strict"]
                import numpy as np
                with np.load(run / f"test_initial_240000_{Path(checkpoint).stem}.npz") as data:
                    initial = (data["qpos"].copy(), data["qvel"].copy())
                if initial_reference is None:
                    initial_reference = initial
                elif not all(np.array_equal(a, b) for a, b in zip(initial_reference, initial)):
                    raise RuntimeError("Independent physical initial states differ across arms")
        comparisons = {}
        rng = np.random.default_rng(240001)
        for checkpoint in ["best.pt", "final.pt"]:
            difference = (np.array(acceptance[("low_actor_lr", checkpoint)], dtype=float)
                          - np.array(acceptance[("control", checkpoint)], dtype=float))
            bootstrap = difference[rng.integers(0, len(difference), (4096, len(difference)))].mean(1)
            comparisons[checkpoint] = {"low_lr_minus_control": float(difference.mean()),
                "initial_state_bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                "warning": "One training seed and nondeterministic GPU physics; CI covers sampled initial states only."}
        summary["paired_comparisons"] = comparisons
        summary["all_independent_physical_initial_states_equal"] = True
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
        state.update(status="complete", finished=time.time())
        save()
    except BaseException as exc:
        for name, child in children.items():
            if child.poll() is None:
                runstate = read_status(name)
                pid = runstate.get("train_pid")
                if pid:
                    cmdline = Path(f"/proc/{pid}/cmdline")
                    if cmdline.exists() and str(a.out / name).encode() in cmdline.read_bytes():
                        os.kill(pid, signal.SIGTERM)
                child.terminate()
        state.update(status="failed", error=repr(exc), finished=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
