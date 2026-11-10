"""Finite continuation with heartbeat, resource sampling and bounded OOM recovery."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def gpu_snapshot(gpu, pid=None):
    r = subprocess.run(
        ["nvidia-smi", f"--id={gpu}",
         "--query-gpu=memory.free,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    if r.returncode:
        raise RuntimeError(r.stderr)
    free, used, total, util = [int(x.strip()) for x in r.stdout.strip().split(",")]
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    own = [int(line.split(",")[1].strip()) for line in apps.stdout.splitlines()
           if line.split(",")[0].strip() == str(pid)]
    return {"free_mib": free, "used_mib": used, "total_mib": total,
            "utilization": util, "train_process_mib": sum(own)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--njmax", type=int, default=1024)
    p.add_argument("--min-free-gib", type=float, default=14)
    p.add_argument("--max-oom-retries", type=int, default=2)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--actor-lr", type=float)
    a = p.parse_args()
    if not a.resume.is_file() or a.hours <= 0:
        p.error("An existing resume checkpoint and positive duration are required")
    a.out.mkdir(parents=True, exist_ok=False)
    lock = (a.out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"supervisor_pid": os.getpid(), "started": time.time(),
             "status": "starting", "gpu_physical_index": a.gpu,
             "hours": a.hours, "checkpoint_seconds": 1800, "njmax": a.njmax,
             "active_seconds": 0, "attempts": [], "max_oom_retries": a.max_oom_retries}
    state.update(num_envs=a.num_envs, batch=a.batch, actor_lr=a.actor_lr)

    def save():
        state["heartbeat"] = time.time()
        tmp = a.out / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(a.out / "status.json")

    def sample(pid=None):
        record = {"time": time.time(), "pid": pid}
        try:
            record.update(gpu_snapshot(a.gpu, pid))
        except Exception as exc:
            record["error"] = repr(exc)
        with (a.out / "resources.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    save()
    child = None
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu), "PYTHONPATH": ".",
               "OPENBLAS_NUM_THREADS": "1", "MUJOCO_GL": "egl"}
        resume = a.resume
        for attempt in range(a.max_oom_retries + 1):
            wait_started = time.monotonic()
            state["status"] = "waiting_for_gpu_memory"
            while True:
                resource = sample()
                state["resources"] = resource
                save()
                if resource.get("free_mib", 0) >= a.min_free_gib * 1024:
                    break
                if time.monotonic() - wait_started > 1800:
                    raise RuntimeError("GPU memory unavailable for30minutes")
                time.sleep(30)
            remaining = a.hours * 3600 - state["active_seconds"]
            if remaining < 60:
                raise RuntimeError("Budget exhausted by failed attempts; no final produced")
            out = a.out / ("wm" if attempt == 0 else f"wm_restart{attempt}")
            cmd = [sys.executable, "-u", "-m", "soccer_sim.gpu_receive.train",
                   "--out", str(out), "--resume", str(resume),
                   "--algorithm", "wm", "--hours", str(remaining / 3600),
                   "--num-envs", str(a.num_envs), "--eval-num-envs", "512",
                   "--batch", str(a.batch), "--micro", "128", "--wm-width", "2048",
                   "--collect-steps", "32", "--eval-seconds", "300",
                   "--checkpoint-seconds", "1800", "--njmax", str(a.njmax)]
            if a.actor_lr is not None:
                cmd.extend(["--actor-lr", str(a.actor_lr)])
            logpath = a.out / ("train.log" if attempt == 0 else f"train_restart{attempt}.log")
            started = time.monotonic()
            with logpath.open("w") as log:
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                rec = {"pid": child.pid, "started": time.time(), "out": str(out),
                       "command": cmd, "remaining_seconds": remaining}
                state["attempts"].append(rec)
                state.update(status="running", train_pid=child.pid, active_run=str(out), command=cmd)
                save()
                progress, last_mtime = time.monotonic(), None
                while True:
                    state["resources"] = sample(child.pid)
                    metrics = out / "metrics.jsonl"
                    if metrics.exists():
                        mtime = metrics.stat().st_mtime_ns
                        if mtime != last_mtime:
                            progress, last_mtime = time.monotonic(), mtime
                    save()
                    if time.monotonic() - progress > 900:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise RuntimeError("Training metrics stalled for15minutes")
                    try:
                        rc = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            elapsed = time.monotonic() - started
            state["active_seconds"] += elapsed
            rec.update(returncode=rc, elapsed_seconds=elapsed, finished=time.time())
            save()
            if rc == 0:
                if not (out / "final.pt").is_file():
                    raise RuntimeError("Child exited without final checkpoint")
                state.update(status="complete", returncode=rc, finished=time.time())
                save()
                return 0
            tail = logpath.read_text()[-6000:]
            if "out of memory" not in tail.lower() and "failed to allocate" not in tail.lower():
                raise RuntimeError(f"Training failed with code{rc}; see{logpath}")
            latest = out / "latest.pt"
            resume = latest if latest.is_file() else out / "resume_start.pt"
            if not resume.is_file():
                raise RuntimeError("OOM before any recoverable checkpoint")
            state.update(status="oom_recovery_pending", recovery_checkpoint=str(resume))
            save()
        raise RuntimeError("OOM retry budget exhausted")
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
        state.update(status="failed", error=repr(exc), finished=time.time())
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
