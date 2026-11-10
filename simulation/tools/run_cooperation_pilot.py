"""Finite serial, fail-fast baseline campaign; never manages unrelated processes."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def fingerprint(source):
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for pattern in ["soccer_sim/**/*.py", "configs/*.yaml", "tests/*.py", "tools/run_cooperation_pilot.py"]
            for p in sorted(source.glob(pattern))}


def gpu_status(gpu):
    result = subprocess.run(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True)
    free, used, util = [int(x.strip()) for x in result.stdout.strip().split(",")]
    return dict(free_mib=free, used_mib=used, utilization_percent=util)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    source = root / "source"
    state_path = root / "status.json"
    plan = json.loads((root / "checkpoint_paths.json").read_text())
    if a.skip_preflight:
        previous = json.loads(state_path.read_text())
        if previous.get("status") != "preflight_complete":
            raise RuntimeError("Validated preflight is required")
        hashes = previous["source_sha256"]
        if hashes != fingerprint(source):
            raise RuntimeError("Source changed after preflight; validate a new version")
        completed = previous["completed_jobs"]
    else:
        if state_path.exists():
            raise FileExistsError("Do not overwrite a campaign; inspect its saved status")
        hashes, completed = fingerprint(source), []
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu), "PYTHONPATH": str(source),
           "MUJOCO_GL": "egl", "OPENBLAS_NUM_THREADS": "1",
           "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    state = {"status": "starting", "pid": os.getpid(), "started_epoch": time.time(),
             "gpu": a.gpu, "source_sha256": hashes, "completed_jobs": completed,
             "heldout_used": False, "serial_execution": True}
    active = None
    checkpoint_args = [x for label, path in plan.items() for x in ["--checkpoint", f"{label}={path}"]]

    def run(name, command, gpu=True):
        nonlocal active
        if fingerprint(source) != hashes:
            raise RuntimeError("Frozen source changed during campaign")
        if shutil.disk_usage(root / "runs").free < 5 * 2**30:
            raise RuntimeError("Disk admission: preserve >=5GiB before a new job; no outputs deleted")
        if gpu:
            deadline = time.monotonic() + 1800
            while gpu_status(a.gpu)["free_mib"] < (10000 if "smoke" in name or "seed" in name else 3500):
                state.update(status="waiting_for_memory", job=name, gpu_status=gpu_status(a.gpu), heartbeat_epoch=time.time())
                write_json(state_path, state)
                if time.monotonic() >= deadline:
                    raise RuntimeError("Memory admission timeout; no process was displaced")
                time.sleep(10)
        log = root / f"{name}.log"
        local_env = dict(env, CUDA_VISIBLE_DEVICES="" if not gpu else str(a.gpu))
        with log.open("x") as f:
            proc = subprocess.Popen([str(a.python), *command], cwd=source, env=local_env,
                                    stdout=f, stderr=subprocess.STDOUT)
            active = proc
            started = last_change = time.monotonic()
            previous_size = -1
            peak_used = 0
            while proc.poll() is None:
                size = log.stat().st_size
                if size != previous_size:
                    previous_size, last_change = size, time.monotonic()
                device = gpu_status(a.gpu)
                peak_used = max(peak_used, device["used_mib"])
                state.update(status="running", job=name, child_pid=proc.pid,
                             command=[str(a.python), *command], heartbeat_epoch=time.time(),
                             job_seconds=time.monotonic() - started, gpu_status=device)
                state["disk_free_gib"] = shutil.disk_usage(root / "runs").free / 2**30
                write_json(state_path, state)
                if state["disk_free_gib"] < 2:
                    raise RuntimeError("Disk reserve reached; own child will be stopped; outputs preserved")
                if time.monotonic() - last_change > 1200:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise RuntimeError(f"Own job {name} stalled; preserved logs and checkpoints")
                time.sleep(5)
            record = {"name": name, "returncode": proc.returncode,
                      "seconds": time.monotonic() - started,
                      "observed_total_gpu_used_peak_mib": peak_used,
                      "log": str(log), "command": [str(a.python), *command]}
            state["completed_jobs"].append(record)
            write_json(state_path, state)
            if proc.returncode:
                raise RuntimeError(f"Job {name} failed ({proc.returncode}); queue halted")
            active = None
            print(json.dumps(record), flush=True)

    try:
        if not a.skip_preflight:
            run("cpu_regressions", ["-m", "pytest", "tests/test_gpu_receive.py",
                "tests/test_sender_consistency.py", "-k", "not closed_loop", "-q"], gpu=False)
            run("controller_conformance", ["-m", "pytest", "tests/test_sender_consistency.py",
                "-k", "closed_loop", "-q", "-s", "--tb=short"])
            for mode in ["legacy", "current"]:
                run(f"audit_{mode}", ["-u", "-m", "soccer_sim.gpu_receive.audit",
                    "--out", str(root / "runs" / f"audit_{mode}"), "--num-envs", "512",
                    "--repeats", "5", "--controller-kinematics", mode, *checkpoint_args])
            run("frozen_forecast", ["-u", "-m", "soccer_sim.gpu_receive.frozen_diagnostic",
                "--out", str(root / "runs/frozen_forecast"), *checkpoint_args])
            for mode in ["held", "recorded", "closed_loop"]:
                run(f"smoke_{mode}", ["-u", "-m", "soccer_sim.gpu_receive.train",
                    "--out", str(root / "runs" / f"smoke_{mode}"), "--num-envs", "2048",
                    "--eval-num-envs", "512", "--batch", "1024", "--micro", "128",
                    "--wm-width", "2048", "--njmax", "1024", "--hours", "1",
                    "--resume", plan["start"], "--sender-control", mode,
                    "--controller-kinematics", "current", "--matched-imagination-starts",
                    "--fresh-training-rng", "--new-updates", "1", "--no-render",
                    "--smoke", "--smoke-horizon", "32"])
            state.update(status="preflight_complete", heartbeat_epoch=time.time())
            write_json(state_path, state)
            if a.preflight_only:
                return
        orders = [["held", "recorded", "closed_loop"], ["recorded", "closed_loop", "held"],
                  ["closed_loop", "held", "recorded"]]
        for seed, modes in zip([104, 105, 106], orders):
            for mode in modes:
                name = f"seed{seed}_{mode}"
                out = root / "runs" / name
                run(name, ["-u", "-m", "soccer_sim.gpu_receive.train", "--out", str(out),
                    "--num-envs", "2048", "--eval-num-envs", "512", "--batch", "1024",
                    "--micro", "128", "--wm-width", "2048", "--njmax", "1024",
                    "--hours", "6", "--resume", plan["start"], "--sender-control", mode,
                    "--controller-kinematics", "current", "--matched-imagination-starts",
                    "--fresh-training-rng", "--seed", str(seed), "--new-updates", "512",
                    "--eval-iterations", "32", "--checkpoint-seconds", "1800", "--no-render"])
                rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
                final = rows[-1]
                if final.get("stage") != "complete" or not final.get("matched_budget_completed"):
                    raise RuntimeError(f"Censored/incomplete matched budget: {name}")
                if final["completed_new_updates"] != 512 or final["new_training_transitions"] != 33554432:
                    raise RuntimeError(f"Mismatched update or transition budget: {name}")
        state.update(status="complete", heartbeat_epoch=time.time(),
                     next_action="Analyse repaired baseline before introducing partner co-learning")
        write_json(state_path, state)
    except BaseException as error:
        if active is not None and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                active.kill()
                active.wait()
        state.update(status="failed", error=str(error), heartbeat_epoch=time.time())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
