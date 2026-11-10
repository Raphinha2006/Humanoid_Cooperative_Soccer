"""Finish normal (ten-WM-fit) gates, then launch the frozen finite study."""

import argparse
import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

from run_cooperation_pilot import fingerprint, gpu_status, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--python", type=Path, required=True)
    a = p.parse_args()
    root, source = a.root.resolve(), a.root.resolve() / "source"
    path = root / "normal_gate_status.json"
    if path.exists():
        raise FileExistsError("Inspect the existing gate; never overwrite its record")
    state = dict(status="waiting_preflight", pid=os.getpid(), gpu=a.gpu, jobs=[])
    active = None

    def record(**updates):
        state.update(updates, heartbeat_epoch=time.time())
        write_json(path, state)

    def run(name, command):
        nonlocal active
        if shutil.disk_usage(root).free < 5 * 2**30:
            raise RuntimeError("Preserve 5GiB before admitting a new job")
        deadline = time.monotonic() + 1800
        while gpu_status(a.gpu)["free_mib"] < 10000:
            record(status="waiting_memory", job=name, gpu_status=gpu_status(a.gpu))
            if time.monotonic() > deadline:
                raise RuntimeError("Memory admission timed out")
            time.sleep(5)
        if fingerprint(source) != frozen_hashes:
            raise RuntimeError("Frozen source changed")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONPATH=str(source),
                   MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1",
                   PYTORCH_ALLOC_CONF="expandable_segments:True")
        started = changed = time.monotonic()
        size = -1
        with (root / f"{name}.log").open("x") as log:
            active = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                free = shutil.disk_usage(root).free / 2**30
                if free < 2:
                    raise RuntimeError("Disk reserve reached")
                now_size = (root / f"{name}.log").stat().st_size
                if now_size != size:
                    changed, size = time.monotonic(), now_size
                if name == "supervisor_full":
                    child_state = json.loads((root / "status.json").read_text())
                    if time.time() - child_state.get("heartbeat_epoch", 0) < 120:
                        changed = time.monotonic()
                if time.monotonic() - changed > 1200:
                    raise RuntimeError("No job output for twenty minutes")
                record(status="running", job=name, child_pid=active.pid,
                       seconds=time.monotonic() - started, command=command,
                       gpu_status=gpu_status(a.gpu), disk_free_gib=free)
                time.sleep(5)
        result = dict(name=name, returncode=active.returncode,
                      seconds=time.monotonic() - started)
        state["jobs"].append(result)
        record()
        if active.returncode:
            raise RuntimeError(f"{name} failed; full study was not launched")
        active = None

    try:
        deadline = time.monotonic() + 3600
        while True:
            previous = json.loads((root / "status.json").read_text())
            if previous["status"] == "preflight_complete":
                break
            if previous["status"] == "failed":
                raise RuntimeError("Frozen preflight failed: " + previous.get("error", ""))
            if time.time() - previous["heartbeat_epoch"] > 120:
                raise RuntimeError("Frozen preflight heartbeat stopped")
            if time.monotonic() > deadline:
                raise RuntimeError("Frozen preflight wait timed out")
            record(preflight_job=previous.get("job"))
            time.sleep(5)
        frozen_hashes = previous["source_sha256"]
        state["source_sha256"] = frozen_hashes
        plan = json.loads((root / "checkpoint_paths.json").read_text())
        for mode in ["held", "recorded", "closed_loop"]:
            name = "normal_gate_" + mode
            out = root / "runs" / name
            command = [str(a.python), "-u", "-m", "soccer_sim.gpu_receive.train",
                       "--out", str(out), "--num-envs", "2048", "--eval-num-envs", "512",
                       "--batch", "1024", "--micro", "128", "--wm-width", "2048",
                       "--njmax", "1024", "--hours", "1", "--resume", plan["start"],
                       "--sender-control", mode, "--controller-kinematics", "current",
                       "--matched-imagination-starts", "--fresh-training-rng", "--seed", "104",
                       "--new-updates", "1", "--eval-iterations", "1", "--no-render"]
            run(name, command)
            settings = json.loads((out / "settings.json").read_text())
            final = json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])
            if settings["smoke"] or final.get("completed_new_updates") != 1 or not final.get("matched_budget_completed"):
                raise RuntimeError("Normal update gate did not finish its required budget")
        record(status="normal_gates_passed")
        run("supervisor_full", [str(a.python), str(source / "tools/run_cooperation_pilot.py"),
            "--root", str(root), "--gpu", str(a.gpu), "--python", str(a.python), "--skip-preflight"])
        record(status="complete")
    except BaseException as error:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
        record(status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
