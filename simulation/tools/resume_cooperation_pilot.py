"""Resume only unstarted arms, preserving the frozen scientific configuration."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from run_cooperation_pilot import fingerprint, gpu_status, write_json


def process_memory_mib(gpu, pid):
    result = subprocess.run(["nvidia-smi", f"--id={gpu}",
        "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True)
    return sum(int(line.split(",")[1].strip()) for line in result.stdout.splitlines()
               if line.split(",")[0].strip() == str(pid))


def inspect_run(out, seed, mode):
    if not out.exists():
        return None
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    final = rows[-1]
    settings = json.loads((out / "settings.json").read_text())
    expected = dict(seed=seed, sender_control=mode, num_envs=2048, eval_num_envs=512,
                    batch=1024, micro=128, wm_width=2048, njmax=1024,
                    controller_kinematics="current", matched_imagination_starts=True,
                    fresh_training_rng=True, new_updates=512, eval_iterations=32,
                    checkpoint_seconds=1800, smoke=False, algorithm="wm")
    if any(settings.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"Settings mismatch in {out.name}")
    if (final.get("stage") != "complete" or not final.get("matched_budget_completed")
            or final.get("completed_new_updates") != 512
            or final.get("new_training_transitions") != 33554432):
        raise RuntimeError(f"Partial/censored run {out.name}: preserve it for inspection")
    checkpoint = out / "final.pt"
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"Missing final checkpoint in {out.name}")
    if any(x.get("training_capacity", {}).get("capacity_overflow_max", 0) for x in rows):
        raise RuntimeError(f"Capacity violation in {out.name}")
    return dict(name=out.name, final_iteration=final["iteration"],
                completed_new_updates=512, new_training_transitions=33554432,
                final_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--label", default="recovery_20261111")
    p.add_argument("--memory-wait-hours", type=float, default=12)
    p.add_argument("--held-admission-mib", type=int, default=10000)
    p.add_argument("--recorded-admission-mib", type=int, default=10000)
    p.add_argument("--closed-loop-admission-mib", type=int, default=10000)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    admission = dict(held=a.held_admission_mib, recorded=a.recorded_admission_mib,
                     closed_loop=a.closed_loop_admission_mib)
    if any(value <= 0 for value in admission.values()):
        p.error("Memory admission values must be positive")
    if not a.label.replace("_", "").isalnum() or a.memory_wait_hours <= 0:
        p.error("Use a simple unique label and positive wait limit")
    root, source = a.root.resolve(), a.root.resolve() / "source"
    lock = (root / "resume_queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((root / "status.json").read_text())
    if previous["status"] != "failed":
        raise RuntimeError("Inspect a non-failed original supervisor before recovery")
    for name in ["status.json", "normal_gate_status.json"]:
        state = json.loads((root / name).read_text())
        if Path(f"/proc/{state['pid']}").exists():
            raise RuntimeError("Original supervisor is still alive")
    hashes = previous["source_sha256"]
    if hashes != fingerprint(source):
        raise RuntimeError("Frozen algorithm/config/test hashes changed")
    gates = json.loads((root / "normal_gate_status.json").read_text())["jobs"]
    for mode in ["held", "recorded", "closed_loop"]:
        if not any(j["name"] == "normal_gate_" + mode and j["returncode"] == 0 for j in gates):
            raise RuntimeError("Missing passed normal update gate")
    order = [(seed, mode) for seed, modes in zip([104, 105, 106], [
        ["held", "recorded", "closed_loop"], ["recorded", "closed_loop", "held"],
        ["closed_loop", "held", "recorded"]]) for mode in modes]
    completed, pending = [], []
    for seed, mode in order:
        done = inspect_run(root / "runs" / f"seed{seed}_{mode}", seed, mode)
        if done:
            completed.append(done)
        else:
            pending.append((seed, mode))
    if a.check_only:
        print(json.dumps(dict(completed=completed, pending=pending, hashes_unchanged=True)))
        return
    path = root / (a.label + ".status.json")
    if path.exists():
        raise FileExistsError("Use a new recovery label; preserve the prior record")
    state = dict(status="starting", pid=os.getpid(), gpu=a.gpu, source_sha256=hashes,
                 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 completed_runs=completed, heldout_used=False, serial_execution=True,
                 memory_admission_mib=admission, memory_wait_hours=a.memory_wait_hours)
    active = None

    def record(**updates):
        state.update(updates, heartbeat_epoch=time.time())
        write_json(path, state)

    try:
        plan = json.loads((root / "checkpoint_paths.json").read_text())
        for seed, mode in pending:
            name = f"seed{seed}_{mode}"
            deadline = time.monotonic() + a.memory_wait_hours * 3600
            while True:
                device = gpu_status(a.gpu)
                if device["free_mib"] >= admission[mode]:
                    break
                record(status="waiting_for_memory", job=name, gpu_status=device,
                       wait_deadline_epoch=time.time() + deadline - time.monotonic())
                if time.monotonic() >= deadline:
                    raise RuntimeError("Memory admission timeout; no training settings changed")
                time.sleep(10)
            if fingerprint(source) != hashes:
                raise RuntimeError("Frozen source changed")
            if shutil.disk_usage(root).free < 5 * 2**30:
                raise RuntimeError("Preserve 5GiB before a new job")
            out = root / "runs" / name
            if out.exists():
                raise RuntimeError("Do not overwrite an existing run")
            command = [str(a.python), "-u", "-m", "soccer_sim.gpu_receive.train",
                "--out", str(out), "--num-envs", "2048", "--eval-num-envs", "512",
                "--batch", "1024", "--micro", "128", "--wm-width", "2048", "--njmax", "1024",
                "--hours", "6", "--resume", plan["start"], "--sender-control", mode,
                "--controller-kinematics", "current", "--matched-imagination-starts",
                "--fresh-training-rng", "--seed", str(seed), "--new-updates", "512",
                "--eval-iterations", "32", "--checkpoint-seconds", "1800", "--no-render"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONPATH=str(source),
                       MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1",
                       PYTORCH_ALLOC_CONF="expandable_segments:True")
            log_path = root / (name + ".log")
            with log_path.open("x") as log:
                active = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                started = changed = time.monotonic()
                last_size = -1
                process_peak = 0
                while active.poll() is None:
                    if log_path.stat().st_size != last_size:
                        last_size, changed = log_path.stat().st_size, time.monotonic()
                    if shutil.disk_usage(root).free < 2 * 2**30:
                        raise RuntimeError("Disk reserve reached")
                    if time.monotonic() - changed > 1200:
                        raise RuntimeError("No job output for twenty minutes")
                    process_used = process_memory_mib(a.gpu, active.pid)
                    process_peak = max(process_peak, process_used)
                    record(status="running", job=name, child_pid=active.pid, command=command,
                           job_seconds=time.monotonic() - started, gpu_status=gpu_status(a.gpu),
                           process_gpu_mib=process_used, max_sampled_process_gpu_mib=process_peak)
                    with (root / (name + ".process_memory.jsonl")).open("a") as memlog:
                        memlog.write(json.dumps(dict(epoch=time.time(), process_mib=process_used)) + "\n")
                    time.sleep(5)
                if active.returncode:
                    raise RuntimeError(f"{name} failed ({active.returncode})")
                active = None
            done = inspect_run(out, seed, mode)
            done["max_sampled_process_gpu_mib"] = process_peak
            state["completed_runs"].append(done)
            record()
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
