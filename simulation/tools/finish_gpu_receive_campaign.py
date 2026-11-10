"""Finite post-training validation queue. No tuning or checkpoint selection on test data."""

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
    p.add_argument("campaign", type=Path)
    p.add_argument("--wait-hours", type=float, default=6)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--cpu-episodes", type=int, default=100)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("research/g1_scale/native_prediction_225000.npz"),
    )
    a = p.parse_args()
    out = a.campaign / "validation"
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "status.json").exists():
        raise FileExistsError(
            "Existing validation queue status: inspect completed stages before rerunning"
        )
    state = {
        "pid": os.getpid(),
        "status": "waiting_for_training",
        "started": time.time(),
        "stages": [],
    }
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "1",
        "PYTHONPATH": ".",
        "OPENBLAS_NUM_THREADS": "1",
        "MUJOCO_GL": "egl",
    }

    def save():
        temp = out / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / "status.json")

    def run(name, args):
        if len(args) > 2 and args[1] in [
            "soccer_sim.gpu_receive.evaluate",
            "soccer_sim.gpu_receive.cpu_confirm",
        ]:
            from soccer_sim.gpu_receive.receipt import verified

            kind = "gpu" if args[1].endswith("evaluate") else "cpu"
            checkpoint = args[args.index("--checkpoint") + 1]
            count = a.num_envs if kind == "gpu" else a.cpu_episodes
            if verified(args[2], checkpoint, 230000, kind, count):
                state["stages"].append(
                    {
                        "name": name,
                        "status": "complete",
                        "reused_verified_artifacts": True,
                        "returncode": 0,
                        "finished": time.time(),
                    }
                )
                save()
                return
        cmd = [sys.executable, "-u", *map(str, args)]
        record = {
            "name": name,
            "command": cmd,
            "started": time.time(),
            "status": "running",
        }
        state["stages"].append(record)
        state["status"] = name
        with (out / f"{name}.log").open("w") as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            save()
            rc = child.wait()
        record.update(
            returncode=rc,
            finished=time.time(),
            status="complete" if rc == 0 else "failed",
        )
        save()
        if rc:
            raise RuntimeError(
                f"Validation stage {name} failed with return code {rc}; see its log"
            )

    save()
    try:
        deadline = time.monotonic() + a.wait_hours * 3600
        while True:
            try:
                campaign = json.loads((a.campaign / "status.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                campaign = None
            if campaign:
                if any(r.get("status") == "failed" for r in campaign["runs"]):
                    raise RuntimeError(
                        "Training failed; refusing independent evaluation"
                    )
                if (
                    campaign.get("status")
                    == "training_complete_pending_independent_evaluation"
                ):
                    break
                proc = Path(f"/proc/{campaign['pid']}/cmdline")
                if not proc.exists() or not proc.read_bytes():
                    raise RuntimeError(
                        "Training supervisor exited before successful completion"
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Training wait budget reached; leave training untouched"
                )
            time.sleep(30)
        for algorithm in ["wm", "ppo"]:
            run_dir = a.campaign / algorithm
            lines = (run_dir / "metrics.jsonl").read_text().splitlines()
            if (
                json.loads(lines[-1])["stage"] != "complete"
                or not (run_dir / "final.pt").exists()
            ):
                raise RuntimeError(f"Missing complete artifacts for {algorithm}")
        wm_last = json.loads(
            (a.campaign / "wm/metrics.jsonl").read_text().splitlines()[-1]
        )
        target = wm_last["transitions"]
        matched = a.campaign / "ppo_sample_matched"
        run(
            "ppo_sample_matched",
            [
                "-m",
                "soccer_sim.gpu_receive.train",
                "--algorithm",
                "ppo",
                "--out",
                matched,
                "--hours",
                "4",
                "--num-envs",
                "4096",
                "--batch",
                "2048",
                "--micro",
                "128",
                "--wm-width",
                "2048",
                "--max-transitions",
                target,
            ],
        )
        matched_last = json.loads(
            (matched / "metrics.jsonl").read_text().splitlines()[-1]
        )
        if matched_last["transitions"] != target:
            raise RuntimeError(
                "PPO reached time budget before matching WM sample count; do not claim a sample-matched result"
            )
        for algorithm in ["wm", "ppo", "ppo_sample_matched"]:
            run_dir = a.campaign / algorithm
            for checkpoint in ["best.pt", "final.pt"]:
                stem = checkpoint.removesuffix(".pt")
                run(
                    f"{algorithm}_{stem}_gpu",
                    [
                        "-m",
                        "soccer_sim.gpu_receive.evaluate",
                        run_dir,
                        "--checkpoint",
                        checkpoint,
                        "--num-envs",
                        a.num_envs,
                        "--seed",
                        "230000",
                    ],
                )
                run(
                    f"{algorithm}_{stem}_cpu",
                    [
                        "-m",
                        "soccer_sim.gpu_receive.cpu_confirm",
                        run_dir,
                        "--checkpoint",
                        checkpoint,
                        "--episodes",
                        a.cpu_episodes,
                        "--seed",
                        "230000",
                    ],
                )
        for checkpoint in ["best.pt", "final.pt"]:
            path = a.campaign / "wm" / checkpoint
            if not path.exists():
                path = a.campaign / "wm/baseline.pt"
            run(
                f"prediction_{checkpoint.removesuffix('.pt')}",
                [
                    "-m",
                    "soccer_sim.gpu_receive.prediction",
                    "validate",
                    "--dataset",
                    a.dataset,
                    "--checkpoint",
                    path,
                    "--out",
                    out / f"prediction_{checkpoint.removesuffix('.pt')}.json",
                    "--device",
                    "cpu",
                ],
            )
        run(
            "report",
            ["tools/report_gpu_receive.py", a.campaign, "--out", out / "report"],
        )
        state.update(
            status="validation_complete_pending_human_analysis", finished=time.time()
        )
        save()
    except Exception as exc:
        state.update(status="failed", error=repr(exc), finished=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
