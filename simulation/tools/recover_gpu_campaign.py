"""Finish the interrupted WM campaign, then run predeclared controls and validation.

All original artifacts remain intact. The assembled WM view links the original
development-selected best and the recovery final; it does not reselect on test data.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    original = Path("runs/g1_gpu_campaign_v2")
    recovery = Path("runs/g1_gpu_v2_memory_recovery")
    root = Path("runs/g1_gpu_campaign_v2_completed_memory")
    root.mkdir(exist_ok=False)
    state = {
        "pid": os.getpid(),
        "started": time.time(),
        "runs": [],
        "status": "waiting_for_memory_recovery",
    }

    def save():
        tmp = root / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(root / "status.json")

    save()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": ".",
        "MUJOCO_GL": "egl",
    }
    try:
        pid = json.loads(Path("research/g1_scale/memory_recovery.json").read_text())[
            "pid"
        ]
        deadline = time.monotonic() + 3600
        while True:
            proc = Path(f"/proc/{pid}/cmdline")
            cmd = proc.read_bytes() if proc.exists() else b""
            if not cmd:
                break
            if (
                b"soccer_sim.gpu_receive.train" not in cmd
                or b"memory_recovery" not in cmd
            ):
                raise RuntimeError("Recovery PID changed identity")
            if time.monotonic() > deadline:
                raise TimeoutError("Recovery still running; leave it untouched")
            time.sleep(30)
        old = [
            json.loads(x)
            for x in (original / "wm/metrics.jsonl").read_text().splitlines()
        ]
        new = [
            json.loads(x) for x in (recovery / "metrics.jsonl").read_text().splitlines()
        ]
        if new[-1]["stage"] != "complete" or not (recovery / "final.pt").exists():
            raise RuntimeError("Recovery did not finish successfully")
        wm = root / "wm"
        wm.mkdir()
        for name, source in {
            "baseline.pt": original / "wm/baseline.pt",
            "best.pt": original / "wm/best.pt",
            "best.json": original / "wm/best.json",
            "final.pt": recovery / "final.pt",
        }.items():
            (wm / name).symlink_to(source.resolve())
        # Original best remains selected. Recovery evaluation is descriptive only.
        combined = list(old)
        for row in new:
            if row["stage"] not in ["baseline", "warmup"]:
                combined.append(
                    {
                        **row,
                        "seconds": old[-1]["seconds"] + row["seconds"],
                        "source_run": str(recovery),
                    }
                )
        (wm / "metrics.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in combined)
        )
        (wm / "assembly.json").write_text(
            json.dumps(
                {
                    "original": str(original),
                    "recovery": str(recovery),
                    "best_selection": "original development iteration129, frozen before independent test",
                    "capacity_change": "njmax384 to512; same model physics, new allocation",
                    "time_note": "Resumed model and optimizers with fresh replay warmup; not identical to uninterrupted training. PPO time2.5h; report actual WM total separately.",
                },
                indent=2,
            )
        )
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "soccer_sim.gpu_receive.train",
            "--out",
            str(root / "ppo"),
            "--algorithm",
            "ppo",
            "--hours",
            "2.5",
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
        ]
        with (root / "ppo.log").open("w") as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            rec = {
                "algorithm": "ppo",
                "pid": child.pid,
                "started": time.time(),
                "status": "running",
                "command": cmd,
            }
            state["runs"].append(rec)
            state["status"] = "ppo_running"
            save()
            rc = child.wait()
        rec.update(
            returncode=rc,
            finished=time.time(),
            status="complete" if rc == 0 else "failed",
        )
        if rc:
            raise RuntimeError(f"PPO failed with return code {rc}")
        state["status"] = "training_complete_pending_independent_evaluation"
        save()
        with (root / "validation_launcher.log").open("w") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "tools/finish_gpu_receive_campaign.py",
                    str(root),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            state["validation_pid"] = child.pid
            save()
            rc = child.wait()
        state.update(validation_returncode=rc, finished=time.time())
        save()
    except Exception as exc:
        state.update(status="failed", error=repr(exc), finished=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
