"""Read-only finite experiment monitor: actual process liveness, GPU1, latest metrics."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("--max-hours", type=float, default=6)
    a = p.parse_args()
    end = time.monotonic() + a.max_hours * 3600
    while time.monotonic() < end:
        if not (a.campaign / "status.json").exists():
            time.sleep(5)
            continue
        try:
            status = json.loads((a.campaign / "status.json").read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        rows = []
        for run in status["runs"]:
            pid = run["pid"]
            live = Path(f"/proc/{pid}/cmdline").exists()
            last = None
            logs = a.campaign / run["algorithm"] / "metrics.jsonl"
            if logs.exists():
                lines = logs.read_text().splitlines()
                for line in reversed(lines):
                    try:
                        last = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            rows.append(
                {"algorithm": run["algorithm"], "pid": pid, "live": live, "last": last}
            )
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--id=1",
                "--query-gpu=uuid,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        sample = {
            "time": time.time(),
            "monitor_pid": os.getpid(),
            "gpu1": gpu,
            "runs": rows,
        }
        with (a.campaign / "monitor.jsonl").open("a") as f:
            f.write(json.dumps(sample) + "\n")
        print(json.dumps(sample), flush=True)
        if status.get(
            "status"
        ) == "training_complete_pending_independent_evaluation" or any(
            r.get("status") == "failed" for r in status["runs"]
        ):
            break
        if not Path(f"/proc/{status['pid']}/cmdline").exists():
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
