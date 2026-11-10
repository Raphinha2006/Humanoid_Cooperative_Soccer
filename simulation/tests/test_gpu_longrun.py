"""Verify finite continuation status and bounded checkpoint recovery without a GPU."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("codes", [[0], [1, 0], [2]])
def test_supervisor_success_oom_recovery_and_non_oom_failure(tmp_path, monkeypatch, codes):
    spec = importlib.util.spec_from_file_location(
        "longrun", Path(__file__).resolve().parents[1] / "tools/run_gpu_receive_longrun.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "gpu_snapshot", lambda *args: {"free_mib": 20000})
    calls = []

    class Child:
        def __init__(self, cmd, **kwargs):
            self.pid = 100 + len(calls)
            self.rc = codes[len(calls)]
            calls.append(cmd)
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir()
            (out / "metrics.jsonl").write_text('{}\n')
            (out / "latest.pt").write_bytes(b"checkpoint")
            if self.rc == 0:
                (out / "final.pt").write_bytes(b"final")
            else:
                kwargs["stdout"].write("CUDA out of memory" if self.rc == 1 else "capacity overflow")

        def wait(self, timeout=None):
            return self.rc

        def poll(self):
            return self.rc

    monkeypatch.setattr(mod.subprocess, "Popen", Child)
    resume = tmp_path / "original.pt"
    resume.write_bytes(b"start")
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run", "--out", str(out), "--resume", str(resume),
        "--num-envs", "2048", "--batch", "1024", "--actor-lr", "0.00005"])
    if codes == [2]:
        with pytest.raises(RuntimeError, match="failed with code"):
            mod.main()
    else:
        assert mod.main() == 0
    state = json.loads((out / "status.json").read_text())
    assert state["status"] == ("failed" if codes == [2] else "complete")
    assert len(calls) == len(codes)
    for cmd in calls:
        assert cmd[cmd.index("--num-envs") + 1] == "2048"
        assert cmd[cmd.index("--batch") + 1] == "1024"
        assert float(cmd[cmd.index("--actor-lr") + 1]) == 5e-5
    if len(codes) == 2:
        assert calls[1][calls[1].index("--resume") + 1] == str(out / "wm/latest.pt")
        assert float(calls[1][calls[1].index("--hours") + 1]) <= 8
