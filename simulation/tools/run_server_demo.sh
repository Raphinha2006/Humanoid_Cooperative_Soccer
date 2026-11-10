#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/pytest -q
MUJOCO_GL=egl .venv/bin/soccer-sim pass --out runs/demo --video
.venv/bin/soccer-sim probe --out runs/calibration
.venv/bin/pip freeze > runs/requirements-tested.txt
