#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
OUT="${1:-runs/policy_delivery}"
export OPENBLAS_NUM_THREADS=1
if [[ "$(uname)" == Linux ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
mkdir -p "$OUT"
"$PYTHON" -m soccer_sim.policy_runner --kind booster_mjlab --scenario stand --seconds 8 --out "$OUT/booster_stand" > "$OUT/booster_stand.log"
"$PYTHON" -m soccer_sim.policy_runner --kind booster_mjlab --scenario walk --seconds 6 --out "$OUT/booster_walk" > "$OUT/booster_walk.log"
"$PYTHON" -m soccer_sim.policy_runner --kind booster_mjlab --scenario ball_contact --seconds 6 --out "$OUT/booster_ball" > "$OUT/booster_ball.log"
"$PYTHON" -m soccer_sim.policy_runner --kind booster_mjlab --scenario pass --config configs/hsl_m_pretrained_pass.yaml --seconds 12 --video --out "$OUT/booster_pass" > "$OUT/booster_pass.log"
"$PYTHON" -m soccer_sim.policy_runner --kind sonic_k1 --scenario motion --seconds 8 --video --out "$OUT/sonic_squat" > "$OUT/sonic_squat.log"
"$PYTHON" -m soccer_sim.policy_runner --kind sonic_k1 --scenario motion --seconds 5 --motion policies/sonic_k1/examples/k1/motions/crouch_walk.npz --out "$OUT/sonic_walk" > "$OUT/sonic_walk.log"
"$PYTHON" -m pytest -q > "$OUT/tests.log"
"$PYTHON" -m pip freeze > "$OUT/requirements-tested.txt"
"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = {p.parent.name: json.loads(p.read_text()) for p in root.glob('*/result.json')}
assert len(results) == 6
assert all(r['all_robots_no_fall'] and r['mujoco_warning_count'] == 0 for r in results.values())
assert results['booster_walk']['robots'][0]['displacement_xy_m'][0] > 1.0
assert results['booster_pass']['ball_contact_robots'] == ['A1', 'A2']
(root / 'summary.json').write_text(json.dumps(results, indent=2))
for name, r in results.items():
    print(name, 'seconds=', round(r['simulated_seconds'], 3),
          'no_fall=', r['all_robots_no_fall'], 'warnings=', r['mujoco_warning_count'],
          'pass_success=', r['pass_success'])
PY
cat "$OUT/tests.log"
