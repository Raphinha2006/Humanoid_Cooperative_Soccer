"""Verify reusable final-evaluation artifacts without inspecting their scores."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sources():
    root = Path(__file__).resolve().parents[1]
    paths = list(root.rglob("*.py")) + [Path("configs/supermimic_g1_receive.yaml")]
    return {str(p.resolve()): sha256(p) for p in paths}


def write_receipt(run, checkpoint, seed, kind, count, artifacts):
    run = Path(run)
    record = {
        "checkpoint": checkpoint,
        "seed": seed,
        "kind": kind,
        "count": count,
        "checkpoint_sha256": sha256(run / checkpoint),
        "sources": sources(),
        "artifacts": {str(Path(p).resolve()): sha256(p) for p in artifacts},
    }
    path = run / f"receipt_{seed}_{Path(checkpoint).stem}_{kind}.json"
    path.write_text(json.dumps(record, indent=2))


def verified(run, checkpoint, seed, kind, count):
    run = Path(run)
    if checkpoint == "best.pt" and not (run / checkpoint).exists():
        checkpoint = "baseline.pt"
    path = run / f"receipt_{seed}_{Path(checkpoint).stem}_{kind}.json"
    try:
        record = json.loads(path.read_text())
        return (
            record["checkpoint"] == checkpoint
            and record["seed"] == seed
            and record["kind"] == kind
            and record["count"] == count
            and record["checkpoint_sha256"] == sha256(run / checkpoint)
            and record["sources"] == sources()
            and bool(record["artifacts"])
            and all(sha256(p) == digest for p, digest in record["artifacts"].items())
        )
    except (OSError, ValueError, KeyError):
        return False
