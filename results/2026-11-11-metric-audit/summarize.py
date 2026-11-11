"""Summarize existing development logs; no simulation or new held-out evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    result = {
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "scope": "Post-hoc descriptive audit, seed104 development logs only",
        "ci_scope": "Approximate Wilson intervals across perturbed worlds for one fixed actor; not training-seed uncertainty or paired significance tests",
        "runs": {},
    }
    keys = ["front_pass_success", "ordered_contact", "captured", "max_hold_mean",
            "mean_hold", "fallen", "handoff_at_end", "final_distance", "final_speed"]
    for name, run in data["runs"].items():
        n = run["settings"]["eval_num_envs"]
        start = next(row for row in run["metrics"] if row.get("stage") == "resume_evaluation")
        final = run["metrics"][-1]
        assert final["stage"] == "complete"
        item = {"nworlds": n}
        for label, row in [("start", start), ("final", final)]:
            metrics = row["evaluation"]
            count = round(metrics["front_pass_success"] * n)
            assert abs(count / n - metrics["front_pass_success"]) < 1e-8
            item[label] = {key: metrics[key] for key in keys}
            item[label]["strict_success_count"] = count
            item[label]["strict_wilson95"] = wilson(count, n)
        result["runs"][name] = item
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
