"""Read-only learning curves and paired evaluation statistics; never select on test data."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_jsonl(path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A running writer may not have finished its last line.
            continue
    return records


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - radius, center + radius]


def paired_binary(baseline, policy):
    b = np.asarray(baseline, dtype=bool)
    p = np.asarray(policy, dtype=bool)
    if b.shape != p.shape or b.ndim != 1 or not len(b):
        raise ValueError("Paired outcomes must be nonempty vectors of identical shape")
    delta = p.astype(float) - b.astype(float)
    # Resample pairs, retaining the within-initial-state dependence.
    rng = np.random.default_rng(81729)
    estimates = []
    for _ in range(20):
        ids = rng.integers(len(b), size=(250, len(b)))
        estimates.extend(delta[ids].mean(axis=1))
    return {
        "n": len(b),
        "baseline": float(b.mean()),
        "policy": float(p.mean()),
        "baseline_wilson_95": wilson(int(b.sum()), len(b)),
        "policy_wilson_95": wilson(int(p.sum()), len(p)),
        "difference": float(delta.mean()),
        "paired_bootstrap_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "improved_pairs": int((p & ~b).sum()),
        "worsened_pairs": int((b & ~p).sum()),
        "uncertainty_scope": "Conditional on this trained policy and saved initial-state distribution; excludes training-seed and GPU repeat variability.",
    }


def convergence(records):
    ev = [r for r in records if r.get("stage") in ["evaluation", "complete"]]
    if not ev:
        return {"status": "no_evaluations"}
    cutoff = ev[-1]["seconds"] - 1800
    recent = [r for r in ev if r["seconds"] >= cutoff]
    before = [r for r in ev if r["seconds"] < cutoff]
    if before:
        recent.insert(0, before[-1])
    span = recent[-1]["seconds"] - recent[0]["seconds"]
    result = {
        "status": "descriptive_window_available"
        if span >= 1800 and len(recent) >= 6
        else "insufficient_30_minute_window",
        "evaluation_count": len(recent),
        "span_seconds": span,
        "note": "A flat development curve is not proof of convergence or generalization. No automatic success verdict.",
        "metrics": {},
    }
    if len({r.get("source_run", "original") for r in recent}) > 1:
        result["status"] = "mixed_development_protocols_no_convergence_verdict"
        result["note"] = (
            "Recovery changed evaluation-world count; pooled trends are descriptive and cannot establish convergence."
        )
    for key in [
        "front_pass_success",
        "handoff_at_end",
        "max_hold_mean",
        "return_mean",
        "ball_rmse_50",
    ]:
        pairs = [
            (r["seconds"], r["evaluation"][key])
            for r in recent
            if key in r["evaluation"]
        ]
        if not pairs:
            continue
        t, y = np.asarray(pairs, dtype=float).T
        slope = float(np.polyfit((t - t[0]) / 3600, y, 1)[0]) if len(t) > 1 else None
        result["metrics"][key] = {
            "first": float(y[0]),
            "last": float(y[-1]),
            "min": float(y.min()),
            "max": float(y.max()),
            "range": float(np.ptp(y)),
            "slope_per_hour": slope,
        }
    return result


def comparisons(run):
    result = {}
    for path in sorted(run.glob("independent_*.json")):
        baseline, policy = json.loads(path.read_text())
        bp, pp = baseline["per_world"], policy["per_world"]
        result[path.name] = {
            "checkpoint": policy["checkpoint"],
            **{
                k: paired_binary(bp[k], pp[k])
                for k in [
                    "strict",
                    "handoff",
                    "fallen",
                    "moving_ball_strict_diagnostic",
                ]
                if k in bp and k in pp
            },
        }
    for path in sorted(run.glob("cpu_confirmation_*.json")):
        baseline, policy = json.loads(path.read_text())
        result[path.name] = {
            "checkpoint": policy["checkpoint"],
            **{
                k: paired_binary(
                    [r[k] for r in baseline["episodes"]],
                    [r[k] for r in policy["episodes"]],
                )
                for k in ["front_pass_success", "handoff_at_end", "fallen"]
            },
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"campaign": str(a.campaign), "runs": {}}
    tables = []
    for algorithm in ["wm", "ppo", "ppo_sample_matched"]:
        run = a.campaign / algorithm
        records = read_jsonl(run / "metrics.jsonl")
        if not records:
            continue
        last = records[-1]
        report["runs"][algorithm] = {
            "training_complete": any(r.get("stage") == "complete" for r in records),
            "last_stage": last["stage"],
            "last_seconds": last["seconds"],
            "last_iteration": last["iteration"],
            "last_transitions": last["transitions"],
            "convergence": convergence(records),
            "independent_comparisons": comparisons(run),
        }
        for r in records:
            if "evaluation" in r:
                tables.append(
                    {
                        "algorithm": algorithm,
                        "stage": r["stage"],
                        "seconds": r["seconds"],
                        "transitions": r["transitions"],
                        "iteration": r["iteration"],
                        "source_run": r.get("source_run", str(run)),
                        **r["evaluation"],
                    }
                )
    report["comparison_limits"] = [
        "WM and PPO collect different numbers of real transitions per update; iteration counts are not matched budgets.",
        "A same-wall-time comparison includes sampling and update cost on a shared GPU. Report actual transitions alongside it.",
        "Sample-matched claims require evaluated checkpoints at a common real-transition budget; do not interpolate success rates between checkpoints.",
        "Development results select checkpoints. Independent results must not be reused to change that selection.",
    ]
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    if tables:
        fields = list(dict.fromkeys(k for row in tables for k in row))
        with (a.out / "evaluations.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(tables)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, key, label, scale in zip(
            axs.flat,
            ["front_pass_success", "handoff_at_end", "max_hold_mean", "ball_rmse_50"],
            [
                "Strict receive success (%)",
                "Handoff at end (%)",
                "Maximum hold, mean (s)",
                "Ball position RMSE at 1 s (m)",
            ],
            [100, 100, 1, 1],
            strict=True,
        ):
            for algorithm in report["runs"]:
                rows = [r for r in tables if r["algorithm"] == algorithm and key in r]
                for source in dict.fromkeys(r["source_run"] for r in rows):
                    segment = [r for r in rows if r["source_run"] == source]
                    ax.plot(
                        [r["seconds"] / 60 for r in segment],
                        [r[key] * scale for r in segment],
                        marker="o",
                        markersize=3,
                        label=algorithm.upper()
                        + (" recovery (512 eval)" if "recovery" in source else ""),
                    )
            ax.set_xlabel("Elapsed minutes (sampling, training, evaluation)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend()
        fig.suptitle("G1 facing receive — development monitoring, not final test")
        fig.savefig(a.out / "learning_curves.png", dpi=180)
        plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
