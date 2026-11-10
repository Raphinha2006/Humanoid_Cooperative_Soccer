"""Report prediction and real-physics outcomes of the larger direct WM runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--test", type=int, default=130000)
    args = p.parse_args()
    run = args.run
    logs = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    initial = next(x for x in logs if x["stage"] == "initial")
    rows = [x for x in logs if "validation" in x and x["stage"] == "round"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, key, label in zip(
        axes[:2],
        ["ball_position_rmse_50", "receiver_body_rmse_50"],
        ["Ball", "Receiver bodies (FK)"],
    ):
        ax.plot(
            [0] + [x["round"] for x in rows],
            [100 * initial["validation"][key]]
            + [100 * x["validation"][key] for x in rows],
            marker="o",
        )
        ax.set(
            xlabel="Round (0 = untrained prior)",
            ylabel="Position RMSE at 1 s (cm)",
            title=label,
        )
        ax.grid(alpha=0.2)
    axes[2].plot(
        [0] + [x["round"] for x in rows],
        [100 * logs[0]["evaluation"]["front_pass_success"]]
        + [100 * x["evaluation"]["front_pass_success"] for x in rows],
        marker="o",
        label="Strict front capture",
    )
    axes[2].plot(
        [0] + [x["round"] for x in rows],
        [100 * logs[0]["evaluation"]["fallen"]]
        + [100 * x["evaluation"]["fallen"] for x in rows],
        marker="o",
        label="Fall",
    )
    axes[2].set(
        xlabel="Round", ylabel="Monitor rate (%)", title="Fixed 20-state monitor"
    )
    rate_max = max(float(line.get_ydata().max()) for line in axes[2].lines)
    axes[2].set_ylim(0, min(100, max(10, rate_max * 1.2)))
    axes[2].legend()
    axes[2].grid(alpha=0.2)
    fig.savefig(run / "direct_learning_curves.png", dpi=180)
    fig.savefig(run / "direct_learning_curves.pdf")
    lines = [
        "# Direct WM architecture: independent evaluation",
        "",
        f"Run: `{run.name}`; one training seed; monitor is not independent test.",
        "",
        "| Checkpoint | Strict front | Handoff at end | Falls | Receiver distance (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    testpath = run / f"paired_{args.test}.json"
    stats = []
    if testpath.exists():
        test = json.loads(testpath.read_text())
        for item in test:
            m = item["mean"]
            lines.append(
                f"| {item['checkpoint']} | {100 * m['front_pass_success']:.1f}% | {100 * m['handoff_at_end']:.1f}% | {100 * m['fallen']:.1f}% | {m['final_receiver_distance']:.3f} |"
            )
        baseline = test[0]
        rng = np.random.default_rng(8261)
        for item in test[1:]:
            assert [m["seed"] for m in item["episodes"]] == [
                m["seed"] for m in baseline["episodes"]
            ]
            for key in ["front_pass_success", "handoff_at_end"]:
                diff = np.array(
                    [
                        int(m[key]) - int(b[key])
                        for m, b in zip(item["episodes"], baseline["episodes"])
                    ]
                )
                means = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(-1)
                stats.append(
                    {
                        "checkpoint": item["checkpoint"],
                        "metric": key,
                        "difference": float(diff.mean()),
                        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
                    }
                )
        (run / "paired_bootstrap.json").write_text(json.dumps(stats, indent=2))
    lines += [
        "",
        "Handoff uses <=1.2 m / <0.6 m/s / >=0.8 s plus ordered real contacts, facing and no falls. Strict G1 capture uses <0.55 m / <0.5 m/s / >=0.8 s and a front region. These are different endpoints. Bootstrap covers paired initial states, not variation across training seeds.",
        "",
        "See docs/ARCHITECTURE_V2.md for deviations from SuperMimic and native G1 deployment settings.",
    ]
    (run / "DIRECT_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"run": run.name, "bootstrap": stats}, indent=2))


if __name__ == "__main__":
    main()
