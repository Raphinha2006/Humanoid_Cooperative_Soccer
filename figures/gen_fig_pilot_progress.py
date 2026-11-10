"""Plot the completed held-sender continuation; development results only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path,
                   default=root / "results/2026-11-11-pilot-progress/metrics.json")
    p.add_argument("--out", type=Path, default=root / "figures")
    a = p.parse_args()
    records = json.loads(a.metrics.read_text())
    rows = [r for r in records if r["stage"] in ["resume_evaluation", "evaluation"]]
    final = next(r for r in reversed(records) if r["stage"] == "complete")
    start = rows[0]["iteration"]
    x = np.array([(r["iteration"] - start) * 32 * 2048 / 1e6 for r in rows])
    scores = [r["evaluation"] for r in rows]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .15, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 3.5,
        "savefig.bbox": "tight", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.1), layout="constrained")
    rates = np.array([s["front_pass_success"] for s in scores]) * 100
    axes[0].plot(x, rates, "o-", color="#0072B2", label="Periodic evaluation")
    axes[0].axhline(rates[0], color=".5", linestyle="--", linewidth=1, label="Start")
    axes[0].scatter(x[-1], final["evaluation"]["front_pass_success"] * 100,
                    color="#D55E00", marker="x", s=36, label="Final re-evaluation", zorder=4)
    axes[0].set_ylabel("Strict reception success (%)")
    axes[0].set_ylim(-.4, 10)
    axes[0].set_title("No sustained improvement")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[1].plot(x, [s["return_mean"] for s in scores], "o-", color="#0072B2")
    axes[1].set_title("Real simulator return")
    axes[1].set_ylabel("Corrected return")
    for steps, color, marker in [(5, "#0072B2", "o"), (25, "#E69F00", "s"), (50, "#009E73", "^")]:
        axes[2].plot(x, [s[f"ball_rmse_{steps}"] for s in scores], color=color,
                     marker=marker, label=f"{steps * .02:g} s")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Ball position RMSE (m)")
    axes[2].set_title("Forecasts on each policy's trajectories")
    axes[2].legend(ncol=3, loc="upper right")
    for ax in axes:
        ax.set_xlabel("Additional real transitions (M)")
        ax.set_xticks([0, 8, 16, 24, 32])
    fig.suptitle("held sender · continuation seed104 · 512 additional updates\n"
                 "512-world development evaluation; one training replicate", fontsize=11)
    a.out.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(a.out / ("fig_pilot_seed104_held." + suffix), dpi=300)


if __name__ == "__main__":
    main()
