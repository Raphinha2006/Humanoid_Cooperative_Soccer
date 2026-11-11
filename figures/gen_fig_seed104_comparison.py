"""Reproduce the first complete seed comparison; no cross-seed inference."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=root / "results/2026-11-11-seed104-comparison/comparison.json")
    p.add_argument("--out", type=Path, default=root / "figures")
    a = p.parse_args()
    data = json.loads(a.data.read_text())
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "legend.fontsize": 9, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .15, "lines.linewidth": 1.6,
        "lines.markersize": 4, "savefig.bbox": "tight", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), layout="constrained")
    for mode, color, marker in [("held", "#0072B2", "o"),
                                ("recorded", "#E69F00", "s"),
                                ("closed_loop", "#009E73", "^")]:
        rows = data["runs"][mode]["metrics"]
        ev = [r for r in rows if r["stage"] in ["resume_evaluation", "evaluation"]]
        final = rows[-1]
        x = [(r["iteration"] - 141) * 32 * 2048 / 1e6 for r in ev]
        axes[0].plot(x, [r["evaluation"]["front_pass_success"] * 100 for r in ev],
                     color=color, marker=marker, label=mode)
        axes[0].scatter(x[-1], final["evaluation"]["front_pass_success"] * 100,
                        marker="x", s=40, color=color, zorder=4)
        axes[1].plot(x, [r["evaluation"]["return_mean"] for r in ev],
                     color=color, marker=marker, label=mode)
    axes[0].set_ylabel("Strict reception success (%)")
    axes[0].set_ylim(-.4, 10)
    axes[0].set_title("No arm sustains a gain in this seed")
    axes[1].set_ylabel("Corrected simulator return")
    axes[1].set_title("Return is not strict reception success")
    for ax in axes:
        ax.set_xlabel("Additional real transitions (M)")
        ax.set_xticks([0, 8, 16, 24, 32])
    axes[0].legend(loc="upper right")
    fig.suptitle("Common start · continuation seed104 · 512 added updates per arm\n"
                 "Development evaluation only; crosses = final re-evaluations", fontsize=11)
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(a.out / ("fig_seed104_comparison." + ext), dpi=300)


if __name__ == "__main__":
    main()
