"""Recorded-sender continuation curves; two seeds, no aggregate inference."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed104", type=Path,
                        default=root / "results/2026-11-11-seed104-comparison/comparison.json")
    parser.add_argument("--seed105", type=Path,
                        default=root / "results/2026-11-11-seed105-recorded/progress.json")
    parser.add_argument("--out", type=Path, default=root / "figures")
    args = parser.parse_args()
    older = json.loads(args.seed104.read_text())["runs"]["recorded"]["metrics"]
    newer = json.loads(args.seed105.read_text())["metrics"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "legend.fontsize": 9, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .15, "lines.linewidth": 1.6,
        "lines.markersize": 4, "savefig.bbox": "tight", "pdf.fonttype": 42})
    panels = [
        ("front_pass_success", 100, "Strict acceptance", "Success (%)"),
        ("captured", 100, "Ever captured (legacy rule)", "Worlds (%)"),
        ("max_hold_mean", 1, "Continuous control; not stopping latency", "Mean maximum duration (s)"),
        ("return_mean", 1, "Corrected simulator return", "Return"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8), layout="constrained")
    for label, rows, color, marker in [("seed104", older, "#0072B2", "o"),
                                       ("seed105", newer, "#E69F00", "s")]:
        series = [r for r in rows if r["stage"] in ["resume_evaluation", "evaluation"]]
        final = rows[-1]
        assert final["stage"] == "complete" and final["matched_budget_completed"]
        x = [(r["iteration"] - 141) * 32 * 2048 / 1e6 for r in series]
        final_x = final["completed_new_updates"] * 32 * 2048 / 1e6
        for ax, (key, scale, title, ylabel) in zip(axes.flat, panels):
            ax.plot(x, [r["evaluation"][key] * scale for r in series],
                    label=label, color=color, marker=marker)
            ax.scatter(final_x, final["evaluation"][key] * scale,
                       color=color, marker="x", s=45, zorder=5)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Additional real transitions (M)")
            ax.set_xticks([0, 8, 16, 24, 32])
    axes[0, 0].legend()
    fig.suptitle("Recorded sender: two continuation seeds, common checkpoint\n"
                 "Original reward; development only; crosses = final re-evaluations", fontsize=11)
    args.out.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(args.out / ("fig_recorded_seed105." + extension), dpi=300)


if __name__ == "__main__":
    main()
