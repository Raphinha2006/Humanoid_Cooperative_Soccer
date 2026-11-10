"""Reproduce fixed-state audits and frozen-trajectory WM forecasts; no selection."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, directory, name):
    fig.savefig(directory / (name + ".pdf"))
    fig.savefig(directory / (name + ".png"), dpi=300)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=root / "results/2026-11-10-control-audit")
    p.add_argument("--out", type=Path, default=root / "figures")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 9, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .15, "axes.axisbelow": True,
        "lines.linewidth": 1.8, "lines.markersize": 4,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    labels = ["native", "start", "peak", "late"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.4, 3.4), layout="constrained")
    for i, (mode, color) in enumerate([("legacy", "#0072B2"), ("current", "#E69F00")]):
        rows = json.loads((a.data_dir / f"{mode}_summary.json").read_text())["summary"]
        values = np.array([rows[k]["strict_rates"] for k in labels]) * 100
        centers = x + (i - .5) * .34
        ax.bar(centers, values.mean(1), .30, yerr=values.std(1), capsize=3,
               color=color, alpha=.75, label=mode + " observations")
        for j, rates in enumerate(values):
            ax.scatter(centers[j] + np.linspace(-.07, .07, len(rates)), rates,
                       s=11, color=color, edgecolor="white", linewidth=.3, zorder=4)
            ax.text(centers[j], rates.max() + 1.0, f"{rates.mean():.1f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x, ["Native skill", "Start\niter 141", "Historical peak\niter 537", "Late\niter 1119"])
    ax.set_ylabel("Strict reception success (%)")
    ax.set_ylim(0, 32)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Same initial states: 512 worlds × 5 repeats; bars = mean ± repeat SD")
    save(fig, a.out, "fig_control_audit")

    data = json.loads((a.data_dir / "forecast_errors.json").read_text())
    phases = ["pre_kick", "flight", "pre_contact", "post_contact"]
    modes = [("held", "#0072B2", "-", "o"),
             ("recorded", "#E69F00", "--", "s"),
             ("closed_loop", "#009E73", ":", "^")]
    metrics = [("ball_position_rmse_m", "Ball position RMSE (m)"),
               ("receiver_root_rmse_m", "Receiver root RMSE (m)")]
    for checkpoint in ["start", "peak", "late"]:
        fig, axes = plt.subplots(2, 4, figsize=(10, 5.0), sharex=True,
                                 sharey="row", layout="constrained")
        rows = [r for r in data["results"] if r["checkpoint"] == checkpoint]
        for col, phase in enumerate(phases):
            for row, (metric, title) in enumerate(metrics):
                ax = axes[row, col]
                for mode, color, style, marker in modes:
                    subset = sorted([r for r in rows if r["phase"] == phase
                                     and r["sender_control"] == mode],
                                    key=lambda r: r["horizon_steps"])
                    ax.plot([r["horizon_seconds"] for r in subset],
                            [r[metric] for r in subset], color=color,
                            linestyle=style, marker=marker, label=mode)
                ax.set_ylim(0, max(r[metric] for r in rows) * 1.10)
                ax.set_xticks([.16, .32, .64, 1.0])
                if row == 0:
                    n = next(r["n"] for r in rows if r["phase"] == phase)
                    ax.set_title(phase.replace("_", " ") + f" (n={n})")
                if col == 0:
                    ax.set_ylabel(title)
                if row == 1:
                    ax.set_xlabel("Forecast horizon (s)")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside lower center", ncol=3)
        iteration = rows[0]["iteration"]
        fig.suptitle(f"Frozen native trajectories · {checkpoint}, iter {iteration}\n"
                     "Receiver controls recorded in every mode; forecast diagnosis only", fontsize=11)
        save(fig, a.out, "fig_forecast_" + checkpoint)


if __name__ == "__main__":
    main()
