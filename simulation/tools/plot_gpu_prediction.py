"""Reproducible fixed-trajectory prediction diagnostics, without checkpoint selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = json.loads(a.result.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, subset in enumerate(["all", "receiver_contact_in_window"]):
        for col, key in enumerate(
            ["ball_position_m", "receiver_body_m", "receiver_feet_m"]
        ):
            ax = axs[row, col]
            for method, color, marker in [
                ("wm", "#0072B2", "o"),
                ("constant_velocity", "#D55E00", "s"),
            ]:
                rows = [
                    r
                    for r in result["aggregates"]
                    if r["method"] == method and r["subset"] == subset
                ]
                ax.plot(
                    [r["horizon_s"] for r in rows],
                    [100 * r["rmse"][key] for r in rows],
                    marker=marker,
                    color=color,
                    label=method.replace("_", " "),
                )
            ax.set(
                xlabel="Prediction horizon (s)",
                ylabel="Position RMSE (cm)",
                title=["Ball", "Receiver bodies", "Receiver feet"][col]
                + (" | all anchors" if row == 0 else " | contact windows"),
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Fixed native-policy trajectories | known future executed joint commands"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(a.out / f"fixed_prediction.{ext}", dpi=300)
    plt.close(fig)
    lines = [
        "# Fixed trajectory prediction diagnostic",
        "",
        f"Checkpoint: `{result['checkpoint']}`, iteration {result['iteration']}.",
        f"Dataset SHA256: `{result['dataset_sha256']}`.",
        "",
        "| Subset | Horizon (s) | WM ball RMSE (cm) | Constant velocity (cm) |",
        "|---|---:|---:|---:|",
    ]
    for subset in ["all", "receiver_contact_in_window"]:
        for h in [0.1, 0.5, 0.64, 1.0]:
            rows = {
                r["method"]: r
                for r in result["aggregates"]
                if r["subset"] == subset and r["horizon_s"] == h
            }
            lines.append(
                f"| {subset} | {h:g} | {100 * rows['wm']['rmse']['ball_position_m']:.2f} | {100 * rows['constant_velocity']['rmse']['ball_position_m']:.2f} |"
            )
    lines += [
        "",
        "256 frozen-native worlds; overlapping windows are correlated. This dataset never enters replay and does not select checkpoints. Future executed controls are provided to the predictors, so this is not an unknown-action online forecast. Prediction improvement is separate from real receive success.",
        "",
        "Contact windows are identified by actual first receiver contact times. Their selected initial states differ by horizon, so the contact-row curves are descriptive rather than repeated forecasts on an identical subset.",
    ]
    (a.out / "FIXED_PREDICTION.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
