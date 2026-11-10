"""Make standalone learning curves and a compact, auditable results table."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluation_context = ""
    for run in args.runs:
        path = run / "paired_evaluation.json"
        if path.exists():
            item = json.loads(path.read_text())[0]
            evaluation_context = f"Task: {item['episodes']} paired initial states, seeds {item['seed_start']}–{item['seed_start'] + item['episodes'] - 1}."
            break
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    fig.suptitle(
        "Prediction: fixed held-out episodes per run. " + evaluation_context,
        fontsize=10,
    )
    lines = [
        "# WM / policy experiment results",
        "",
        "All evaluations use floating-base MuJoCo and physical foot-ball contact.",
        "Compare prediction learning within each run; changing scene or action masks changes its validation distribution. "
        + evaluation_context,
        "",
        "| Run | Round | Ball RMSE at 1 s (cm) | Body RMSE at 1 s (cm) | Monitor capture event | Monitor falls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summaries = []
    for run in args.runs:
        logs = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        rows = [r for r in logs if r["stage"] == "round"]
        initial = next(r for r in logs if r["stage"] == "untrained_wm")
        name = run.name.replace("wm_supermimic_k1_", "")
        rounds = [0] + [r["round"] for r in rows]
        vals = [initial["validation"]] + [r["validation"] for r in rows]
        for ax, key in zip(axes[:2], ["ball_rmse_10step_m", "body_rmse_10step_m"]):
            ax.plot(rounds, [100 * v[key] for v in vals], marker="o", label=name)
        for row in rows:
            v, e = row["validation"], row["evaluation"]
            lines.append(
                f"| {name} | {row['round']} | {100 * v['ball_rmse_10step_m']:.2f} | {100 * v['body_rmse_10step_m']:.2f} | {100 * e['captured']:.1f}% | {100 * e['fallen']:.1f}% |"
            )
        paired_path = run / "paired_evaluation.json"
        if paired_path.exists():
            paired = json.loads(paired_path.read_text())
            axes[2].plot(
                range(len(paired)),
                [100 * r["mean"]["pass_success"] for r in paired],
                marker="o",
                label=name,
            )
            baseline = paired[0]
            final = paired[-1]
            baseline_monitor = next(r for r in logs if r["stage"] == "baseline")[
                "evaluation"
            ]
            selected_name, selected_score = "baseline", baseline_monitor["captured"]
            selection_metric = "captured"
            raw_baseline_path = run / "baseline_episodes.json"
            raw_baseline = json.loads(raw_baseline_path.read_text())
            if all("pass_success" in e for e in raw_baseline):
                selection_metric = "pass_success"
                selected_score = np.mean([e[selection_metric] for e in raw_baseline])
            for row in rows:
                e = row["evaluation"]
                raw_path = run / f"evaluation_{row['round']}.json"
                score = (
                    np.mean(
                        [
                            episode.get(selection_metric, False)
                            for episode in json.loads(raw_path.read_text())
                        ]
                    )
                    if selection_metric == "pass_success"
                    else e["captured"]
                )
                if e["fallen"] == 0 and score > selected_score:
                    selected_name, selected_score = (
                        f"round_{row['round']}.pt",
                        score,
                    )
            b = np.array([e["final_target_error"] for e in baseline["episodes_detail"]])
            f = np.array([e["final_target_error"] for e in final["episodes_detail"]])
            rng = np.random.default_rng(123)
            boot = (f - b)[rng.integers(0, len(b), (5000, len(b)))].mean(1)
            summary = {
                "run": name,
                "selected_by_monitor": selected_name,
                "selection_rule": f"Highest {selection_metric} rate with zero falls on monitor seeds 20000-20019; earliest tie; baseline allowed",
                "paired": [
                    {k: v for k, v in item.items() if k != "episodes_detail"}
                    for item in paired
                ],
                "paired_final_target_error_change_m": float((f - b).mean()),
                "paired_bootstrap_95ci_m": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
            summaries.append(summary)
            (run / "selected_checkpoint.json").write_text(
                json.dumps(
                    {"checkpoint": selected_name, "rule": summary["selection_rule"]},
                    indent=2,
                )
            )
    for ax, title in zip(
        axes,
        [
            "Held-out ball prediction, 1 s",
            "Held-out body prediction, 1 s",
            "Independent strict pass success",
        ],
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training round (0 = baseline)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Position RMSE (cm)")
    axes[1].set_ylabel("Position RMSE (cm)")
    axes[2].set_ylabel("Success (%)")
    fig.savefig(args.out / "learning_curves.png", dpi=180)
    fig.savefig(args.out / "learning_curves.pdf")
    plt.close(fig)
    lines += [
        "",
        "Monitor seeds 20000–20019 informed the conservative follow-up; use the independent paired set for final conclusions.",
        "",
        "| Run / checkpoint | Strict success | Capture event | Falls | Target error (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in summaries:
        for item in run["paired"]:
            m = item["mean"]
            lines.append(
                f"| {run['run']} / {item['checkpoint']} | {100 * m['pass_success']:.1f}% | {100 * m['captured']:.1f}% | {100 * m['fallen']:.1f}% | {m['final_target_error']:.3f} |"
            )
    comparisons = []
    by_name = {r.name.replace("wm_supermimic_k1_", ""): r for r in args.runs}
    pair = next(
        (
            pair
            for pair in [
                ("wm_facing_v7_gentle", "wm_facing_v8_gentle_no_wm"),
                ("wm_facing_v5", "wm_facing_v6_no_wm"),
                ("v3", "v4_no_wm"),
            ]
            if all(name in by_name for name in pair)
        ),
        (),
    )
    if pair:
        wm = json.loads((by_name[pair[0]] / "paired_evaluation.json").read_text())
        no_wm = json.loads((by_name[pair[1]] / "paired_evaluation.json").read_text())
        for index in [1, 3]:
            left, right = wm[index], no_wm[index]
            assert [e["seed"] for e in left["episodes_detail"]] == [
                e["seed"] for e in right["episodes_detail"]
            ]
            delta = np.array(
                [int(e["pass_success"]) for e in left["episodes_detail"]]
            ) - np.array([int(e["pass_success"]) for e in right["episodes_detail"]])
            rng = np.random.default_rng(222)
            boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
            ci = np.quantile(boot, [0.025, 0.975])
            comparisons.append(
                {
                    "round": index,
                    "strict_success_rate_difference": float(delta.mean()),
                    "paired_bootstrap_95ci": ci.tolist(),
                }
            )
            lines += [
                "",
                f"WM minus matched no-WM distillation, round {index}: {100 * delta.mean():.1f} percentage points, paired bootstrap 95% CI [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}].",
            ]
    (args.out / "ablation_comparison.json").write_text(
        json.dumps(comparisons, indent=2)
    )
    lines += [
        "",
        "Target error is the ball's distance from the receiver's initial location; it can increase during successful mobile reception. Strict success uses ordered contacts, >=0.3 m ball travel, final hold, and no fall. The bootstrap intervals only cover paired initial-state variation; all training uses one random seed, so they do not establish repeatable gains across training seeds.",
    ]
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
