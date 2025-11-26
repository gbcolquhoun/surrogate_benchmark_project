#!/usr/bin/env python3
import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evo_search.architectures.hat_arch import hat_architecture


def run_elitist_random(problem, population_size, generations):
    sampling = problem.custom_sampling()
    elite_X = None
    elite_F = None
    history = []

    for gen in range(1, generations + 1):
        X_new = sampling.do(problem, population_size, None)
        F_new = problem.evaluate(X_new, return_values_of="F")

        if elite_X is not None:
            X_comb = np.vstack([elite_X, X_new])
            F_comb = np.vstack([elite_F, F_new])
        else:
            X_comb, F_comb = X_new, F_new

        idx = np.argsort(F_comb[:, 1])[:population_size]
        elite_X = X_comb[idx]
        elite_F = F_comb[idx]

        for i, f in enumerate(F_comb):
            history.append(
                {
                    "algo": "random",
                    "gen": int(gen),
                    "idx": int(i),
                    "latency_ms": float(f[0]),
                    "valid_loss": float(f[1]),
                }
            )

    return pd.DataFrame(history)


def main():
    parser = argparse.ArgumentParser(
        description="Run elitist random baseline and plot vs NSGA-II for a given validation run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the results/<run_id> directory produced by run_validation_study().",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    hist_path = os.path.join(run_dir, "validation_history.csv")
    meta_path = os.path.join(run_dir, "validation_meta.json")

    df_nsga = pd.read_csv(hist_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if "algo" not in df_nsga.columns:
        df_nsga["algo"] = "nsga2"

    design_space = meta.get("design_space")
    population_size = int(meta["evolution"]["population_size"])
    generations = int(meta["evolution"]["generations"])

    problem = hat_architecture(design_space)

    print(f"[random] Running elitist random search: pop={population_size}, gens={generations}")
    df_rand = run_elitist_random(problem, population_size, generations)

    random_hist_path = os.path.join(run_dir, "random_history.csv")
    df_rand.to_csv(random_hist_path, index=False)
    print(f"[random] Saved random history -> {random_hist_path}")

    df_all = pd.concat([df_nsga, df_rand], ignore_index=True)

    df_summary = (
        df_all.groupby(["algo", "gen"], as_index=False)
        .agg(
            best_valid_loss=("valid_loss", "min"),
            mean_valid_loss=("valid_loss", "mean"),
            median_valid_loss=("valid_loss", "median"),
        )
        .sort_values(["algo", "gen"])
    )

    summary_path = os.path.join(run_dir, "random_vs_nsga_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[plot] Saved combined summary -> {summary_path}")

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo, color, label in [
        ("nsga2", "tab:red", "NSGA-II"),
        ("random", "tab:blue", "Random (elitist)"),
    ]:
        sub = df_summary[df_summary["algo"] == algo]
        if not sub.empty:
            ax.plot(
                sub["gen"],
                sub["best_valid_loss"],
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=label,
                color=color,
            )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Validation Loss")
    ax.legend()
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    plot_path = os.path.join(run_dir, "random_vs_nsga.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"[plot] Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()