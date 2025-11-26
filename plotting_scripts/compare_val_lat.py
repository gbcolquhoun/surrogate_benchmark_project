import os
import pandas as pd
import matplotlib.pyplot as plt


CSV_FILES = [
    "hardware-aware-transformers/results/HAT_pareto_front_deduped.csv",  
    "results/20250913_054522_sel=prefix_metric=na_ops=vanilla/pareto_front.csv",  
    "results/20250918_000152_sel=prefix_metric=na_ops=importance/pareto_front.csv",  
]

LABELS = [
    "Baseline HAT",         
    "Custom NSGA2",  
    "Custom NSGA2 + Importance",       
]

OUTPUT_PATH = "results/compare_fronts.png"


def main():
    plt.figure(figsize=(7,5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, path in enumerate(CSV_FILES):
        if not os.path.isfile(path):
            print(f"Skip (not found): {path}")
            continue

        df = pd.read_csv(path)
        if "latency_ms" not in df.columns or "valid_loss" not in df.columns:
            print(f"Skip (missing latency_ms/valid_loss): {path}")
            continue

        label = LABELS[i] if i < len(LABELS) else os.path.splitext(os.path.basename(path))[0]
        plt.scatter(df["latency_ms"], df["valid_loss"], s=25, alpha=0.7,
                    label=label, color=colors[i % len(colors)])

    plt.xlabel("Latency (ms)")
    plt.ylabel("Validation Loss")
    plt.title("Pareto Front Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved figure to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()