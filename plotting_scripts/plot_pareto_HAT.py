#!/usr/bin/env python3
# this one plots a front given a results directory but throws in some fun extra points courtesy of HAT to compare with
import argparse, json, datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run(run_dir: Path, genome_len: int = 40):
    """Load objectives (latency, loss), genomes, and Pop/Gen from run_dir."""
    pf = run_dir / "pareto_F.csv"
    px = run_dir / "pareto_X.csv"
    meta_p = run_dir / "run_meta.json"

    if not pf.exists() or not px.exists() or not meta_p.exists():
        raise FileNotFoundError("Expected pareto_F.csv, pareto_X.csv, run_meta.json in the run directory.")

    # Objectives
    F = pd.read_csv(pf)
    # Prefer columns named latency/loss (any case), else take first two numeric cols.
    cols = {c.lower(): c for c in F.columns}
    if "latency" in cols and "loss" in cols:
        obj = F[[cols["latency"], cols["loss"]]].to_numpy(float)
    else:
        obj = F.select_dtypes(include=[np.number]).iloc[:, :2].to_numpy(float)

    # Genomes
    X = pd.read_csv(px).select_dtypes(include=[np.number]).to_numpy(int)
    if X.shape[1] != genome_len:
        raise RuntimeError(f"pareto_X.csv has {X.shape[1]} columns; expected {genome_len}")

    # Meta (Pop/Gen)
    meta = json.loads(meta_p.read_text())
    evo  = meta.get("evolution", {})
    sp   = meta.get("search_params", {})

    pop = evo.get("population_size", sp.get("population_size"))
    gen = evo.get("generations",     sp.get("evolution_iterations"))

    try:
        pop = int(pop)
    except Exception:
        pop = "?"
    try:
        gen = int(gen)
    except Exception:
        gen = "?"

    return obj, X, {"population": pop, "generations": gen}


def plot_pareto(obj: np.ndarray, genomes: np.ndarray, meta: dict, out_path: Path, hat_latency, hat_loss):
    """
    Plot Pareto front:
      - All evo_search points in black.
      - 5 HAT-paper reference points in red.
    """
    latency, loss = obj[:, 0], obj[:, 1]

   

    plt.figure(figsize=(12, 4))

    # Evo search points (current run) in black
    plt.scatter(latency, loss, color="black", s=20, label="Genetic Algorithm")

    # HAT paper points in red
    plt.scatter(hat_latency, hat_loss, color="red", s=40, marker="o", label="HAT Benchmark")

    plt.xlabel("Predicted Latency (ms)")
    plt.ylabel("Validation Loss (NLL/Token)")

    now = dt.datetime.now().strftime("%d %b %Y %H:%M")
    plt.title(f"NSGA-II Pareto Front | Pop {meta.get('population','?')} | Gen {meta.get('generations','?')}")

    plt.legend()
    plt.grid(True, ls=":")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    print(f"Saved → {out_path}")

def compute_improvements(obj, hat_latency, hat_loss):
    latency = obj[:,0]
    loss    = obj[:,1]

    rows = []
    for L_h, loss_h in zip(hat_latency, hat_loss):
        # find all points at or below this latency
        mask = latency <= L_h
        if not np.any(mask):
            best_loss = None
        else:
            best_loss = np.min(loss[mask])

        if best_loss is None:
            rows.append((L_h, loss_h, None, None, None))
        else:
            abs_improve = loss_h - best_loss
            pct_improve = 100 * abs_improve / loss_h
            rows.append((L_h, loss_h, best_loss, abs_improve, pct_improve))

    df = pd.DataFrame(rows, columns=[
        "hat_latency", "hat_loss",
        "our_best_loss", "abs_improvement", "percent_improvement"
    ])

    return df

def main():
     # HAT paper reference points
    hat_latency = np.array([
        48.58507215976715,
        79.34585809707642,
        110.8258843421936,
        143.4543013572693,
        175.49684047698975,
    ], dtype=float)

    hat_loss = np.array([
        1.4955,
        1.3876,
        1.3215,
        1.3095,
        1.3048,
    ], dtype=float)
    
    ap = argparse.ArgumentParser(description="Plot Pareto front from a results directory.")
    ap.add_argument("run_dir", help="Path to results directory (contains pareto_F.csv, pareto_X.csv, run_meta.json).")
    ap.add_argument("--out", default=None, help="Output PNG (default: graphs/<run_dir_name>_pareto.png)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        # also try ./results/<name>
        maybe = Path(__file__).parent / "results" / args.run_dir
        if maybe.is_dir():
            run_dir = maybe
        else:
            raise FileNotFoundError(f"Not a directory: {args.run_dir}")

    obj, X, meta = load_run(run_dir, genome_len=40)

    out = Path(args.out) if args.out else (Path(__file__).parent / "graphs" / f"{run_dir.name}_pareto.png")
    #plot_pareto(obj, X, meta, out)
    df_imp = compute_improvements(obj, hat_latency, hat_loss)
    print("\n=== Improvement over HAT baseline ===")
    print(df_imp.to_string(index=False))
    imp_path = run_dir / "improvement_over_HAT.csv"
    df_imp.to_csv(imp_path, index=False)
    print(f"Saved improvement table → {imp_path}")

if __name__ == "__main__":
    main()