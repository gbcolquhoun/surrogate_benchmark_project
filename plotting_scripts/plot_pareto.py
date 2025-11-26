#!/usr/bin/env python3
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

    try: pop = int(pop)
    except: pop = "?"
    try: gen = int(gen)
    except: gen = "?"

    return obj, X, {"population": pop, "generations": gen}


def plot_pareto(obj: np.ndarray, genomes: np.ndarray, meta: dict, out_path: Path):
    """Old-style plot: bigger fig, two-line title, dotted grid, legend by decoder depth."""
    latency, loss = obj[:, 0], obj[:, 1]
    dec_layers = genomes[:, 1].astype(int)   # gene-1 = decoder depth

    colours = {
        1: "tab:gray",
        2: "tab:red",
        3: "tab:purple",
        4: "tab:blue",
        5: "tab:green",
        6: "tab:orange",
    }

    plt.figure(figsize=(12, 4))
    for dl in sorted(colours):
        idx = dec_layers == dl
        if idx.any():
            plt.scatter(latency[idx], loss[idx], label=f"dec={dl}", color=colours[dl])

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


def main():
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
    plot_pareto(obj, X, meta, out)


if __name__ == "__main__":
    main()