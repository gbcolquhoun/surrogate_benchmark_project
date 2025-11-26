#!/usr/bin/env python3
# this one reads a front given a results directory and compares against HAT
import argparse, json, datetime as dt
from pathlib import Path

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


def compute_improvements(obj, hat_latency, hat_loss):
    """
    For each HAT point (L_hat, loss_hat):

      • Loss improvement:
          - Consider all our points with latency <= L_hat.
          - Take best_loss = min loss in that set.
          - abs_improvement = loss_hat - best_loss
          - percent_improvement = 100 * abs_improvement / loss_hat

      • Latency improvement:
          - Consider all our points with loss <= loss_hat.
          - Take best_latency = min latency in that set.
          - latency_improvement_ms = L_hat - best_latency
          - latency_improvement_s  = latency_improvement_ms / 1000
    """
    latency = obj[:, 0]
    loss    = obj[:, 1]

    rows = []
    for L_h, loss_h in zip(hat_latency, hat_loss):
        # ---- Loss-side improvement: match or beat latency, improve loss ----
        mask_lat = latency <= L_h
        if np.any(mask_lat):
            best_loss = float(np.min(loss[mask_lat]))
            abs_improve = loss_h - best_loss
            pct_improve = 100.0 * abs_improve / loss_h
        else:
            best_loss = None
            abs_improve = None
            pct_improve = None

        # ---- Latency-side improvement: match or beat loss, improve latency ----
        mask_loss = loss <= loss_h
        if np.any(mask_loss):
            best_latency = float(np.min(latency[mask_loss]))
            lat_improve_ms = L_h - best_latency
            lat_improve_s  = lat_improve_ms / 1000.0
        else:
            best_latency = None
            lat_improve_ms = None
            lat_improve_s  = None

        rows.append(
            (
                L_h,
                loss_h,
                best_loss,
                abs_improve,
                pct_improve,
                best_latency,
                lat_improve_ms,
                lat_improve_s,
            )
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "hat_latency_ms",
            "hat_loss",
            "our_best_loss_at_or_below_hat_latency",
            "abs_loss_improvement",
            "percent_loss_improvement",
            "our_best_latency_ms_at_or_below_hat_loss",
            "latency_improvement_ms",
            "latency_improvement_s",
        ],
    )

    return df


def main():
    # HAT paper reference points (latency in ms, loss in NLL/token)
    hat_latency = np.array(
        [
            48.58507215976715,
            79.34585809707642,
            110.8258843421936,
            143.4543013572693,
            175.49684047698975,
        ],
        dtype=float,
    )

    hat_loss = np.array(
        [
            1.4955,
            1.3876,
            1.3215,
            1.3095,
            1.3048,
        ],
        dtype=float,
    )

    ap = argparse.ArgumentParser(
        description="Compute improvement over HAT baselines from a results directory."
    )
    ap.add_argument(
        "run_dir",
        help="Path to results directory (contains pareto_F.csv, pareto_X.csv, run_meta.json).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV for improvements (default: <run_dir>/improvement_over_HAT.csv)",
    )
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

    df_imp = compute_improvements(obj, hat_latency, hat_loss)
    print("\n=== Improvement over HAT baseline ===")
    print(df_imp.to_string(index=False))

    if args.out:
        imp_path = Path(args.out)
    else:
        imp_path = run_dir / "improvement_over_HAT.csv"

    df_imp.to_csv(imp_path, index=False)
    print(f"\nSaved improvement table → {imp_path}")


if __name__ == "__main__":
    main()