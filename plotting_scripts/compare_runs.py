#!/usr/bin/env python3
# compare_runs.py
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------- IO -------------------- #
def load_run(run_dir: Path):
    """Return dict with: name, F (n,2), X (n,40), meta(dict)."""
    run_dir = Path(run_dir)
    F = pd.read_csv(run_dir / "pareto_F.csv").to_numpy(float)
    X = pd.read_csv(run_dir / "pareto_X.csv").to_numpy(int)

    # meta: support your run_meta.json structure
    meta = {}
    mp = run_dir / "run_meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {}

    # A human-friendly label
    modes = meta.get("modes", {})
    sel = "magnitude" if modes.get("use_weight_magnitude_selection") else "prefix"
    ops = "importance" if modes.get("use_importance_operators") else "vanilla"
    label = f"{run_dir.name} | sel={sel} | ops={ops}"

    # Pull pop/gen where they live in your file
    pop = meta.get("evolution", {}).get("population_size")
    gen = meta.get("evolution", {}).get("generations")
    meta_flat = {"population": pop, "generations": gen, "label": label}

    return {"name": run_dir.name, "dir": run_dir, "F": F, "X": X, "meta": meta_flat}


# ---------------- Pareto helpers (minimize) ---------------- #
def is_dominated(p, q):
    """p dominated by q (minimization): q <= p in all AND q < p in at least one."""
    return np.all(q <= p) and np.any(q < p)

def filter_nondominated(F):
    """Return only non-dominated rows (minimization)."""
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]: 
            continue
        for j in range(len(F)):
            if i == j or not keep[j]:
                continue
            if is_dominated(F[i], F[j]):
                keep[i] = False
                break
    return F[keep]

def hypervolume_2d(F, ref):
    """
    2D HV for minimization w.r.t. ref (worse is larger).
    Assumes F is non-dominated. Sort by x (obj0) ascending and integrate rectangles.
    """
    if len(F) == 0:
        return 0.0
    P = F[np.argsort(F[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in P:
        width  = max(0.0, ref[0] - x)
        height = max(0.0, prev_y - y)
        hv += width * height
        prev_y = min(prev_y, y)
    return hv

def coverage(A, B):
    """Fraction of points in A dominated by at least one point in B."""
    if len(A) == 0:
        return 0.0
    count = 0
    for p in A:
        if any(is_dominated(p, q) for q in B):
            count += 1
    return count / len(A)


# ---------------- Comparison & Plot ---------------- #
def compare_runs(runA, runB, out_png: Path, out_txt: Path):
    A = load_run(runA)
    B = load_run(runB)

    # Re-filter true non-dominated sets (defensive)
    A_nd = filter_nondominated(A["F"])
    B_nd = filter_nondominated(B["F"])

    # Shared reference point (a bit worse than the worst across both)
    both = np.vstack([A_nd, B_nd]) if len(A_nd) and len(B_nd) else (A_nd if len(A_nd) else B_nd)
    ref = np.max(both, axis=0) + np.array([5.0, 0.2]) if len(both) else np.array([210.0, 2.5])

    hv_A = hypervolume_2d(A_nd, ref)
    hv_B = hypervolume_2d(B_nd, ref)
    cov_A_in_B = coverage(A_nd, B_nd)   # fraction of A dominated by B
    cov_B_in_A = coverage(B_nd, A_nd)   # fraction of B dominated by A

    # ---- Plot (old style sizing) ----
    plt.figure(figsize=(9, 6))
    plt.scatter(A_nd[:, 0], A_nd[:, 1], s=36, color="tab:blue",  alpha=0.85, label=A["meta"]["label"])
    plt.scatter(B_nd[:, 0], B_nd[:, 1], s=36, color="tab:orange", alpha=0.85, label=B["meta"]["label"])
    plt.xlabel("Latency (ms)")
    plt.ylabel("Validation Loss")
    # title with pop/gen when available
    a_pop, a_gen = A["meta"]["population"], A["meta"]["generations"]
    b_pop, b_gen = B["meta"]["population"], B["meta"]["generations"]
    title = (f"NSGA-II Pareto Front Comparison\n"
             f"A: Pop {a_pop if a_pop is not None else '?'} | Gen {a_gen if a_gen is not None else '?'}   "
             f"•   B: Pop {b_pop if b_pop is not None else '?'} | Gen {b_gen if b_gen is not None else '?'}")
    plt.title(title)
    plt.grid(True, ls=":")
    plt.legend(loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot → {out_png}")

    # ---- Text report ----
    lines = []
    lines.append("NSGA-II Pareto Front Comparison")
    lines.append("================================")
    lines.append(f"A: {A['name']}")
    lines.append(f"   Pop={A['meta']['population']}  Gen={A['meta']['generations']}  Points={len(A_nd)}")
    lines.append(f"B: {B['name']}")
    lines.append(f"   Pop={B['meta']['population']}  Gen={B['meta']['generations']}  Points={len(B_nd)}")
    lines.append("")
    lines.append(f"Reference point (lat, loss): {ref.tolist()}")
    lines.append(f"Hypervolume A: {hv_A:.6f}")
    lines.append(f"Hypervolume B: {hv_B:.6f}")
    lines.append(f"Coverage A dominated by B: {100*cov_A_in_B:.1f}%")
    lines.append(f"Coverage B dominated by A: {100*cov_B_in_A:.1f}%")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Higher hypervolume is better (more area dominated toward the origin).")
    lines.append("- Coverage shows how much of one front is strictly dominated by the other.")
    lines.append("")
    lines.append("Front A (latency, loss):")
    for p in A_nd[np.argsort(A_nd[:, 0])]:
        lines.append(f"  {p[0]:8.4f}, {p[1]:7.4f}")
    lines.append("Front B (latency, loss):")
    for p in B_nd[np.argsort(B_nd[:, 0])]:
        lines.append(f"  {p[0]:8.4f}, {p[1]:7.4f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines))
    print(f"Saved report → {out_txt}")


# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser(description="Compare two HAT NSGA-II run folders.")
    ap.add_argument("runA")
    ap.add_argument("runB")
    ap.add_argument("--out-stem", default=None,
                    help="Output stem name (defaults to '<runA>_vs_<runB>').")
    args = ap.parse_args()

    runA = Path(args.runA)
    runB = Path(args.runB)
    stem = args.out_stem or f"{runA.name}_vs_{runB.name}"

    out_png = Path("graphs") / f"{stem}.png"
    out_txt = Path("results") / "comparisons" / f"{stem}.txt"

    compare_runs(runA, runB, out_png, out_txt)


if __name__ == "__main__":
    main()