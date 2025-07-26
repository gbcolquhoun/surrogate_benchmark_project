#!/usr/bin/env python3
"""
Plot a HAT Pareto front saved by surrogate_benchmark_project.
Usage
-----
python3 plot_pareto.py evo_search_20250718_105647.txt 

The script looks for the file inside
surrogate_benchmark_project/results/ by default.

"""
import argparse, re, os, datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------ #
# robust log-parsers
# ------------------------------------------------------------------ #
def load_objectives(path: Path) -> np.ndarray:
    """Return an (n,2) array of [latency, loss] points."""
    txt = path.read_text()

    # grab the block after the header up to the next header
    hdr = "Pareto Front Objectives (F):"
    start = txt.find(hdr)
    if start == -1:
        raise RuntimeError("Header 'Pareto Front Objectives (F):' not found.")

    tail = txt[start + len(hdr):]
    stop = tail.find("Pareto Front Solutions")
    tail = tail[:stop] if stop != -1 else tail

    # collect every float in order
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", tail)
    if len(nums) % 2:
        raise RuntimeError("Uneven number of floats in objective block.")
    return np.asarray(nums, float).reshape(-1, 2)


def load_population(path: Path, genome_len: int = 40) -> np.ndarray:
    """
    Read *all* ints after the 'Pareto Front Solutions (X):' header
    and chunk them into genomes of length `genome_len`.
    """
    txt = path.read_text()
    hdr = "Pareto Front Solutions (X):"
    start = txt.find(hdr)
    if start == -1:
        raise RuntimeError("Header 'Pareto Front Solutions (X):' not found.")

    # grab text until 'Additional' or end
    tail = txt[start + len(hdr):]
    stop = tail.find("Additional Info")
    if stop != -1:
        tail = tail[:stop]

    # every integer in order
    ints = [int(x) for x in re.findall(r"-?\d+", tail)]
    if len(ints) % genome_len:
        raise RuntimeError(f"Total ints ({len(ints)}) not a multiple of {genome_len}.")

    genomes = np.asarray(ints, int).reshape(-1, genome_len)
    return genomes


def load_meta(path: Path) -> dict:
    """Get 'generations' and 'population size' (may be missing)."""
    txt = path.read_text()
    meta = {}
    gen = re.search(r"Number of generations:\s*(\d+)", txt)
    pop = re.search(r"Population size:\s*(\d+)", txt)
    if gen:
        meta["generations"] = int(gen.group(1))
    if pop:
        meta["population"] = int(pop.group(1))
    return meta


# ------------------------------------------------------------------ #
def plot_pareto(obj: np.ndarray,
                genomes: np.ndarray,
                meta: dict,
                out_path: Path):
    """Create and save the scatter plot."""
    latency, loss = obj[:, 0], obj[:, 1]   # already positive



    decoder_layers = genomes[:, 1].astype(int)          # gene-1 is decoder depth

    colours = {
        1: "tab:gray",
        2: "tab:red",
        3: "tab:purple",
        4: "tab:blue",
        5: "tab:green",
        6: "tab:orange",
    }

    for dl in sorted(colours):                 # loop over 1-6 even if some don’t appear
        idx = decoder_layers == dl
        if idx.any():                          # skip colours with no points
            plt.scatter(latency[idx],
                        loss[idx],
                        label=f"{dl}-layer decoder",
                        color=colours[dl])

    plt.xlabel("Latency (ms)")
    plt.ylabel("Validation Loss")
    nice_time = dt.datetime.now().strftime("%d %B %Y %H:%M")
    sub = f"{nice_time}"
    if meta:
        sub = (f"{sub} | Pop {meta.get('population','?')}"
               f" | Gen {meta.get('generations','?')}")
    plt.title(f"NSGA-II Pareto Front\n{sub}")
    plt.legend()
    plt.grid(True, ls=":")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile",
                        help="log filename (just the file, no path needed)")
    parser.add_argument("--out", default=None,
                        help="PNG filename to save (default: auto-named)")
    args = parser.parse_args()

    # Resolve paths -------------------------------------------------
    base_dir = Path(__file__).parent / "results"
    log_path = base_dir / args.logfile
    if not log_path.exists():
        raise FileNotFoundError(f"Can't find {log_path}")

    out_path = (Path(args.out) if args.out else
                Path("graphs") / (log_path.stem + "_pareto.png"))

    # Parse + plot --------------------------------------------------
    obj = load_objectives(log_path)
    genomes = load_population(log_path)
    meta = load_meta(log_path)

    plot_pareto(obj, genomes, meta, out_path)


if __name__ == "__main__":
    main()