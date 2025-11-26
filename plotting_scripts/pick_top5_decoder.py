# pick_top5_decoder.py
import os, csv
from pathlib import Path
from collections import defaultdict

# to be edited 
CSV_IN = "results/20250918_000152_sel=prefix_metric=na_ops=importance/pareto_front.csv"


def has_col(header, name):
    return any(h.strip().lower() == name for h in header)

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def to_int_list(row):
    """Convert the first 40 gene columns to ints, ignoring conversion errors."""
    out = []
    for x in row[:40]:
        try:
            out.append(int(float(x)))
        except Exception:
            out.append(x)
    return out

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    in_path = Path(CSV_IN)
    out_path = in_path.with_name(in_path.stem + "_picked_by_decoder_layers.csv")

    with open(in_path, "r", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
        if not rows:
            print("Empty CSV.")
            return
        header = rows[0]
        data = rows[1:]

    idx = {name: i for i, name in enumerate(header)}
    required = ["gene_1", "latency_ms", "valid_loss"]
    for col in required:
        if col not in idx:
            raise ValueError(f"Missing required column: {col}")
    has_bleu = "bleu" in idx

    groups = defaultdict(list)
    for row in data:
        if not row or len(row) < len(header):
            continue
        dec_layers = int(row[idx["gene_1"]])
        groups[dec_layers].append(row)

    picked = []
    for dec_layers, rows_g in sorted(groups.items()):
        if has_bleu:
            rows_g.sort(
                key=lambda r: (
                    -(to_float(r[idx["bleu"]], float("-inf"))),
                    to_float(r[idx["latency_ms"]], float("inf")),
                    to_float(r[idx["valid_loss"]], float("inf")),
                )
            )
        else:
            rows_g.sort(
                key=lambda r: (
                    to_float(r[idx["valid_loss"]], float("inf")),
                    to_float(r[idx["latency_ms"]], float("inf")),
                )
            )

        best = rows_g[0]
        picked.append(best)

        # print info
        b = to_float(best[idx["bleu"]], None) if has_bleu else None
        lat = to_float(best[idx["latency_ms"]], None)
        vl = to_float(best[idx["valid_loss"]], None)
        print(
            f"gene_1={dec_layers} -> picked row with "
            + (f"BLEU={b:.3f}, " if b is not None else "")
            + f"latency_ms={lat:.3f}, valid_loss={vl:.6f}"
        )
        # print the gene in python-ready integer list form
        gene_ints = to_int_list(best)
        print("Gene:", gene_ints, "\n")

    # write out file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(picked)

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()