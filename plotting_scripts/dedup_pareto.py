# dedup_pareto.py
import csv, sys, os
from pathlib import Path

def dedup_pareto(csv_path):
    csv_path = Path(csv_path)
    out_path = csv_path.with_name(csv_path.stem + "_deduped.csv")

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Empty file.")
        return

    header = rows[0]
    data = rows[1:]

    seen = set()
    deduped = []

    for row in data:
        # treat the gene portion as the unique key (everything before latency/loss columns)
        gene_key = tuple(row[:-2])  # assumes last 2 columns are latency_ms and valid_loss
        if gene_key not in seen:
            seen.add(gene_key)
            deduped.append(row)

    print(f"Original rows: {len(data)} | Unique rows: {len(deduped)}")

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(deduped)

    print(f"Saved deduped file to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dedup_pareto.py path/to/pareto_front.csv")
        sys.exit(1)
    dedup_pareto(sys.argv[1])