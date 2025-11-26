# score_pareto.py
import os, csv
from pathlib import Path
from evo_search.searcher import evo_search



REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "results/20250918_000152_sel=prefix_metric=na_ops=importance/pareto_front.csv"
DESIGN_SPACE = REPO_ROOT / "configs" / "hat_design_space.yaml"
SEARCH_PARAMS = REPO_ROOT / "configs" / "evo_search_params.yaml"

def read_genes(csv_path):
    genes, header = [], None
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
        if not rows:
            return [], None
        if rows[0] and rows[0][0].strip().lower().startswith("gene_"):
            header, rows = rows[0], rows[1:]
        for row in rows:
            if not row or all(c.strip() == "" for c in row):
                continue
            row = row[:40]  # ensure length 40
            genes.append([int(x.strip()) for x in row])
    return genes, header

def write_scored(csv_path, header, genes, scores):
    out_path = Path(csv_path).with_name(Path(csv_path).stem + "_scored.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow([*header[:40], "bleu"])
        for g, s in zip(genes, scores):
            w.writerow([*g, s])
    return out_path

def main():
    # Run from this script's directory (like main.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Starting score_pareto.py")
    print(" initializing evo_search")
    search = evo_search(DESIGN_SPACE, SEARCH_PARAMS)

    genes, header = read_genes(CSV_PATH)
    if not genes:
        print("No genes found in CSV.")
        return

    scores = []
    for i, gene in enumerate(genes, 1):
        s = search.quick_bleu_eval(gene)
        scores.append(s)
        print(f"row {i}: {s}")

    out_path = write_scored(CSV_PATH, header, genes, scores)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()