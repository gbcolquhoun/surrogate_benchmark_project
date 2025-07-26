#!/usr/bin/env python3
"""
make_iwslt_split.py  –  sample N sentence-pairs from IWSLT XML files
✓ works on Python 3.6+
"""
import argparse, random, xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

# ----------------------------------------------------------------------
def extract_sentences(xml_path: Path) -> List[str]:
    """Return all <seg> text strings from an IWSLT XML file."""
    tree = ET.parse(xml_path)
    return [seg.text.strip() for seg in tree.iterfind(".//seg")]

def sample_iwslt(
    xml_de: List[Path],
    xml_en: List[Path],
    n: int,
    out_prefix: Path,
    seed: int = 42,
) -> None:
    """Randomly sample *n* aligned pairs and write *.de / *.en files."""
    de_sents = [s for path in xml_de for s in extract_sentences(path)]
    en_sents = [s for path in xml_en for s in extract_sentences(path)]

    if len(de_sents) != len(en_sents):
        raise ValueError("Source and target corpora have different lengths!")

    if n > len(de_sents):
        raise ValueError(f"Requested {n} > corpus size {len(de_sents)}")

    rng = random.Random(seed)
    idx = rng.sample(range(len(de_sents)), n)

    out_de = out_prefix.with_suffix(".de")
    out_en = out_prefix.with_suffix(".en")
    out_de.parent.mkdir(parents=True, exist_ok=True)

    with out_de.open("w", encoding="utf-8") as fde, \
         out_en.open("w", encoding="utf-8") as fen:
        for i in idx:
            fde.write(de_sents[i] + "\n")
            fen.write(en_sents[i] + "\n")

    print(f"Wrote {n} pairs → {out_de} / {out_en}")

# ----------------------------------------------------------------------
def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--de", nargs="+", required=True, type=Path,
                   help="German XML files")
    p.add_argument("--en", nargs="+", required=True, type=Path,
                   help="English XML files (same order)")
    p.add_argument("--n",  type=int, required=True,
                   help="Number of pairs to sample")
    p.add_argument("--out", type=Path, required=True,
                   help="Output prefix (no extension)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    sample_iwslt(a.de, a.en, a.n, a.out, a.seed)

if __name__ == "__main__":
    _cli()