import json
import os
import csv

# ----- mappings -----
EMBED_DIM_MAP = {512: 0, 640: 1}
FFN_DIM_MAP   = {512: 0, 1024: 1, 2048: 2}
HEADS_MAP     = {2: 0, 4: 1}
ARBITRARY_MAP = {-1: 0, 1: 1, 2: 2}

def as_int(x):
    if isinstance(x, (int, float)) and float(x).is_integer():
        return int(x)
    return x

def map_list(values, mapping, name):
    out = []
    for v in values:
        vi = as_int(v)
        if vi not in mapping:
            raise ValueError(f"{name}: value {vi} not in mapping {mapping}")
        out.append(mapping[vi])
    return out

def pad_to_six(codes, pad_val=-1):
    return (codes + [pad_val] * (6 - len(codes)))[:6]

def config_to_genome(cfg, pad_val=-1):
    enc = cfg["encoder"]
    dec = cfg["decoder"]

    enc_layers = 6
    dec_layers = int(as_int(dec["decoder_layer_num"]))
    enc_embed_code = EMBED_DIM_MAP[as_int(enc["encoder_embed_dim"])]
    dec_embed_code = EMBED_DIM_MAP[as_int(dec["decoder_embed_dim"])]

    enc_ffn_codes = map_list(enc["encoder_ffn_embed_dim"][:6], FFN_DIM_MAP, "encoder_ffn_embed_dim")

    dec_ffn_codes = map_list(dec["decoder_ffn_embed_dim"][:dec_layers], FFN_DIM_MAP, "decoder_ffn_embed_dim")
    dec_ffn_codes = pad_to_six(dec_ffn_codes, pad_val)

    enc_heads_codes = map_list(enc["encoder_self_attention_heads"][:6], HEADS_MAP, "encoder_self_attention_heads")

    dec_heads_codes = map_list(dec["decoder_self_attention_heads"][:dec_layers], HEADS_MAP, "decoder_self_attention_heads")
    dec_heads_codes = pad_to_six(dec_heads_codes, pad_val)

    dec_ende_codes = map_list(dec["decoder_ende_attention_heads"][:dec_layers], HEADS_MAP, "decoder_ende_attention_heads")
    dec_ende_codes = pad_to_six(dec_ende_codes, pad_val)

    dec_arbitrary_codes = map_list(dec["decoder_arbitrary_ende_attn"][:dec_layers], ARBITRARY_MAP, "decoder_arbitrary_ende_attn")
    dec_arbitrary_codes = pad_to_six(dec_arbitrary_codes, pad_val)

    genome = (
        [enc_layers, dec_layers, enc_embed_code, dec_embed_code] +
        enc_ffn_codes +
        dec_ffn_codes +
        enc_heads_codes +
        dec_heads_codes +
        dec_ende_codes +
        dec_arbitrary_codes
    )

    if len(genome) != 40:
        raise RuntimeError(f"Genome length is {len(genome)}, expected 40.")
    return genome

def main():
    out_csv = "HAT_genes.csv"
    json_files = [f for f in os.listdir(".") if f.endswith(".json")]
    json_files.sort()

    rows = []
    for fname in json_files:
        with open(fname, "r") as f:
            cfg = json.load(f)
        genome = config_to_genome(cfg)
        rows.append([fname] + genome)

    headers = ["filename"] + [f"gene_{i}" for i in range(40)]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f" Wrote {len(rows)} genomes to {out_csv}")

if __name__ == "__main__":
    main()