import pandas as pd
import json
import os

class HATGeneConverter:
    def __init__(self, super_encoder_layer_num=6, super_decoder_layer_num=6):
        self.super_encoder_layer_num = super_encoder_layer_num
        self.super_decoder_layer_num = super_decoder_layer_num

    def gene2config(self, gene):
        config = {
            'encoder': {
                'encoder_embed_dim': None,
                'encoder_layer_num': None,
                'encoder_ffn_embed_dim': None,
                'encoder_self_attention_heads': None,
            },
            'decoder': {
                'decoder_embed_dim': None,
                'decoder_layer_num': None,
                'decoder_ffn_embed_dim': None,
                'decoder_self_attention_heads': None,
                'decoder_ende_attention_heads': None,
                'decoder_arbitrary_ende_attn': None
            }
        }
        current_index = 0

        config['encoder']['encoder_embed_dim'] = gene[current_index]; current_index += 1
        config['encoder']['encoder_layer_num'] = gene[current_index]; current_index += 1

        config['encoder']['encoder_ffn_embed_dim'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num
        config['encoder']['encoder_self_attention_heads'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num

        config['decoder']['decoder_embed_dim'] = gene[current_index]; current_index += 1
        config['decoder']['decoder_layer_num'] = gene[current_index]; current_index += 1

        config['decoder']['decoder_ffn_embed_dim'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num
        config['decoder']['decoder_self_attention_heads'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num
        config['decoder']['decoder_ende_attention_heads'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num
        config['decoder']['decoder_arbitrary_ende_attn'] = gene[current_index: current_index + self.super_decoder_layer_num]

        return config


def main():
    input_csv = "hardware-aware-transformers/results/HAT_pareto_front_deduped.csv"     
    output_dir = "HAT_front"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    converter = HATGeneConverter()

    for i, row in df.iterrows():
        gene = row.values[:-2]  # exclude latency_ms and valid_loss
        config = converter.gene2config(gene.tolist())

        # include metadata if desired
        config["latency_ms"] = row["latency_ms"]
        config["valid_loss"] = row["valid_loss"]

        out_path = os.path.join(output_dir, f"config_{i:02d}.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()