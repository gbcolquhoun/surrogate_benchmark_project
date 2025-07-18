# /home/graham/Documents/surrogate_benchmark_project/evo_search/architectures/hat_arch.py
from __future__ import annotations
import random
import numpy as np
from .base_architecture import BaseArchitecture
import torch
import re
from fairseq.models.transformer import TransformerModel
from fairseq.models.transformer_super import TransformerSuperModel
import pandas as pd
import yaml
import subprocess
import os, pathlib, sys
from latency_predictor import LatencyPredictor
import time
from fairseq.sequence_generator import SequenceGenerator
import sacrebleu
from fairseq import checkpoint_utils, tasks, utils, options

HAT_REPO = pathlib.Path(__file__).resolve().parents[3] / "hardware-aware-transformers"
ckpt_rel = "latency_dataset/predictors/iwslt14deen_gpu_titanxp.pt"


class hat_architecture(BaseArchitecture):
   
    def __init__(self, design_space):
        self.fixed_length = design_space.get('fixed_length')
        num_obj = len(design_space.get('metrics'))
        search_space = design_space.get('search_space', {})
        xl, xu = self.build_bounds(search_space)
        
        ckpt_path = HAT_REPO / ckpt_rel
        FEATURE_NORM = [640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2]  # from paper
        LAT_NORM     = 200                                      # from paper
        self.latency_predictor = LatencyPredictor(
            ckpt_path        = ckpt_path,
            lat_dataset_path = "",      # not needed for inference
            feature_norm     = FEATURE_NORM,
            lat_norm         = LAT_NORM,
            feature_dim      = 10,
            hidden_dim       = 400,
            hidden_layer_num = 3,
        )
        self.latency_predictor.load_ckpt()


        # set up some constants for supertransformer
        self.MODEL_DIR = "./"  
        self.CHECKPOINT_FILE = "/home/graham/Documents/hardware-aware-transformers/HAT_iwslt14deen_super_space1.pt"
        self.DATA_BIN = "/home/graham/Documents/hardware-aware-transformers/data/binary/iwslt14_de_en"
        #self.DATA_BIN = "/home/graham/Documents/data/binary/iwslt14_de_en_valid1000"
        self.SAMPLE_CONFIG_PATH = "/home/graham/Documents/surrogate_benchmark_project/output/intermediate_config.yaml"


        # load the supertransfomer
        hub_interface = TransformerSuperModel.from_pretrained(
            self.MODEL_DIR,
            checkpoint_file=self.CHECKPOINT_FILE,
            data_name_or_path=self.DATA_BIN,
            verbose = False
        )
        self.super_model = hub_interface.models[0]
        self.super_model.eval()
        self.task = hub_interface.task

        legacy_args = self.task.args                 # a Namespace

        if not hasattr(legacy_args, "dataset_impl"):
            legacy_args.dataset_impl = "mmap"

        if not hasattr(legacy_args, "combine"):
            legacy_args.combine = False
        self.task.load_dataset("valid")
        full_valid   = self.task.dataset('valid')
        self.eval_samples = [full_valid[i] for i in range(min(200, len(full_valid)))]

        super().__init__(n_var=self.fixed_length, n_obj=num_obj, n_ieq_constr=0, xl=xl, xu=xu, vtype=int)
        
        
    def build_bounds(self, search_space):
        """
        Build lower and upper bounds (xl and xu) for the genome,
        based on the search space definitions and mapping dictionaries.
        
        The genome is ordered as follows (total length 40):
        0. Encoder layer num (1 gene)
        1. Decoder layer num (1 gene)
        2. Encoder embed dim (1 gene)
        3. Decoder embed dim (1 gene)
        4. Encoder FFN embed dims (6 genes)
        5. Decoder FFN embed dims (6 genes)
        6. Encoder self-attention heads (6 genes)
        7. Decoder self-attention heads (6 genes)
        8. Decoder encoder-decoder heads (6 genes)
        9. Decoder arbitrary attention (6 genes)
        """
        xl = []  # lower bounds
        xu = []  # upper bounds

        # Identity mapping for layer numbers: 1->1, 2->2, ..., 6->6.
        layer_num_mapping = {i: i for i in range(1, 7)}
        
        # 0. Encoder layer num (1 gene)
        enc_layer_choices = search_space["encoder-layer-num-choice"]
        lb = min(enc_layer_choices)
        ub = max(enc_layer_choices)
        xl.append(lb)
        xu.append(ub)
        
        # 1. Decoder layer num (1 gene)
        dec_layer_choices = search_space["decoder-layer-num-choice"]
        lb = min(dec_layer_choices)
        ub = max(dec_layer_choices)
        xl.append(lb)
        xu.append(ub)
        
        # Mapping for embed dims: {512:0, 640:1}
        embed_dim_mapping = {512: 0, 640: 1}
        
        # 2. Encoder embed dim (1 gene)
        enc_embed_choices = search_space["encoder-embed-choice"]
        mapped = [embed_dim_mapping[x] for x in enc_embed_choices]
        xl.append(min(mapped))
        xu.append(max(mapped))
        
        # 3. Decoder embed dim (1 gene)
        dec_embed_choices = search_space["decoder-embed-choice"]
        mapped = [embed_dim_mapping[x] for x in dec_embed_choices]
        xl.append(min(mapped))
        xu.append(max(mapped))
        
        # Mapping for FFN embed dims: {512:0, 1024:1, 2048:2}
        ffn_mapping = {512: 0, 1024: 1, 2048: 2}
        
        # 4. Encoder FFN embed dims (6 genes)
        enc_ffn_choices = search_space["encoder-ffn-embed-dim-choice"]
        mapped = [ffn_mapping[x] for x in enc_ffn_choices]
        lb = min(mapped)
        ub = max(mapped)
        for _ in range(6):
            xl.append(lb)
            xu.append(ub)
            
        # 5. Decoder FFN embed dims (6 genes)
        dec_ffn_choices = search_space["decoder-ffn-embed-dim-choice"]
        mapped = [ffn_mapping[x] for x in dec_ffn_choices]
        lb = min(mapped)
        ub = max(mapped)
        for _ in range(6):
            xl.append(lb)
            xu.append(ub)
            
        # Mapping for self-attention heads: {2:0, 4:1}
        sa_mapping = {2: 0, 4: 1}
        
        # 6. Encoder self-attention heads (6 genes)
        enc_sa_choices = search_space["encoder-self-attention-heads-choice"]
        mapped = [sa_mapping[x] for x in enc_sa_choices]
        lb = min(mapped)
        ub = max(mapped)
        for _ in range(6):
            xl.append(lb)
            xu.append(ub)
            
        # 7. Decoder self-attention heads (6 genes)
        dec_sa_choices = search_space["decoder-self-attention-heads-choice"]
        mapped = [sa_mapping[x] for x in dec_sa_choices]
        lb = min(mapped)
        ub = max(mapped)
        for _ in range(6):
            xl.append(lb)
            xu.append(ub)
            
        # 8. Decoder encoder-decoder heads (6 genes)
        dec_ende_choices = search_space["decoder-ende-attention-heads-choice"]
        mapped = [sa_mapping[x] for x in dec_ende_choices]
        lb = min(mapped)
        ub = max(mapped)
        for _ in range(6):
            xl.append(lb)
            xu.append(ub)
            
        # Mapping for arbitrary encoder-decoder attention: {-1:0, 1:1, 2:2}
        arbitrary_mapping = {-1: 0, 1: 1, 2: 2}
        
        # 9. Decoder arbitrary attention (6 genes)
        dec_arbitrary_choices = search_space["decoder-arbitrary-ende-attn-choice"]
        mapped = [arbitrary_mapping[x] for x in dec_arbitrary_choices]
        lb = min(mapped)
        ub = max(mapped)
        for _ in range(6):
            xl.append(lb)
            xu.append(ub)
            
        # Convert to numpy arrays and return.
        return np.array(xl), np.array(xu)
    
    def new_evaluate_accuracy(self, genome: np.ndarray):
        bleu, enc_ms, dec_ms = 0.0, 0.0, 0.0          # default fallback

        # ----- 1. genome → sub-transformer cfg & activate it -----
        cfg = self._genome_to_subtransformer_cfg(genome)
        self.super_model.set_sample_config(cfg)
        model = self.super_model.cuda() if torch.cuda.is_available() else self.super_model
        model.eval()

        # ----- 2. timed generator -----
        gen = TimedSequenceGenerator(
        self.task,                 # <-- pass the task explicitly
        [model],
        beam_size=1,
        max_len_b=256,
        eos = self.task.target_dictionary.eos()
        )

        sys_out, ref_out, enc_list, dec_list = [], [], [], []
        with torch.no_grad():
            for sample in self.eval_samples:
                sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample
                hypos, enc_t, dec_t = gen.generate([model], sample)
                enc_list.append(enc_t);  dec_list.append(dec_t)

                hyp_tok = hypos[0][0]["tokens"].int().cpu()
                tgt_tok = sample["target"].int().cpu()
                sys_out.append(self.task.target_dictionary.string(hyp_tok))
                ref_out.append(self.task.target_dictionary.string(tgt_tok))

        bleu   = sacrebleu.corpus_bleu(sys_out, [ref_out]).score
        enc_ms = float(np.mean(enc_list))
        dec_ms = float(np.mean(dec_list))
        return bleu, enc_ms, dec_ms


    def _genome_to_subtransformer_cfg(self, genome: np.ndarray) -> dict:
        """
        Convert a 40-gene chromosome to the nested sub-Transformer config
        understood by HAT (and by self.latency_predictor).

        Returns
        -------
        dict  with "encoder" and "decoder" fields containing plain Python
            ints / lists (no numpy types!).
        """
        # ---- inverse lookup tables -----------------------------------
        embed_dim_inv       = {0: 512, 1: 640}
        ffn_dim_inv         = {0: 512, 1: 1024, 2: 2048}
        nheads_inv          = {0: 2,   1: 4}
        arbitrary_attn_inv  = {0: -1,  1: 1,  2: 2}

        # ---- global genes --------------------------------------------
        enc_layers   = int(genome[0])
        dec_layers   = int(genome[1])
        enc_emb_dim  = embed_dim_inv[int(genome[2])]
        dec_emb_dim  = embed_dim_inv[int(genome[3])]

        # ---- per-layer blocks (each padded to 6 -> slice) ------------
        enc_ffn  = [ffn_dim_inv[int(x)]        for x in genome[4:10][:enc_layers]]
        dec_ffn  = [ffn_dim_inv[int(x)]        for x in genome[10:16][:dec_layers]]

        enc_sa   = [nheads_inv[int(x)]         for x in genome[16:22][:enc_layers]]
        dec_sa   = [nheads_inv[int(x)]         for x in genome[22:28][:dec_layers]]

        dec_ende = [nheads_inv[int(x)]         for x in genome[28:34][:dec_layers]]
        dec_arbi = [arbitrary_attn_inv[int(x)] for x in genome[34:40][:dec_layers]]

        # ---- pack into the format HAT expects ------------------------
        return {
            "encoder": {
                "encoder_embed_dim"            : enc_emb_dim,
                "encoder_layer_num"            : enc_layers,
                "encoder_ffn_embed_dim"        : enc_ffn,
                "encoder_self_attention_heads" : enc_sa,
            },
            "decoder": {
                "decoder_embed_dim"            : dec_emb_dim,
                "decoder_layer_num"            : dec_layers,
                "decoder_ffn_embed_dim"        : dec_ffn,
                "decoder_self_attention_heads" : dec_sa,
                "decoder_ende_attention_heads" : dec_ende,
                "decoder_arbitrary_ende_attn"  : dec_arbi,
            },
        }
    def evaluate_accuracy(self, genome):
        """
        Given a candidate genome (list of 40 discrete genes), extract the configuration 
        parameters and compute a BLEU score
        """
        
        encoder_layer_num = genome[0]   # Already in [1,6]
        decoder_layer_num = genome[1]   # Already in [1,6]
        embed_dim_mapping_inv = {0: 512, 1: 640}
        ffn_embed_dim_mapping_inv = {0: 512, 1: 1024, 2: 2048}
        sa_mapping_inv = {0: 2, 1: 4}
        arbitrary_mapping_inv = {0: -1, 1: 1, 2: 2}
        encoder_embed_dim = embed_dim_mapping_inv[genome[2]]
        decoder_embed_dim = embed_dim_mapping_inv[genome[3]]
        enc_ffn_genes = genome[4:10]
        dec_ffn_genes = genome[10:16]
        enc_sa_genes = genome[16:22]
        dec_sa_genes = genome[22:28]
        dec_ende_genes = genome[28:34]
        dec_arbitrary_genes = genome[34:40]
        
        # Extract only the entries corresponding to the actual number of layers:
        enc_ffn_values = [ffn_embed_dim_mapping_inv[val] for val in enc_ffn_genes[:encoder_layer_num]]
        dec_ffn_values = [ffn_embed_dim_mapping_inv[val] for val in dec_ffn_genes[:decoder_layer_num]]
        enc_sa_values  = [sa_mapping_inv[val] for val in enc_sa_genes[:encoder_layer_num]]
        dec_sa_values  = [sa_mapping_inv[val] for val in dec_sa_genes[:decoder_layer_num]]
        dec_ende_values = [sa_mapping_inv[val] for val in dec_ende_genes[:decoder_layer_num]]
        dec_arbitrary_values = [arbitrary_mapping_inv[val] for val in dec_arbitrary_genes[:decoder_layer_num]]
        
        # Construct a configuration dictionary from the genome:
        config = {
            "encoder": {
                "encoder_embed_dim": encoder_embed_dim,
                "encoder_ffn_embed_dim": enc_ffn_values,
                "encoder_layer_num": encoder_layer_num,
                "encoder_self_attention_heads": enc_sa_values,
            },
            "decoder": {
                "decoder_embed_dim": decoder_embed_dim,
                "decoder_ffn_embed_dim": dec_ffn_values,
                "decoder_layer_num": decoder_layer_num,
                "decoder_self_attention_heads": dec_sa_values,
                "decoder_ende_attention_heads": dec_ende_values,
                "decoder_arbitrary_ende_attn": dec_arbitrary_values,
            }
        }
        
        #print("Extracted configuration from genome:")
        #print(config)

        flattened = {
            "encoder-embed-dim-subtransformer": config["encoder"]["encoder_embed_dim"],
            "decoder-embed-dim-subtransformer": config["decoder"]["decoder_embed_dim"],

            "encoder-ffn-embed-dim-all-subtransformer": config["encoder"]["encoder_ffn_embed_dim"],
            "decoder-ffn-embed-dim-all-subtransformer": config["decoder"]["decoder_ffn_embed_dim"],

            "encoder-layer-num-subtransformer": config["encoder"]["encoder_layer_num"],
            "decoder-layer-num-subtransformer": config["decoder"]["decoder_layer_num"],

            "encoder-self-attention-heads-all-subtransformer": config["encoder"]["encoder_self_attention_heads"],
            "decoder-self-attention-heads-all-subtransformer": config["decoder"]["decoder_self_attention_heads"],
            "decoder-ende-attention-heads-all-subtransformer": config["decoder"]["decoder_ende_attention_heads"],
            "decoder-arbitrary-ende-attn-all-subtransformer": config["decoder"]["decoder_arbitrary_ende_attn"]
        }

        def to_builtin(x):
            if isinstance(x, np.generic):          # numpy int/float/bool
                return x.item()
            if isinstance(x, list):
                return [to_builtin(v) for v in x]
            if isinstance(x, dict):
                return {k: to_builtin(v) for k, v in x.items()}
            return x

        clean_flat = to_builtin(flattened)

        class FlowDumper(yaml.SafeDumper):
            pass

        FlowDumper.add_representer(
            list,
            lambda self, value:
                self.represent_sequence('tag:yaml.org,2002:seq',
                                        value, flow_style=True)
        )

        
        with open(self.SAMPLE_CONFIG_PATH, "w") as fout:
            yaml.dump(clean_flat, fout,Dumper=FlowDumper, sort_keys=False, width=1000)

        # Constant reference file 
        REF_FILE = "/home/graham/Documents/hardware-aware-transformers/output/reference.txt"

        gen_output_path = self.call_generate_script(self.MODEL_DIR, self.CHECKPOINT_FILE, self.DATA_BIN, self.SAMPLE_CONFIG_PATH)
        
        # extract only the hypothesis for scoring
        sys_output_file = os.path.join("output", "sys_output.txt")
        self.extract_hypothesis(gen_output_path, sys_output_file)
        
        # run score.py with hypothesis file (-s) and constant reference file (-r)
        score_script = "/home/graham/Documents/hardware-aware-transformers/score.py"
        score_command = f"python3 {score_script} -s {sys_output_file} -r {REF_FILE}"
        #print("Running score.py with command:")
        #print(score_command)
        score_result = subprocess.run(score_command, shell=True, capture_output=True, text=True)
        
        print("Score script output:")
        print(score_result.stdout)
        if score_result.returncode != 0:
            print("Error during scoring:")
            print(score_result.stderr)

        # parse bleu score from output
        # Expected output contains something like: "BLEU4 = 30.80, 63.0/37.9/24.5/16.1 ..."
        match = re.search(r"BLEU4\s*=\s*([\d\.]+)", score_result.stdout)
        if match:
            bleu_score = float(match.group(1))
        else:
            bleu_score = 0.0
        print("Final BLEU Score (accuracy):", bleu_score)
        
        return bleu_score
    
    def placeholder_evaluate_accuracy(self, genome):
        """
        Placeholder accuracy evaluation that returns a random float between 25 and 40.
        """
        return random.uniform(25, 40)
    
    def call_generate_script(self, model_dir, checkpoint_file, data_bin, sample_config_path):
        """Call generate.py using subprocess and return the path to the output file."""
        generate_script = "/home/graham/Documents/hardware-aware-transformers/generate.py"
        beam_size = 1
        batch_size = 1
        gen_subset = "valid"
        gpu = 0  # Use GPU 0
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "generate_output.txt")
        
        # Construct the command
        command = [
            "CUDA_VISIBLE_DEVICES={}".format(gpu),
            "python3", generate_script,
            "--data", data_bin,
            "--path", os.path.join(model_dir, checkpoint_file),
            "--gen-subset", gen_subset,
            "--beam", str(beam_size),
            "--batch-size", str(batch_size),
            "--remove-bpe",
            "--configs", sample_config_path
        ]
        print("running generate command")
        #print("Running generate.py with command:")
        #print(" ".join(command))

        result = subprocess.run(" ".join(command), shell=True, capture_output=True, text=True)

        # Save the generation output to file
        with open(output_file, "w") as f:
            f.write(result.stdout)

        print(f"Generation complete. Results saved to {output_file}")
        if result.returncode != 0:
            print("Error during generation:")
            print(result.stderr)
        return output_file

    def extract_hypothesis(self, gen_output_file, sys_output_file):
        """
        Extract only the hypothesis lines (lines starting with "H-") from the generate output.
        For each such line, split by tabs and take the third field (if available) which contains
        the actual generated translation.
        """
        with open(gen_output_file, "r") as fin, open(sys_output_file, "w") as fout:
            for line in fin:
                if line.startswith("H-"):
                    parts = line.strip().split("\t")
                    # Use the third field if available; otherwise, fallback to the second.
                    if len(parts) >= 3:
                        fout.write(parts[2] + "\n")
                    elif len(parts) >= 2:
                        fout.write(parts[1] + "\n")
        print(f"Hypothesis extracted to {sys_output_file}")


    def evaluate_latency(self, genome) -> float:
        print("evaluating latency")
        # ---------- build the sub‑transformer dict -----------------
        encoder_layer_num = genome[0]
        decoder_layer_num = genome[1]

        embed_dim_mapping_inv     = {0: 512, 1: 640}
        ffn_embed_dim_mapping_inv = {0: 512, 1: 1024, 2: 2048}
        sa_mapping_inv            = {0: 2,   1: 4}
        arbitrary_mapping_inv     = {0: -1,  1: 1,  2: 2}

        encoder_embed_dim = embed_dim_mapping_inv[genome[2]]
        decoder_embed_dim = embed_dim_mapping_inv[genome[3]]

        enc_ffn_genes   = genome[4:10]
        dec_ffn_genes   = genome[10:16]
        enc_sa_genes    = genome[16:22]
        dec_sa_genes    = genome[22:28]
        dec_ende_genes  = genome[28:34]
        dec_arbi_genes  = genome[34:40]

        enc_ffn_values = [ffn_embed_dim_mapping_inv[g] for g in enc_ffn_genes[:encoder_layer_num]]
        dec_ffn_values = [ffn_embed_dim_mapping_inv[g] for g in dec_ffn_genes[:decoder_layer_num]]
        enc_sa_values  = [sa_mapping_inv[g]            for g in enc_sa_genes[:encoder_layer_num]]
        dec_sa_values  = [sa_mapping_inv[g]            for g in dec_sa_genes[:decoder_layer_num]]
        dec_ende_vals  = [sa_mapping_inv[g]            for g in dec_ende_genes[:decoder_layer_num]]
        dec_arbi_vals  = [arbitrary_mapping_inv[g]     for g in dec_arbi_genes[:decoder_layer_num]]

        config = {
            "encoder": {
                "encoder_embed_dim"        : encoder_embed_dim,
                "encoder_layer_num"        : encoder_layer_num,
                "encoder_ffn_embed_dim"    : enc_ffn_values,
                "encoder_self_attention_heads": enc_sa_values,
            },
            "decoder": {
                "decoder_embed_dim"        : decoder_embed_dim,
                "decoder_layer_num"        : decoder_layer_num,
                "decoder_ffn_embed_dim"    : dec_ffn_values,
                "decoder_self_attention_heads"   : dec_sa_values,
                "decoder_ende_attention_heads"   : dec_ende_vals,
                "decoder_arbitrary_ende_attn"    : dec_arbi_vals,
            },
        }

        # ---------- latency prediction -----------------------------
        latency_ms = self.latency_predictor.predict_lat(config)
        print(f"latency: {latency_ms} ms")
        return latency_ms
    
    def _evaluate(self, x, out, *args, **kwargs):
        print("Starting evaluation:", x)
        accuracy = self.new_evaluate_accuracy(x)
        latency = self.evaluate_latency(x)
        print("Completed evaluation:", x)
        out["F"] = np.array([latency, -accuracy])

    def generate_random_genome(self, fixed_length=40):
        """
        Build a random valid genome of fixed length 40 with the following order:

        1. Encoder layer num (1)
        2. Decoder layer num (1)
        3. Encoder embed dim (1)
        4. Decoder embed dim (1)
        5. Encoder FFN embed dims (6) [per encoder layer, padded to 6]
        6. Decoder FFN embed dims (6) [per decoder layer, padded to 6]
        7. Encoder self-attention heads (6) [per encoder layer, padded to 6]
        8. Decoder self-attention heads (6) [per decoder layer, padded to 6]
        9. Decoder encoder-decoder heads (6) [per decoder layer, padded to 6]
        10. Decoder arbitrary attention (6) [per decoder layer, padded to 6]

        Total genome length = 1+1+1+1+6+6+6+6+6+6 = 40.
        
        encoder layer is always six because this paper sucks 

        The number of decoder layers are chosen randomly from 1 to 6.
        """
        # Mapping dictionaries
        layer_num_mapping = {i: i for i in range(1, 7)}
        embed_dim_mapping = {512: 0, 640: 1}
        ffn_embed_dim_mapping = {512: 0, 1024: 1, 2048: 2}
        SA_num_heads_mapping = {2: 0, 4: 1}
        arbitrary_ende_atten_mapping = {-1: 0, 1: 1, 2: 2}

        genome = []

        # 1. Encoder layer num: is always 6, man
        encoder_layers = 6
        genome.append(layer_num_mapping[encoder_layers])

        # 2. Decoder layer num: choose randomly between 1 and 6.
        decoder_layers = random.randint(1, 6)
        genome.append(layer_num_mapping[decoder_layers])

        # 3. Encoder embed dim
        encoder_embed_dim = random.choice([512, 640])
        genome.append(embed_dim_mapping[encoder_embed_dim])

        # 4. Decoder embed dim
        decoder_embed_dim = random.choice([512, 640])
        genome.append(embed_dim_mapping[decoder_embed_dim])

        # Helper function to generate per-layer sections and pad them to 6 entries.
        def generate_and_pad(n, choices, mapping):
            values = [mapping[random.choice(choices)] for _ in range(n)]
            # Pad with -1 if fewer than 6 values.
            values.extend([-1] * (6 - len(values)))
            return values

        # 5. Encoder FFN embed dims: one per encoder layer, then pad to 6.
        encoder_ffn = generate_and_pad(encoder_layers, [512, 1024, 2048], ffn_embed_dim_mapping)
        genome.extend(encoder_ffn)

        # 6. Decoder FFN embed dims: one per decoder layer, then pad to 6.
        decoder_ffn = generate_and_pad(decoder_layers, [512, 1024, 2048], ffn_embed_dim_mapping)
        genome.extend(decoder_ffn)

        # 7. Encoder self-attention heads: one per encoder layer, then pad to 6.
        encoder_sa = generate_and_pad(encoder_layers, [2, 4], SA_num_heads_mapping)
        genome.extend(encoder_sa)

        # 8. Decoder self-attention heads: one per decoder layer, then pad to 6.
        decoder_sa = generate_and_pad(decoder_layers, [2, 4], SA_num_heads_mapping)
        genome.extend(decoder_sa)

        # 9. Decoder encoder-decoder heads: one per decoder layer, then pad to 6.
        decoder_ende = generate_and_pad(decoder_layers, [2, 4], SA_num_heads_mapping)
        genome.extend(decoder_ende)

        # 10. Decoder arbitrary attention: one per decoder layer, then pad to 6.
        decoder_arbitrary = generate_and_pad(decoder_layers, [-1, 1, 2], arbitrary_ende_atten_mapping)
        genome.extend(decoder_arbitrary)

        return genome

    class custom_sampling(BaseArchitecture.Sampling):

        def _do(self, problem, n_samples, **kwargs):
            samples = []
            for _ in range(n_samples):
                genome = problem.generate_random_genome(problem.fixed_length)
                samples.append(genome)
            return np.array(samples)
        
    
    class custom_crossover(BaseArchitecture.Crossover):
        """
        Group-level crossover operator that, for each mating pair, selects each "group" of
        layer dimensions entirely from one parent or the other. Global genes (indices 0-3)
        are crossed over gene-by-gene.
        
        Assumes the problem's genome has 40 genes ordered as:
            0-3: Global parameters
            4-9: Encoder FFN embed dims
            10-15: Decoder FFN embed dims
            16-21: Encoder self-attention heads
            22-27: Decoder self-attention heads
            28-33: Decoder encoder-decoder heads
            34-39: Decoder arbitrary attention
        """

        def __init__(self, n_parents=2, n_offsprings=2, prob=0.9, **kwargs):
            super().__init__(n_parents, n_offsprings, prob, **kwargs)
        
        def _do(self, problem, X, **kwargs):
            """
            Apply group-level crossover to each mating pair.
            
            Parameters
            ----------
            problem : object
                The problem instance (not used in this operator).
            X : numpy.ndarray
                The parent chromosomes with shape (n_parents, n_matings, n_variables).
            
            Returns
            -------
            offspring : numpy.ndarray
                The offspring chromosomes (same shape as X).
            """
            # X is expected to have shape (2, n_matings, 40).
            n_parents, n_matings, n_var = X.shape
            
            # Pre-allocate the offspring array.
            offspring = np.empty_like(X)
            
            # Indices for global genes:
            global_indices = list(range(0, 4))
            # Define the six groups for per-layer parameters (each group has 6 consecutive genes):
            group_ranges = [(4, 10),    # Encoder FFN embed dims
                            (10, 16),   # Decoder FFN embed dims
                            (16, 22),   # Encoder self-attention heads
                            (22, 28),   # Decoder self-attention heads
                            (28, 34),   # Decoder encoder-decoder heads
                            (34, 40)]   # Decoder arbitrary attention

            # Iterate over all mating pairs.
            for j in range(n_matings):
                # Extract the two parents for mating pair j.
                parent_A = X[0, j, :]
                parent_B = X[1, j, :]
                
                # Produce two offspring for this mating pair.
                for i in range(n_parents):  # i=0 and i=1 for two offspring.
                    child = np.empty(n_var, dtype=X.dtype)
                    
                    # Global genes: perform gene-wise coin toss for each gene.
                    for idx in global_indices:
                        if np.random.rand() < 0.5:
                            child[idx] = parent_A[idx]
                        else:
                            child[idx] = parent_B[idx]
                    
                    # For each group of layer parameters, choose the entire block from one parent.
                    for (start, end) in group_ranges:
                        if np.random.rand() < 0.5:
                            child[start:end] = parent_A[start:end]
                        else:
                            child[start:end] = parent_B[start:end]
                    
                    offspring[i, j, :] = child
                    
            return offspring
        

    class CustomMutation(BaseArchitecture.Mutation):

        def __init__(self, mutation_rate=0.1, **kwargs):
            super().__init__()
            self.mutation_rate = mutation_rate

        def _do(self, problem, X, **kwargs):
            """
            Mutate each gene with a probability self.mutation_rate.
            For each gene that is to be mutated, a new random value is selected uniformly
            from the range defined by the lower and upper bounds (inclusive) for that gene.
            
            Parameters
            ----------
            problem : object
                The problem instance. Assumes problem.xl and problem.xu are numpy arrays 
                defining the lower and upper bounds for each variable.
            X : numpy.ndarray
                The current population matrix with shape (n_individuals, n_variables).
            
            Returns
            -------
            X_mut : numpy.ndarray
                The mutated population matrix.
            """
            X_mut = X.copy()
            n_ind, n_var = X.shape
            # Ensure that problem.xl and problem.xu are arrays of length n_var.
            lb = problem.xl
            ub = problem.xu
            
            for i in range(n_ind):
                for j in range(n_var):
                    if np.random.rand() < self.mutation_rate:
                        # Randomly choose a new valid gene in the interval [lb[j], ub[j]] (inclusive).
                        X_mut[i, j] = np.random.randint(lb[j], ub[j] + 1)
                        
            return X_mut

    class custom_duplicate_elimination(BaseArchitecture.ElementwiseDuplicateElimination):
        def is_equal(self, a, b):
            return np.array_equal(a.X, b.X)

class TimedSequenceGenerator:
    def __init__(self, task, models, **gen_kwargs):
        self.inner  = task.build_generator(gen_kwargs)
        self.task   = task
        self.models = models

    def generate(self, models, sample):
        """returns (hypos, enc_ms, dec_ms)"""
        start = time.time()
        hypos = self.task.inference_step(self.inner, self.models, sample)
        dec_ms = (time.time() - start) * 1000

        # crude encoder timing = forward of src-tokens only
        enc_start = time.time()
        with torch.no_grad():
            models[0].encoder(sample['net_input']['src_tokens'])
        enc_ms = (time.time() - enc_start) * 1000

        return hypos, enc_ms, dec_ms