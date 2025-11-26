# /home/graham/Documents/surrogate_benchmark_project/evo_search/architectures/hat_arch.py
import random
import numpy as np
from .base_architecture import BaseArchitecture
import torch
import json
from fairseq.models.transformer import TransformerModel
from fairseq.models.transformer_super import TransformerSuperModel
from fairseq.modules import LinearSuper
from fairseq.modules.multihead_attention_super import MultiheadAttentionSuper

import pandas as pd

import os, pathlib, sys
from pathlib import Path
from latency_predictor import LatencyPredictor
from fairseq import tasks, utils
from fairseq import checkpoint_utils, sequence_generator
import math


HAT_REPO = pathlib.Path(__file__).resolve().parents[3] / "surrogate_benchmark_project" / "hardware-aware-transformers"
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
        self._load_lut("/home/graham/Documents/cache/hat_lut.csv")
        self.CHECKPOINT_FILE = HAT_REPO / "HAT_iwslt14deen_super_space1.pt"
        self.DATA_BIN = HAT_REPO / "data/binary/iwslt14_de_en"
        self.REF_FILE = HAT_REPO / "output/reference.txt"
        ensemble, self.args, self.task = checkpoint_utils.load_model_ensemble_and_task([self.CHECKPOINT_FILE], arg_overrides={"data": str(self.DATA_BIN)})
        self.model = ensemble[0]

        #self._enable_magnitude_selection(self.model, metric='l2')
        self.model.eval()

        args_ns = self.task.args                       # Namespace saved in checkpoint
        args_ns.dataset_impl = getattr(args_ns, "dataset_impl", "mmap")
        args_ns.combine       = getattr(args_ns, "combine", False)
        self.task.load_dataset("valid")

        # setattr(self.args, "sentence_avg", getattr(self.args, "sentence_avg", False))
        # setattr(self.args, "criterion", getattr(self.args, "criterion", "cross_entropy"))

        args_ns.sentence_avg     = False                       # token average
        args_ns.label_smoothing  = 0.1                         # paper setting
        args_ns.criterion        = "label_smoothed_cross_entropy"
        self.criterion = self.task.build_criterion(args_ns) 
        self.pad = self.task.target_dictionary.pad()

        self.metric = "l2"         # or "l1" – matches _enable_magnitude_selection
        self.eval_batches = 10 

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
    
    def _enable_magnitude_selection(self, model, metric):
        for m in model.modules():
            if isinstance(m, LinearSuper):
                m.selection_mode = 'topk'
                m.metric = metric

            if isinstance(m, MultiheadAttentionSuper):
                m.selection_mode = "magnitude"
                m.metric = metric
                
        model.eval()
        torch.set_grad_enabled(False) 

    def _disable_magnitude_selection(self, model):
        for m in model.modules():
            if isinstance(m, LinearSuper):
                m.selection_mode = 'prefix'
                m.metric = 'l2'
                # clear any overrides if your LinearSuper exposes them
                if hasattr(m, "set_col_idx_override"): m.set_col_idx_override(None)
                if hasattr(m, "set_row_idx_override"): m.set_row_idx_override(None)

            if isinstance(m, MultiheadAttentionSuper):
                m.selection_mode = 'prefix'
                m.metric = 'l2'
                # clear head selection & out_proj override
                m._keep_heads = None
                if hasattr(m.out_proj, "set_col_idx_override"):
                    m.out_proj.set_col_idx_override(None)

        model.eval()
        torch.set_grad_enabled(False)

    def enable_weight_magnitude(self, metric='l2'):
        self._enable_magnitude_selection(self.model, metric=metric)

    def disable_weight_magnitude(self):
        self._disable_magnitude_selection(self.model)


    def _set_selection_modes(self, lin_mode: str, mha_mode: str, metric: str = "l2"):
        """
        lin_mode in {"prefix", "topk"}
        mha_mode in {"prefix", "magnitude"}  (your current custom behavior)
        """
        for m in self.model.modules():
            if isinstance(m, LinearSuper):
                m.selection_mode = lin_mode
                m.metric = metric
                # clear any explicit column/row overrides
                if hasattr(m, "set_col_idx_override"):
                    m.set_col_idx_override(None)
                if hasattr(m, "set_row_idx_override"):
                    m.set_row_idx_override(None)

            if isinstance(m, MultiheadAttentionSuper):
                m.selection_mode = mha_mode
                m.metric = metric
                if hasattr(m, "_keep_heads"):
                    m._keep_heads = None
                if hasattr(m, "out_proj") and hasattr(m.out_proj, "set_col_idx_override"):
                    m.out_proj.set_col_idx_override(None)
        self.model.eval()
        torch.set_grad_enabled(False)
     
    def evaluate_latency(self, genome) -> float:
       
        sub_cfg = self._genome_to_subtransformer_cfg(genome)
        latency_ms = self.latency_predictor.predict_lat(sub_cfg)
        print(f"latency: {latency_ms} ms")
        return latency_ms

    def validate_loss_new(self, genome, max_batches=None):
        # meant to match HAT's method
        print("validating loss")
        sub_cfg = self._genome_to_subtransformer_cfg(genome)
        self.model.set_sample_config(sub_cfg)
        self.model.eval()
        self.criterion.eval()

        args = self.task.args
        itr = self.task.get_batch_iterator(
            dataset=self.task.dataset("valid"),
            max_tokens=getattr(args, "max_tokens_valid", None),
            max_sentences=getattr(args, "max_sentences_valid", None),
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=getattr(args, "skip_invalid_size_inputs_valid_test", True),
            required_batch_size_multiple=getattr(args, "required_batch_size_multiple", 1),
            seed=getattr(args, "seed", 1),
            num_shards=getattr(args, "distributed_world_size", 1),
            shard_id=getattr(args, "distributed_rank", 0),
            num_workers=getattr(args, "num_workers", 0),
        ).next_epoch_itr(shuffle=False)

        sentence_avg = getattr(args, "sentence_avg", False)
        use_cuda = torch.cuda.is_available()

        total_weighted_loss = 0.0  # sum over batches of (loss_value * weight)
        total_weight = 0.0         # sum of weights (sample_size or nsentences)
        seen = 0
        cap = max_batches if max_batches is not None else getattr(self, "eval_batches", None)

        with torch.no_grad():
            for sample in itr:
                if cap is not None and seen >= cap:
                    break
                seen += 1

                if use_cuda:
                    sample = utils.move_to_cuda(sample)

                net_output = self.model(**sample["net_input"])
                loss, sample_size, logging_output = self.criterion(self.model, sample, net_output)

                weight = logging_output.get("nsentences", sample_size) if sentence_avg else sample_size
                loss_value = float(logging_output.get("loss", 0.0))

                total_weighted_loss += loss_value * float(weight)
                total_weight += float(weight)

        if total_weight == 0:
            return float("inf")
        return total_weighted_loss / total_weight


    def evaluate_validation_loss(self, genome, max_batches: int = 10) -> float:
        """
        compute avg nll loss / token
        """
        
        sub_cfg = self._genome_to_subtransformer_cfg(genome)
        self.model.set_sample_config(sub_cfg)
        self.model.eval()

        # build a fresh iterator
        valid_set = self.task.dataset("valid")
        iterator = self.task.get_batch_iterator(
            dataset=valid_set,
            max_tokens   = self.args.max_tokens,
            max_sentences= getattr(self.args, "batch_size", None),
            max_positions= utils.resolve_max_positions(
                            self.task.max_positions(),
                            *[m.max_positions() for m in [self.model]]),
            ignore_invalid_inputs=True,
        ).next_epoch_itr(shuffle=False)

        nll_sum = 0.0
        token_sum = 0.0
        cuda = torch.cuda.is_available()
        batch_nll = []
        num_batch = 0
        losses = 0
        with torch.no_grad():
            for idx, batch in enumerate(iterator):
                if idx >= max_batches:
                    break
                batch = utils.move_to_cuda(batch) if cuda else batch
                num_batch += 1

                net_output = self.model(**batch["net_input"])
                loss, sample_size, logging_output = self.criterion(self.model, batch, net_output)
                nll_per_tok = logging_output["nll_loss"] / sample_size
                batch_nll.append(nll_per_tok)
                losses += loss
             
        mean_nll = float(np.mean(batch_nll)) 
        loss = losses / num_batch    
        print(f"mean_nll: {mean_nll:.4f}")  
        return mean_nll

    def evaluate_bleu_score(self, genome):
        import os, re, subprocess, yaml, numpy as np

        # build flat config 
        sub_cfg = self._genome_to_subtransformer_cfg(genome)
        flattened = {
            "encoder-embed-dim-subtransformer": sub_cfg["encoder"]["encoder_embed_dim"],
            "decoder-embed-dim-subtransformer": sub_cfg["decoder"]["decoder_embed_dim"],
            "encoder-ffn-embed-dim-all-subtransformer": sub_cfg["encoder"]["encoder_ffn_embed_dim"],
            "decoder-ffn-embed-dim-all-subtransformer": sub_cfg["decoder"]["decoder_ffn_embed_dim"],
            "encoder-layer-num-subtransformer": sub_cfg["encoder"]["encoder_layer_num"],
            "decoder-layer-num-subtransformer": sub_cfg["decoder"]["decoder_layer_num"],
            "encoder-self-attention-heads-all-subtransformer": sub_cfg["encoder"]["encoder_self_attention_heads"],
            "decoder-self-attention-heads-all-subtransformer": sub_cfg["decoder"]["decoder_self_attention_heads"],
            "decoder-ende-attention-heads-all-subtransformer": sub_cfg["decoder"]["decoder_ende_attention_heads"],
            "decoder-arbitrary-ende-attn-all-subtransformer": sub_cfg["decoder"]["decoder_arbitrary_ende_attn"],
        }

        class FlowDumper(yaml.SafeDumper): pass
        FlowDumper.add_representer(list, lambda self, v: self.represent_sequence('tag:yaml.org,2002:seq', v, flow_style=True))

        sample_cfg_path = str(self.HAT_REPO / "output" / "intermediate_config.yaml") if hasattr(self, "HAT_REPO") else "/tmp/intermediate_config.yaml"
        os.makedirs(os.path.dirname(sample_cfg_path), exist_ok=True)
        with open(sample_cfg_path, "w") as f:
            yaml.dump(flattened, f, Dumper=FlowDumper, sort_keys=False, width=1000)

        # run generate.py 
        generate_py = os.path.join(os.path.dirname(str(self.CHECKPOINT_FILE)), "generate.py")
        out_dir = "output"; os.makedirs(out_dir, exist_ok=True)
        gen_out = os.path.join(out_dir, "generate_output.txt")
        cmd = [
            f"CUDA_VISIBLE_DEVICES=0",
            "python3", generate_py,
            "--data", str(self.DATA_BIN),
            "--path", str(self.CHECKPOINT_FILE),
            "--gen-subset", "test",
            "--beam", "5",
            "--batch-size", "1",
            "--remove-bpe",
            "--configs", sample_cfg_path,
        ]
        res = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        with open(gen_out, "w") as f: f.write(res.stdout)

        # extract H- lines to sys_output.txt 
        sys_out_file = os.path.join(out_dir, "sys_output.txt")
        with open(gen_out) as fin, open(sys_out_file, "w") as fout:
            for line in fin:
                if line.startswith("H-"):
                    parts = line.rstrip("\n").split("\t")
                    fout.write((parts[2] if len(parts) >= 3 else parts[-1]) + "\n")

        # score with score.py
        score_py = os.path.join(os.path.dirname(str(self.CHECKPOINT_FILE)), "score.py")
        score_cmd = f"python3 {score_py} -s {sys_out_file} -r {self.REF_FILE}"
        score = subprocess.run(score_cmd, shell=True, capture_output=True, text=True)

        m = re.search(r"BLEU4\s*=\s*([\d\.]+)", score.stdout)
        bleu = float(m.group(1)) if m else 0.0
        print(f"BLEU: {bleu:.2f}")
        return bleu
    
    def _evaluate(self, x, out, *args, **kwargs):
        print("Starting evaluation:", x)
        key = self._genome_key(x)
        cached = key in self.lut.index

        if cached:
            row = self.lut.loc[key]
            loss      = float(row.loss)
            latency  = float(row.latency)
            print(f"CACHE:  {key[:40]}…  loss={loss:.4f}  lat={latency:.2f}")

        else:
            
            loss = self.evaluate_validation_loss(x, max_batches=self.eval_batches)
            latency = self.evaluate_latency(x)

            self.lut.loc[key, ["loss", "latency"]] = [loss, latency]

            self._save_lut()



        print("Completed evaluation!")
        out["F"] = np.array([latency, loss])

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
    
    @staticmethod
    def _genome_to_subtransformer_cfg(gene: np.ndarray) -> dict:
        """Decode a 40-gene chromosome into a HAT sub-transformer config."""
      
        map_embed = {0: 512, 1: 640}
        map_ffn   = {0: 512, 1: 1024, 2: 2048}
        map_head  = {0: 2,   1: 4}
        map_arbi  = {0: -1,  1: 1,   2: 2}

        enc_layers, dec_layers = int(gene[0]), int(gene[1])
        enc_emb = map_embed[int(gene[2])]
        dec_emb = map_embed[int(gene[3])]

        enc_ffn_raw  = gene[4:10]
        dec_ffn_raw  = gene[10:16]
        enc_sa_raw   = gene[16:22]
        dec_sa_raw   = gene[22:28]
        dec_ende_raw = gene[28:34]
        dec_arbi_raw = gene[34:40]

       
        enc_ffn  = [_map(v, map_ffn)   for v in enc_ffn_raw[:enc_layers]]
        dec_ffn  = [_map(v, map_ffn)   for v in dec_ffn_raw[:dec_layers]]
        enc_sa   = [_map(v, map_head)  for v in enc_sa_raw[:enc_layers]]
        dec_sa   = [_map(v, map_head)  for v in dec_sa_raw[:dec_layers]]
        dec_ende = [_map(v, map_head)  for v in dec_ende_raw[:dec_layers]]
        dec_arbi = [_map(v, map_arbi)  for v in dec_arbi_raw[:dec_layers]]

       
        return {
            "encoder": {
                "encoder_embed_dim":            enc_emb,
                "encoder_layer_num":            enc_layers,
                "encoder_ffn_embed_dim":        enc_ffn,
                "encoder_self_attention_heads": enc_sa,
            },
            "decoder": {
                "decoder_embed_dim":            dec_emb,
                "decoder_layer_num":            dec_layers,
                "decoder_ffn_embed_dim":        dec_ffn,
                "decoder_self_attention_heads": dec_sa,
                "decoder_ende_attention_heads": dec_ende,
                "decoder_arbitrary_ende_attn":  dec_arbi,
            },
        }



    def _genome_key(self, genome: np.ndarray) -> str:
        base = json.dumps(genrome := genome.tolist())

        # infer current modes (default to 'prefix' if fields are missing)
        lin_mode = "prefix"
        mha_mode = "prefix"
        for m in self.model.modules():
            if isinstance(m, LinearSuper):
                lin_mode = getattr(m, "selection_mode", "prefix")
                break
        for m in self.model.modules():
            if isinstance(m, MultiheadAttentionSuper):
                mha_mode = getattr(m, "selection_mode", "prefix")
                break

        metric  = getattr(self, "metric", "l2")
        batches = getattr(self, "eval_batches", 10)
        ckpt    = Path(getattr(self, "CHECKPOINT_FILE", "ckpt.pt")).name

        # single-line suffix appended to your existing key format
        return f"{base}|lin={lin_mode}|mha={mha_mode}|met={metric}|ckpt={ckpt}|b={batches}"
    
    def _load_lut(self, lut_path: str):

        lut_file = Path(lut_path)
        if lut_file.exists():
            self.lut = pd.read_csv(lut_file, index_col="genome_key")  
        else:
            self.lut = pd.DataFrame(columns=["loss", "latency"])

    def _save_lut(self):
        lut_path = Path("/home/graham/Documents/cache/hat_lut_newloss.csv")
        self.lut.to_csv(lut_path, index_label="genome_key") 

   
    class custom_sampling(BaseArchitecture.Sampling):

        def _do(self, problem, n_samples, **kwargs):
            samples = []
            for _ in range(n_samples):
                genome = problem.generate_random_genome(problem.fixed_length)
                samples.append(genome)
            return np.array(samples)
        
    
    class custom_crossover(BaseArchitecture.Crossover):

        def __init__(self, n_parents=2, n_offsprings=2, prob=0.9, **kwargs):
            super().__init__(n_parents, n_offsprings, prob, **kwargs)
        
        def _do(self, problem, X, **kwargs):

            n_parents, n_matings, n_var = X.shape # X is expected to have shape (2, n_matings, 40).
            offspring = np.empty_like(X)
            
            global_indices = list(range(0, 4))

            # need to iterate through the decoder layers dynamicaly based on gene 2 (layer num)
            encoder_group_ranges = [(4, 10),    # encoder FFN embed dims
                                    (16, 22)]   # encoder self-attention heads
                              
            decoder_group_ranges =  [(10, 16),  # decoder FFN embed dims
                                    (22, 28),   # decoder self-attention heads 
                                    (28, 34),   # decoder encoder-decoder heads
                                    (34, 40)]   # decoder arbitrary attention        
                                     

            for j in range(n_matings):
                parent_A = X[0, j, :]
                parent_B = X[1, j, :]
                
                for i in range(n_parents):  # i=0 and i=1 for two children
                    child = np.empty(n_var, dtype=X.dtype)
                    
                    
                    for idx in global_indices: 
                        if np.random.rand() < 0.5:
                            child[idx] = parent_A[idx]
                        else:
                            child[idx] = parent_B[idx]
                    
                    for (start, end) in encoder_group_ranges: 
                        if np.random.rand() < 0.5:
                            child[start:end] = parent_A[start:end]
                        else:
                            child[start:end] = parent_B[start:end]

                    dec_layers = int(child[1])          # gene-1 (decoder-layer-num)
                    for (start, end) in decoder_group_ranges:      
                        for k in range(6):
                            pos = start + k
                            if k < dec_layers:           # the real layers
                                if np.random.rand() < 0.5:
                                    child[pos] = parent_A[pos]
                                else:
                                    child[pos] = parent_B[pos]
                            else:                       
                                child[pos] = -1
                    
                    offspring[i, j, :] = child
                    
            return offspring
        

    class CustomMutation(BaseArchitecture.Mutation):

        def __init__(self, mutation_rate=0.1, max_retries=3, **kwargs):
            super().__init__()
            self.mutation_rate = mutation_rate
            self.max_retries = max_retries
        
        @staticmethod
        def _try_valid(problem, genome):
            """Return True iff genome can be decoded without exception."""
            try:
                _ = problem._genome_to_subtransformer_cfg(genome)
                return True
            except Exception:
                return False

        def _do(self, problem, X, **kwargs):
            X_mut = X.copy()
            n_ind, n_var = X.shape
           
            lb = problem.xl  
            ub = problem.xu
            
            for i in range(n_ind):
                parent = X[i].copy() 
                for _ in range(self.max_retries):

                    child = parent.copy()
                    for j in range(2, n_var):       # genes 0 & 1 are frozen
                        if np.random.rand() < self.mutation_rate:
                            child[j] = np.random.randint(lb[j], ub[j] + 1)

                    if self._try_valid(problem, child):
                        X_mut[i] = child     
                        break
                    else:
                        print("failed mutation")
                else:
                    print("hit max failed mutations. taking original parent gene")
                    X_mut[i] = parent
            return X_mut

    class custom_duplicate_elimination(BaseArchitecture.ElementwiseDuplicateElimination):
        def is_equal(self, a, b):
            return np.array_equal(a.X, b.X)


    class custom_crossover_importance(BaseArchitecture.Crossover):
        
        def __init__(self, n_parents=2, n_offsprings=2, prob=0.9, **kwargs):
            super().__init__(n_parents, n_offsprings, prob, **kwargs)

            # fixed genome layout (40 genes)
            self.E_FFN0, self.E_FFNN = 4, 10    
            self.D_FFN0, self.D_FFNN = 10, 16   
            self.E_SA0,  self.E_SAN  = 16, 22    
            self.D_SA0,  self.D_SAN  = 22, 28    
            self.D_EN0,  self.D_ENN  = 28, 34   
            self.D_AR0,  self.D_ARN  = 34, 40  

        # helpers

        @staticmethod
        def _sum_abs_params(layer) -> float:
            s = 0.0; n = 0
            for p in layer.parameters():
                s += p.detach().abs().sum().item()
                n += p.numel()
            score = s / max(1, n) 
            return score

        def _cache_parent_layer_mags(self, problem, parent):
            """return (enc_mags[6], dec_mags[dec_L], total_mag)"""
            cfg = problem._genome_to_subtransformer_cfg(parent)
            model = problem.model
            model.set_sample_config(cfg)

            enc_mags = [self._sum_abs_params(model.encoder.layers[i]) for i in range(6)]
            dec_L = int(parent[1])
            dec_mags = [self._sum_abs_params(model.decoder.layers[i]) for i in range(dec_L)]
            total = sum(enc_mags) + sum(dec_mags)
            return enc_mags, dec_mags, total

        def _make_child(self, parent_A, parent_B,
                    encA, decA, totA,
                    encB, decB, totB,
                    tie_break='A',
                    eps=0.1, temp=2.0):
            """
            Build one child. tie_break used only as a fallback.
            eps:  small exploration prob (epsilon-greedy)
            temp: soft selection temperature (>1 = flatter/probabilistic, <1 = sharper)
            """

            def _soft_pick(mA, mB, tie_break='A'):
                if np.random.rand() < eps: # greed attack!!
                    return np.random.rand() < 0.5   # coin flip

                # soft preference by layer-normalized magnitude 
                mA = 1e-12 if not np.isfinite(mA) or mA <= 0 else mA # check is not bullshit 
                mB = 1e-12 if not np.isfinite(mB) or mB <= 0 else mB
                wA = mA ** (1.0 / temp)
                wB = mB ** (1.0 / temp)
                denom = wA + wB
                if not np.isfinite(denom) or denom <= 0:
                    return (tie_break == 'A')
                pA = wA / denom
                return np.random.rand() < pA

            child = np.empty_like(parent_A)  # returns bool where a is true and b is false

            child[0] = 6  # encoder depth is always 6 in hat

            decLA, decLB = int(parent_A[1]), int(parent_B[1]) # get decoder layer numbers
            decA_sum = float(np.nansum(decA[:decLA])) if decLA > 0 else 0.0
            decB_sum = float(np.nansum(decB[:decLB])) if decLB > 0 else 0.0
            useA_depth = _soft_pick(decA_sum, decB_sum, tie_break=tie_break)
            child_dec_layers = decLA if useA_depth else decLB
            child_dec_layers = max(1, min(6, child_dec_layers))   # safety clamp
            child[1] = child_dec_layers

            #  pick embed dims from more important parent
            useA_tot = _soft_pick(totA, totB, tie_break=tie_break)
            winner = parent_A if useA_tot else parent_B
            child[2] = winner[2]   
            child[3] = winner[3]  

            # encoder per-layer genes 
            for l in range(6):
                mA, mB = encA[l], encB[l]
                useA = _soft_pick(mA, mB, tie_break=tie_break)
                donor = parent_A if useA else parent_B
                child[self.E_FFN0 + l] = donor[self.E_FFN0 + l]
                child[self.E_SA0  + l] = donor[self.E_SA0  + l]

            # decoder: only layers that exist in parents
            kept = 0
            

            for l in range(6):
                if l >= child_dec_layers:
                    child[self.D_FFN0 + l] = -1
                    child[self.D_SA0  + l] = -1
                    child[self.D_EN0  + l] = -1
                    child[self.D_AR0  + l] = -1
                    continue

                hasA = l < decLA
                hasB = l < decLB

                if hasA and hasB:
                    mA = decA[l]
                    mB = decB[l]
                    useA = _soft_pick(mA, mB, tie_break=tie_break)
                    donor = parent_A if useA else parent_B
                elif hasA:
                    donor = parent_A
                elif hasB:
                    donor = parent_B
                else:
                    print("decoder broken!!")
                    donor = None

                if donor is None:
                    child[self.D_FFN0 + l] = -1
                    child[self.D_SA0  + l] = -1
                    child[self.D_EN0  + l] = -1
                    child[self.D_AR0  + l] = -1
                else:
                    child[self.D_FFN0 + l] = donor[self.D_FFN0 + l]
                    child[self.D_SA0  + l] = donor[self.D_SA0  + l]
                    child[self.D_EN0  + l] = donor[self.D_EN0  + l]
                    child[self.D_AR0  + l] = donor[self.D_AR0  + l]
                    kept += 1
            

            return child

            # main API

        def _do(self, problem, X, **kwargs):
            """
            X: shape (2, n_matings, 40)
            returns offspring: shape (2, n_matings, 40)
            """
            import numpy as np

            n_parents, n_matings, n_var = X.shape
            assert n_parents == 2, "Expect 2 parents."
            offspring = np.empty_like(X)

            for j in range(n_matings):
                A = X[0, j].copy()
                B = X[1, j].copy()

                encA, decA, totA = self._cache_parent_layer_mags(problem, A)
                encB, decB, totB = self._cache_parent_layer_mags(problem, B)

                offspring[0, j, :] = self._make_child(A, B, encA, decA, totA, encB, decB, totB, tie_break='A')
                offspring[1, j, :] = self._make_child(A, B, encA, decA, totA, encB, decB, totB, tie_break='B')

            return offspring
        
    class custom_mutation_importance(BaseArchitecture.Mutation):
        def __init__(self, mutation_rate=0.1, max_retries=3, **kwargs):
            super().__init__(**kwargs)
            self.mutation_rate = mutation_rate
            self.max_retries = max_retries
            self.ENC_FFN = list(range(4, 10))
            self.DEC_FFN = list(range(10, 16))
            self.ENC_SA  = list(range(16, 22))
            self.DEC_SA  = list(range(22, 28))
            self.DEC_EN  = list(range(28, 34))
            self.DEC_AR  = list(range(34, 40))
            self.C_EMB = [0, 1]
            self.C_FFN = [0, 1, 2]
            self.C_SA  = [0, 1]
            self.C_EN  = [0, 1]
            self.C_AR  = [0, 1, 2]

        @staticmethod
        def _try_valid(problem, g):
            try:
                _ = problem._genome_to_subtransformer_cfg(g)
                return True
            except Exception:
                return False

        def _layer_mags(self, problem, g):
            cfg = problem._genome_to_subtransformer_cfg(g)
            m = problem.model
            m.set_sample_config(cfg)
            enc = np.array([sum(float(p.detach().abs().sum().item()) for p in m.encoder.layers[i].parameters())
                            for i in range(6)], dtype=np.float64)
            dl = int(g[1])
            dec = np.full(6, np.nan, dtype=np.float64)
            for i in range(dl):
                dec[i] = sum(float(p.detach().abs().sum().item()) for p in m.decoder.layers[i].parameters())
            return enc, dec, dl

        @staticmethod
        def _inv_scale(x):
            x = x.copy()
            fin = np.isfinite(x)
            if not fin.any():
                return np.zeros_like(x)
            inv = np.nanmax(x) - x
            inv[~fin] = np.nan
            m = np.nanmax(inv)
            if not np.isfinite(m) or m <= 1e-12:
                out = np.zeros_like(x); out[fin] = 1.0; return out
            return inv / m

        @staticmethod
        def _mut_code(curr, choices):
            c = [v for v in choices if v != curr]
            return curr if not c else np.random.choice(c)

        def _mut_layer(self, g, i, prob, dec_layers):
            r = self.mutation_rate * prob
            if np.random.rand() < r: g[self.ENC_FFN[i]] = self._mut_code(g[self.ENC_FFN[i]], self.C_FFN)
            if np.random.rand() < r: g[self.ENC_SA[i]]  = self._mut_code(g[self.ENC_SA[i]],  self.C_SA)
            if i >= dec_layers:
                g[self.DEC_FFN[i]] = g[self.DEC_SA[i]] = g[self.DEC_EN[i]] = g[self.DEC_AR[i]] = -1
                return
            if np.random.rand() < r: g[self.DEC_FFN[i]] = self._mut_code(g[self.DEC_FFN[i]], self.C_FFN)
            if np.random.rand() < r: g[self.DEC_SA[i]]  = self._mut_code(g[self.DEC_SA[i]],  self.C_SA)
            if np.random.rand() < r: g[self.DEC_EN[i]]  = self._mut_code(g[self.DEC_EN[i]],  self.C_EN)
            if np.random.rand() < r: g[self.DEC_AR[i]]  = self._mut_code(g[self.DEC_AR[i]],  self.C_AR)

        def _fix_pad(self, g):
            dl = int(g[1])
            for i in range(dl, 6):
                g[self.DEC_FFN[i]] = g[self.DEC_SA[i]] = g[self.DEC_EN[i]] = g[self.DEC_AR[i]] = -1
            return g

        def _do(self, problem, X, **kwargs):
            Xm = X.copy()
            n, _ = X.shape #individuals, variables -> so pop, 40
            for i in range(n): 
                parent = X[i].copy()
                for _ in range(self.max_retries):
                    child = parent.copy()
                    if np.random.rand() < self.mutation_rate: child[2] = self._mut_code(child[2], self.C_EMB)
                    if np.random.rand() < self.mutation_rate: child[3] = self._mut_code(child[3], self.C_EMB)
                    enc_imp, dec_imp, dl = self._layer_mags(problem, child)
                    enc_p = self._inv_scale(enc_imp)
                    dec_p = self._inv_scale(dec_imp)
                    for l in range(6):
                        p = enc_p[l] if np.isfinite(enc_p[l]) else 0.0
                        q = (dec_p[l] if np.isfinite(dec_p[l]) else 0.0)
                        self._mut_layer(child, l, prob=max(p, q), dec_layers=dl)
                    child = self._fix_pad(child)
                    if self._try_valid(problem, child):
                        Xm[i] = child
                        break
                else:
                    Xm[i] = parent
            return Xm
            

def _map(code, table, default_key=0):
    if code == -1:
        code = default_key
    return table[code]       
    