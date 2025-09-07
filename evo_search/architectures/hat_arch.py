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

        ensemble, self.args, self.task = checkpoint_utils.load_model_ensemble_and_task([self.CHECKPOINT_FILE], arg_overrides={"data": str(self.DATA_BIN)})
        self.model = ensemble[0]

        self._enable_magnitude_selection(self.model, metric='l2')
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

    
    def placeholder_evaluate_accuracy(self, genome):
        """
        Placeholder accuracy evaluation that returns a random float between 25 and 40.
        """
        return random.uniform(25, 40)
    
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

    def evaluate_latency(self, genome) -> float:
        # ---------- build the sub‑transformer dict -----------------
        sub_cfg = self._genome_to_subtransformer_cfg(genome)

        # ---------- latency prediction -----------------------------
        latency_ms = self.latency_predictor.predict_lat(sub_cfg)
        print(f"latency: {latency_ms} ms")
        return latency_ms
    
    def evaluate_validation_loss(self, genome, max_batches: int = 10) -> float:
        """
        compute label-smoothed cross entropy loss using dataloader
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
            
            loss = self.evaluate_validation_loss(x)
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
        # ---------- inverse maps ----------
        map_embed = {0: 512, 1: 640}
        map_ffn   = {0: 512, 1: 1024, 2: 2048}
        map_head  = {0: 2,   1: 4}
        map_arbi  = {0: -1,  1: 1,   2: 2}

        # ---------- slice ----------
        enc_layers, dec_layers = int(gene[0]), int(gene[1])
        enc_emb = map_embed[int(gene[2])]
        dec_emb = map_embed[int(gene[3])]

        enc_ffn_raw  = gene[4:10]
        dec_ffn_raw  = gene[10:16]
        enc_sa_raw   = gene[16:22]
        dec_sa_raw   = gene[22:28]
        dec_ende_raw = gene[28:34]
        dec_arbi_raw = gene[34:40]

        # ---------- safe mapping (ignore padding) ----------
        enc_ffn  = [_map(v, map_ffn)   for v in enc_ffn_raw[:enc_layers]]
        dec_ffn  = [_map(v, map_ffn)   for v in dec_ffn_raw[:dec_layers]]
        enc_sa   = [_map(v, map_head)  for v in enc_sa_raw[:enc_layers]]
        dec_sa   = [_map(v, map_head)  for v in dec_sa_raw[:dec_layers]]
        dec_ende = [_map(v, map_head)  for v in dec_ende_raw[:dec_layers]]
        dec_arbi = [_map(v, map_arbi)  for v in dec_arbi_raw[:dec_layers]]

        # ---------- build dict ----------
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
        """Stable string representation – JSON keeps order & is easy to read."""
        return json.dumps(genome.tolist())          # e.g. "[6,2,0,1,...,-1]"
    
    def _load_lut(self, lut_path: str):

        lut_file = Path(lut_path)
        if lut_file.exists():
            self.lut = pd.read_csv(lut_file, index_col="genome_key")  
        else:
            self.lut = pd.DataFrame(columns=["loss", "latency"])

    def _save_lut(self):
        lut_path = Path("/home/graham/Documents/cache/hat_lut.csv")
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
        
        def _total_mag(self, problem, genome):
            cfg   = problem._genome_to_subtransformer_cfg(genome)
            model = problem.model
            model.set_sample_config(cfg)
            param_means = [param.abs().mean().item() for param in model.parameters()]
            total_mag = sum(param_means)
            return total_mag
        
        def _do(self, problem, X, **kwargs):

            n_parents, n_matings, n_var = X.shape # X expected to have shape (2, n_matings, 40)
            offspring = np.empty_like(X)
            
            global_indices = list(range(0, 4))
            encoder_group_ranges = [(4, 10),    # encoder FFN embed dims
                                    (16, 22)]   # encoder self-attention heads
                              
            decoder_group_ranges =  [(10, 16),  # decoder FFN embed dims
                                    (22, 28),   # decoder self-attention heads 
                                    (28, 34),   # decoder encoder-decoder heads
                                    (34, 40)]   # decoder arbitrary attention  

            for j in range(n_matings):
                parent_A = X[0, j] # split up parents from x variable
                parent_B = X[1, j]
                
                mA, mB = self._total_mag(problem, parent_A), self._total_mag(problem, parent_B)
                pa = 0.5 if mA + mB == 0 else mA / (mA + mB)

                for i in range(n_parents):  # i=0 and i=1 for two offspring
                    child = np.empty(n_var, dtype=X.dtype)
                    
                    
                    for idx in global_indices: # for singular genes 
                        donor = parent_A if np.random.rand() < pa else parent_B
                        child[idx] = donor[idx]
                    
                    for (start, end) in encoder_group_ranges: # for each group of layer parameters, choose the entire block from one parent
                        donor = parent_A if np.random.rand() < pa else parent_B
                        child[start:end] = donor[start:end]
                    
                    dec_layers = int(child[1]) 

                    for (start, end) in decoder_group_ranges:      
                        for k in range(6):
                            pos = start + k
                            if k < dec_layers:           # the real layers
                                donor = parent_A if np.random.rand() < pa else parent_B
                                child[pos] = donor[pos]
                            else:                       
                                child[pos] = -1

                    offspring[i, j, :] = child
                    
            return offspring
        
    class custom_mutation_importance(BaseArchitecture.Mutation):

        def __init__(self, mutation_rate=0.1, n_mut_blocks=1, max_retries=3, **kwargs):

            super().__init__(**kwargs)
            self.n_mut_blocks = n_mut_blocks
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
                
            # fixed groups used everywhere in the architecture
            self.global_idx  = list(range(0, 4))
            self.group_ranges = [
                (4, 10),   # enc-FFN dims
                (10, 16),  # dec-FFN dims
                (16, 22),  # enc-SA heads
                (22, 28),  # dec-SA heads
                (28, 34),  # dec En-De heads
                (34, 40)   # dec arbitrary attn
            ]

        def _do(self, problem, X, **kwargs):
            X_mut = X.copy()
            n_ind, n_var = X.shape

            lb, ub = problem.xl, problem.xu          # bounds per gene
            FIXED = {0, 1}                           # <- keep encoder/decoder depth

            for i in range(n_ind):
                for j in range(n_var):
                    if j in FIXED:                   # skip the two block-count genes
                        continue
                    if np.random.rand() < self.mutation_rate:
                        X_mut[i, j] = np.random.randint(lb[j], ub[j] + 1)

            return X_mut
        

        def _block_mags(self, problem, genome):
            """return average |weight| for each group -> importance score"""
            cfg   = problem._genome_to_subtransformer_cfg(genome)
            model = problem.model
            model.set_sample_config(cfg)

            mags = []
            for p in model.parameters():
                mags.append(p.detach().abs().mean().item())

            mean_mag = np.mean(mags)
            return [mean_mag] * (len(self.group_ranges) + len(self.global_idx))

        def _pick_blocks(self, I_tilde):
            """sample indices according to  (1-Î) / Σ(1-Î)"""
            prob = (1.0 - I_tilde) / np.sum(1.0 - I_tilde)
            return np.random.choice(len(I_tilde),
                                    size=self.n_mut_blocks,
                                    replace=False,
                                    p=prob)


def _map(code, table, default_key=0):
    if code == -1:
        code = default_key
    return table[code]       
    