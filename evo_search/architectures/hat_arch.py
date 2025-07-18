# /home/graham/Documents/surrogate_benchmark_project/evo_search/architectures/hat_arch.py
import random
import numpy as np
from .base_architecture import BaseArchitecture
import torch
import os
import sys
import re
from fairseq.models.transformer import TransformerModel
from fairseq.models.transformer_super import TransformerSuperModel
import pandas as pd
import yaml
import subprocess
import os, pathlib, sys
from latency_predictor import LatencyPredictor
from fairseq import tasks, utils
from fairseq import checkpoint_utils, sequence_generator

# HAT_REPO = "/home/graham/Documents/hardware-aware-transformers"
# import sys;  sys.path.append(HAT_REPO)        # so `latency_predictor` is importable

'''
import sys, pathlib, os
ROOT = pathlib.Path.cwd()
HAT_REPO = ROOT.parent / "hardware-aware-transformers"
sys.path.append(str(ROOT))
sys.path.append(str(HAT_REPO))

'''

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

        ensemble, self.args, self.task = checkpoint_utils.load_model_ensemble_and_task([self.CHECKPOINT_FILE], arg_overrides={"data": str(self.DATA_BIN)})

        self.model = ensemble[0]
        self.model.eval()

        args_ns = self.task.args                       # Namespace saved in checkpoint
        args_ns.dataset_impl = getattr(args_ns, "dataset_impl", "mmap")
        args_ns.combine       = getattr(args_ns, "combine", False)

        self.task.load_dataset("valid")
        #valid_set = self.task.dataset("valid")
        #self.dataloader = self.task.get_batch_iterator(dataset=valid_set).next_epoch_itr(shuffle=False)
        setattr(self.args, "sentence_avg", getattr(self.args, "sentence_avg", False))
        setattr(self.args, "criterion", getattr(self.args, "criterion", "cross_entropy"))
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
    
   
    def evaluate_latency(self, genome) -> float:
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

        loss_sum, sent_sum = 0.0, 0
        cuda = torch.cuda.is_available()

        with torch.no_grad():
            for idx, batch in enumerate(iterator):
                if idx >= max_batches:
                    break
                batch = utils.move_to_cuda(batch) if cuda else batch

                net_output                      = self.model(**batch["net_input"])
                loss, sample_size, _            = self.criterion(self.model, batch, net_output)
                loss_sum  += loss.item()
                sent_sum  += sample_size        # sentence-average criterion
        loss = loss_sum / sent_sum
        print (f"loss: {loss}")
        return loss_sum / sent_sum if sent_sum > 0 else float("nan")

    def _evaluate(self, x, out, *args, **kwargs):
        print("Starting evaluation:", x)
        loss = self.evaluate_validation_loss(x)
        latency = self.evaluate_latency(x)
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
        """
        Decode a 40-gene chromosome into HAT sub-transformer config.

        Genome layout
        -------------
        0   : encoder layer-num             ∈ {1..6}
        1   : decoder layer-num             ∈ {1..6}
        2   : encoder embed-dim             0→512, 1→640
        3   : decoder embed-dim             0→512, 1→640
        4-9 : encoder FFN dims (padded)     0→512, 1→1024, 2→2048
        10-15: decoder FFN dims (padded)     same mapping
        16-21: encoder SA heads (padded)     0→2,   1→4
        22-27: decoder SA heads (padded)     same mapping
        28-33: decoder En-De heads (padded)  same mapping
        34-39: decoder arbitrary attn        0→-1, 1→1, 2→2 (padded)
        """
        # -------- inverse maps --------
        map_embed  = {0: 512, 1: 640}
        map_ffn    = {0: 512, 1: 1024, 2: 2048}
        map_heads  = {0: 2,   1: 4}
        map_arbit  = {0: -1,  1: 1,  2: 2}

        # -------- slice genome --------
        enc_layers = int(gene[0])
        dec_layers = int(gene[1])

        enc_emb    = map_embed[int(gene[2])]
        dec_emb    = map_embed[int(gene[3])]

        enc_ffn_raw  = gene[4:10]
        dec_ffn_raw  = gene[10:16]
        enc_sa_raw   = gene[16:22]
        dec_sa_raw   = gene[22:28]
        dec_ende_raw = gene[28:34]
        dec_arbi_raw = gene[34:40]

        # keep only the first **n_layers** entries, then map to real values
        enc_ffn  = [map_ffn[int(v)]   for v in enc_ffn_raw[:enc_layers]]
        dec_ffn  = [map_ffn[int(v)]   for v in dec_ffn_raw[:dec_layers]]
        enc_sa   = [map_heads[int(v)] for v in enc_sa_raw[:enc_layers]]
        dec_sa   = [map_heads[int(v)] for v in dec_sa_raw[:dec_layers]]
        dec_ende = [map_heads[int(v)] for v in dec_ende_raw[:dec_layers]]
        dec_arbi = [map_arbit[int(v)] for v in dec_arbi_raw[:dec_layers]]

        # -------- build cfg dict --------
        return {
            "encoder": {
                "encoder_embed_dim":           enc_emb,
                "encoder_layer_num":           enc_layers,
                "encoder_ffn_embed_dim":       enc_ffn,
                "encoder_self_attention_heads": enc_sa,
            },
            "decoder": {
                "decoder_embed_dim":           dec_emb,
                "decoder_layer_num":           dec_layers,
                "decoder_ffn_embed_dim":       dec_ffn,
                "decoder_self_attention_heads": dec_sa,
                "decoder_ende_attention_heads": dec_ende,
                "decoder_arbitrary_ende_attn":  dec_arbi,
            },
        }
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

