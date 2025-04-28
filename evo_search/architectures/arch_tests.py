import os
import numpy as np
import pandas as pd
#from .hat_arch import hat_architecture

import yaml

# --- Step 1. Create a test CSV benchmark file ---


# benchmark_file = "evo_search/datasets/iwslt14deen_gpu_titanxp_all.csv"

# # --- Step 2. Create a design_space dictionary ---
# # This dictionary mimics your YAML design space.
# design_space = {
#     "benchmark_file": benchmark_file,
#     "fixed_length": 40,
#     "metrics": ["accuracy", "latency"],  # dummy list; only its length matters here.
#     "architecture": "hat",
#     "search_space": {
#         "encoder-layer-num-choice": [6],
#         "decoder-layer-num-choice": [6, 5, 4, 3, 2, 1],
#         "encoder-embed-choice": [640, 512],
#         "decoder-embed-choice": [640, 512],
#         "encoder-ffn-embed-dim-choice": [2048, 1024, 512],
#         "decoder-ffn-embed-dim-choice": [2048, 1024, 512],
#         "encoder-self-attention-heads-choice": [4, 2],
#         "decoder-self-attention-heads-choice": [4, 2],
#         "decoder-ende-attention-heads-choice": [4, 2],
#         "decoder-arbitrary-ende-attn-choice": [-1, 1, 2],
#     }
# }

# # --- Step 3. Instantiate hat_architecture ---
# print("Instantiating hat_architecture...")
# hat_arch = hat_architecture(design_space)

# # --- Step 4. Define a known genome that should match the CSV row ---
# #
# # We construct a genome with the following:
# # Global positions:
# # 0: encoder layer num = 6
# # 1: decoder layer num = 2
# # 2: encoder embed dim: using embed_dim_mapping_inv {0:512, 1:640} so use 1 (->640)
# # 3: decoder embed dim: similarly 1 (->640)
# #
# # Per-layer parameters are stored in groups of 6:
# #
# # Encoder FFN (indices 4 to 9): we want mapped values [1024, 512, 512, 2048, 1024, 512]
# #   - Using ffn_embed_dim_mapping {512:0, 1024:1, 2048:2} → [1, 0, 0, 2, 1, 0]
# #
# # Decoder FFN (indices 10 to 15) for 2 layers: we want [512, 1024] → [0, 1] then pad with four -1's.
# #
# # Encoder self-attention heads (indices 16 to 21) for 6 layers: we want [2,2,2,4,2,2] → using sa_mapping {2:0,4:1} gives [0,0,0,1,0,0]
# #
# # Decoder self-attention heads (indices 22 to 27) for 2 layers: we want [4,4] → [1,1] then pad with four -1's.
# #
# # Decoder encoder-decoder heads (indices 28 to 33) for 2 layers: we want [4,4] → [1,1] then pad with four -1's.
# #
# # Decoder arbitrary attention (indices 34 to 39) for 2 layers: we want [1,2] (which gives average (1+2)/2 = 1.5)
# #   using arbitrary_mapping { -1:0, 1:1, 2:2 } → [1,2] then pad with four -1's.
# #
# genome = [
#     6,    # encoder layer num
#     2,    # decoder layer num
#     1,    # encoder embed dim (=>640)
#     1,    # decoder embed dim (=>640)
#     # Encoder FFN embed dims (6 genes):
#     1, 0, 0, 2, 1, 0,
#     # Decoder FFN embed dims (6 genes: first two for decoder layers, then padding):
#     0, 1, -1, -1, -1, -1,
#     # Encoder self-attention heads (6 genes):
#     0, 0, 0, 1, 0, 0,
#     # Decoder self-attention heads (6 genes: first two for layers, then padding):
#     1, 1, -1, -1, -1, -1,
#     # Decoder encoder-decoder heads (6 genes):
#     1, 1, -1, -1, -1, -1,
#     # Decoder arbitrary attention (6 genes):
#     1, 2, -1, -1, -1, -1
# ]
# genome = np.array(genome)
# print("Test genome:")
# print(genome)

# # --- Step 5. Test the evaluate_latency function ---
# print("Testing evaluate_latency()...")
# latency = hat_arch.evaluate_latency(genome)
# print(f"Computed latency: {latency}")

# # Expected latency is the sum of latency_mean_encoder and latency_mean_decoder from our CSV.
# # With our CSV row: 5.0 + 120.0 = 125.0



