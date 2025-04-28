
import os
import sys
import argparse
import numpy as np
import random
import yaml
import pandas as pd
import ast
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


'''
TODO:

0. take arguments for EA 
    -
1. take in hyperparam arguments for model, parse them
    -flexi: [SA, C, LT]
    -HAT [SA]
2. init population with genomes (same enconding as flexibert)

3. ngsaII step
    -latency -> bug negin
    -accuracy -> use task argument 

'''
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def run_dummy_search(self, iterations=10):
        """
        creates genomes, prints stuff
        """
        print("Starting Evolutionary Search...")
        for i in range(iterations):
            print(f"| Start Iteration {i}:")
            for idx, genome in enumerate(self.population):
                print(f"Iteration {i} - Genome {idx+1}: {genome}")


def unpackcsv(filename):
    df = pd.read_csv(filename)
    df['vector'] = df['vector'].apply(ast.literal_eval)
    #vectors = np.array(df['vector'].tolist())
    return df

def generate_random_genome(fixed_length):
    # Mapping dictionaries
    hidden_size_mapping = {128: 0, 256: 1}
    operation_type_mapping = {"SA": 0, "LT": 1, "DSC": 2}
    num_operation_heads_mapping = {2: 0, 4: 1}
    feed_forward_dimension_mapping = {512: 0, 1024: 1}
    num_feed_forward_mapping = {1: 0, 3: 1}
    SA_mapping = {"SDP": 0, "WMA": 1}
    LT_mapping = {"DFT": 0, "DCT": 1}
    DSC_mapping = {5: 0, 9: 1}

    genome = []

    # Randomly choose a hidden size and encode it.
    chosen_hidden_size = random.choice(list(hidden_size_mapping.keys()))
    genome.append(hidden_size_mapping[chosen_hidden_size])

    # Randomly decide on the number of layers (between 1 and 4)
    num_layers = random.randint(1, 4)

    # For each layer, randomly choose each parameter and append its encoded value.
    for i in range(num_layers):
        op_type = random.choice(list(operation_type_mapping.keys()))
        genome.append(operation_type_mapping[op_type])
        num_heads = random.choice(list(num_operation_heads_mapping.keys()))
        genome.append(num_operation_heads_mapping[num_heads])
        ff_dim = random.choice(list(feed_forward_dimension_mapping.keys()))
        genome.append(feed_forward_dimension_mapping[ff_dim])
        ff_num = random.choice(list(num_feed_forward_mapping.keys()))
        genome.append(num_feed_forward_mapping[ff_num])
        if op_type == "SA":
            param = random.choice(list(SA_mapping.keys()))
            genome.append(SA_mapping[param])
        elif op_type == "LT":
            param = random.choice(list(LT_mapping.keys()))
            genome.append(LT_mapping[param])
        elif op_type == "DSC":
            param = random.choice(list(DSC_mapping.keys()))
            genome.append(DSC_mapping[param])

    # append number of layer as int
    genome.append(num_layers)

    # pad genome with -1's until it reaches the fixed length.
    if len(genome) < fixed_length:
        genome.extend([-1] * (fixed_length - len(genome)))
    elif len(genome) > fixed_length:
        raise ValueError("Generated genome exceeds the fixed length.")

    return genome

def create_population(population, fixed_length):
    herd = []
    for x in range(population):
        herd.append(generate_random_genome(fixed_length))
    return herd



def parse():
    parser = argparse.ArgumentParser(
        description='input for EA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--design_space_file',
        metavar='', 
        type=str, 
        help='path to yaml file for the design space')


    design_args = parser.parse_args()
    print("design args: ", design_args)

    with open(design_args.design_space_file, 'r') as f:
        design_space = yaml.safe_load(f)
    print("Datasets:", design_space['datasets'])
    print("Hidden sizes:", design_space['architecture']['hidden_size'])
    print("Number of encoder layers:", design_space['architecture']['encoder_layers'])

    # Access a nested value, e.g., operation parameters for self-attention ("sa"):
    print("Self-attention parameters:", design_space['architecture']['operation_parameters']['sa'])


def main():
    df = unpackcsv("graham_folder/flexibert_data.csv")

    population = create_population(5, 22)
    for i, genome in enumerate(population):
        print(f"Genome {i+1}: {genome}")


if __name__ == '__main__':
    main()
	

# args = Namespace(encoder_layers=6,
#                      decoder_layers=7,
#                      encoder_embed_choice=[768, 512],
#                      decoder_embed_choice=[768, 512],
#                      encoder_ffn_embed_dim_choice=[3072, 2048],
#                      decoder_ffn_embed_dim_choice=[3072, 2048],
#                      encoder_layer_num_choice=[6, 5],
#                      decoder_layer_num_choice=[6, 5, 4, 3],
#                      encoder_self_attention_heads_choice=[8, 4],
#                      decoder_self_attention_heads_choice=[8, 4],
#                      decoder_ende_attention_heads_choice=[8],
#                      decoder_arbitrary_ende_attn_choice=[1, 2]
#                      )