

import yaml
import numpy as np
import random

def test(design_space):

        benchmark_file = design_space.get('benchmark_file')
        fixed_length = design_space.get('fixed_length')
        #num_obj = len(design_space.get('metrics'))
        search_space = design_space.get('search_space', {})
        xl, xu = build_bounds(search_space)
        print(xl,"\n", xu)

def build_bounds(search_space):
    fixed_length = 22
    padding = -1
    # Gene 0: hidden_size
    hidden_sizes = search_space.get('hidden_size', [])
    ub_hidden = len(hidden_sizes) - 1

    # For layer parameters:
    op_types = search_space.get('operation_types', [])
    ub_op_type = len(op_types) - 1

    num_heads = search_space.get('num_heads', [])
    ub_num_heads = len(num_heads) - 1

    ff_hidden = search_space.get('feed-forward_hidden', [])
    ub_ff_hidden = len(ff_hidden) - 1

    ff_stacks = search_space.get('number_of_feed-forward_stacks', [])
    ub_ff_stacks = len(ff_stacks) - 1

    # For operation parameters, take the maximum across all types.
    op_params = search_space.get('operation_parameters', {})
    ub_op_param = 0
    for key, options in op_params.items():
        ub_op_param = max(ub_op_param, len(options) - 1)

    # Gene for number of layers:
    encoder_layers = search_space.get('encoder_layers', [])
    ub_encoder_layers = len(encoder_layers) - 1

    # Build the lower and upper bounds lists:
    lb = []
    ub = []
    
    # Gene 0: hidden_size
    lb.append(padding)
    ub.append(ub_hidden)
    
    # There are 4 layers, each with 5 genes.
    num_layers_max = 4
    for _ in range(num_layers_max):
        # Operation type:
        lb.append(padding)
        ub.append(ub_op_type)
        # Number of heads:
        lb.append(padding)
        ub.append(ub_num_heads)
        # Feed-forward hidden size:
        lb.append(padding)
        ub.append(ub_ff_hidden)
        # Number of feed-forward stacks:
        lb.append(padding)
        ub.append(ub_ff_stacks)
        # Operation parameter:
        lb.append(padding)
        ub.append(ub_op_param)
    
    # Final gene: number of layers
    lb.append(padding)
    ub.append(ub_encoder_layers)
    
    # Verify we have fixed_length elements
    if len(lb) != fixed_length or len(ub) != fixed_length:
        raise ValueError("Bounds length does not match fixed_length")
    
    return np.array(lb), np.array(ub)


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
          # Randomly decide on the number of layers (between 1 and 4)
        num_layers = random.choice([2, 4])

        # append number of layer as int
        genome.append(num_layers)

        # Randomly choose a hidden size and encode it.
        chosen_hidden_size = random.choice(list(hidden_size_mapping.keys()))
        genome.append(hidden_size_mapping[chosen_hidden_size])

      

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


        # pad genome with -1's until it reaches the fixed length.
        if len(genome) < fixed_length:
            genome.extend([-1] * (fixed_length - len(genome)))
        elif len(genome) > fixed_length:
            raise ValueError("Generated genome exceeds the fixed length.")

        return genome

design_space = "surrogate_benchmark_project/configs/flexibert_design_space.yaml"
search_space = "configs/evo_search_params.yaml"

with open(design_space, 'r') as f:
    design_space = yaml.safe_load(f)

print(generate_random_genome(23))