# evo_search/architectures/flexibert_arch.py
import random
from .base_architecture import BaseArchitecture
import numpy as np

class flexibert_architecture(BaseArchitecture):
    def __init__(self, design_space):

        benchmark_file = design_space.get('benchmark_file')
        self.fixed_length = design_space.get('fixed_length')
        num_obj = len(design_space.get('metrics'))

        self.data_df = self.unpack_csv(benchmark_file)
        #print (f"dataframe: {self.data_df}")
        search_space = design_space.get('search_space', {})
        xl, xu = self.build_bounds(search_space)
        super().__init__(n_var=self.fixed_length, n_obj=num_obj, n_ieq_constr=0, xl=xl, xu=xu, vtype=int)

    def unpack_csv(self, benchmark_file):
        """
        Load CSV file and convert genome columns to integers.
        
        Assumes the CSV has columns like 'vector_0', 'vector_1', ..., 'vector_n',
        as well as columns 'accuracy' and 'latency'.
        """
        import pandas as pd

        #print( "unpacking csv")
        self.data_df = pd.read_csv(benchmark_file)
        
        # Identify genome columns (assuming they start with "vector_")
        genome_cols = [col for col in self.data_df.columns if col.startswith("vector_")]
        
        # Convert genome columns to integer type if they are not already
        self.data_df[genome_cols] = self.data_df[genome_cols].astype(int)
        
        return self.data_df


    def build_bounds(self, search_space):

        padding = -1

        # gene 1: hidden_size
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
        
        # gene 0: number of layers
        lb.append(padding)
        ub.append(ub_encoder_layers)

        # Gene 1: hidden_size
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
        
        
        # Verify we have fixed_length elements
        if len(lb) != self.fixed_length or len(ub) != self.fixed_length:
            raise ValueError("Bounds length does not match fixed_length")
        
        return np.array(lb), np.array(ub)

    def _evaluate(self, x, out, *args, **kwargs):
        # Convert candidate genome to a list of integers.
        candidate = [int(round(i)) for i in x]
        
        # Build a list of column names for the genome.
        genome_cols = [f"vector_{i}" for i in range(self.fixed_length)]

        # Debug: print the genome columns being used.
        #print("Genome columns:", genome_cols)

        # Define a helper function that extracts the genome from a DataFrame row.
        def row_genome_debug(row, cols):
            genome = []
            for col in cols:
                val = row[col]
                #print(f"  {col} -> {val}")  # Debug print for each column value.
                genome.append(val)
            #print("  Constructed genome from row:", genome)
            return genome

        # Create an empty list to store indices of matching rows.
        matching_indices = []

        # Iterate over all rows in the DataFrame.
        for idx, row in self.data_df.iterrows():
            #print(f"Processing row index {idx}:")
            row_gen = row_genome_debug(row, genome_cols)
            if row_gen == candidate:
                #print(f"Row {idx} matches candidate!")
                matching_indices.append(idx)
            else:
                continue
                #print(f"Row {idx} does NOT match candidate.")

        # Now filter the DataFrame using the indices where a match was found.
        match = self.data_df.loc[matching_indices]


        if not match.empty:
            #print("Matching rows found:")
            #print(match)
            row = match.iloc[0]
            latency = float(row['latency'])
            accuracy = float(row['accuracy'])
        else:
            #print('agghh!! genome not found')
            latency = 1e6    # High penalty
            accuracy = 0     # Low accuracy penalty
        
        # Since NSGA-II minimizes objectives, return latency and negative accuracy.
        out["F"] = np.array([latency, -accuracy])
    
    def generate_random_genome(self, fixed_length):
        """generates a randome genome according to FlexiBERT encoding scheme, pads to fixed length

        TODO: take in encoding scheme as argument to allow for different model architectures
              change so it only produces valid genomes
        """
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

    def generate_true_POF(self):
        """
        Use pymoo's non-dominated sorting to extract the pareto front using the entire tabluar dataset
        This is the true/exhaustive solution.
        """

        import matplotlib.pyplot as plt
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        if self.data_df is None:
            print('no csv')

        # prepare objectives: latency and "negative accuracy"
        F = self.data_df[['latency', 'accuracy']].values.astype(float)
        F[:, 1] = -F[:, 1]  # convert accuracy to minimization objective
        nds = NonDominatedSorting()
        pareto_indices = nds.do(F, only_non_dominated_front=True)
        pareto_front = self.data_df.iloc[pareto_indices].copy()  # copy to allow adding columns
        
        print(pareto_front)
        
        print("Pareto optimal solutions from CSV:")
        for idx, row in pareto_front.iterrows():
            print("Genome: \n", row[:-2])
            print("Latency:", row['latency'], "Accuracy:", row['accuracy'])
            print("-" * 40)

        colors = {2: 'red', 4: 'blue'}

        plt.figure(figsize=(8, 6))
        for num_layers, group in pareto_front.groupby('vector_0'):
            plt.scatter(group['latency'], group['accuracy'],
                        color=colors.get(num_layers, 'black'),
                        label=f"{num_layers} layer{'s' if num_layers > 1 else ''}")

        plt.xlabel('Latency')
        plt.ylabel('Accuracy')
        plt.title('Exhaustive Pareto Front of FlexiBERT on GLUE benchmark')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_valid_genome(self):
        genome_cols = [f"vector_{i}" for i in range(self.fixed_length)]
        random_row = self.data_df.sample(n=1).iloc[0]
        # Extract the genome values from the sampled row and convert them to integers.
        genome = [int(random_row[col]) for col in genome_cols]
        #print(genome)
        return np.array(genome)



    class custom_sampling(BaseArchitecture.Sampling):

        def _do(self, problem, n_samples, **kwargs):
            samples = []
            for _ in range(n_samples):
                genome = problem.generate_valid_genome()
                samples.append(genome)
            return np.array(samples)
            
    class custom_crossover(BaseArchitecture.Crossover):

        def __init__(self, **kwargs):
            super().__init__(n_parents=2, n_offsprings=2, prob=0.9, **kwargs)


        def _do(self, problem, X, **kwargs):

            """
            Perform structure-aware crossover.

            Assumes X has shape (n_parents, n_matings, n_var), i.e.:
                X[0, k, :] is parent1 for mating event k,
                X[1, k, :] is parent2 for mating event k.
            """
            # Unpack dimensions: here, n_parents should be 2.
            n_parents, n_matings, n_var = X.shape
            
            # Define genome structure parameters.
            # todo: make these design inputs
            block_size = 5     # Each layer block has 5 genes.
            max_layers = 4     # Maximum number of layers.
            
            # Prepare lists for the two offspring.
            offspring1 = []
            offspring2 = []
            
            # For each mating event (indexed by k)
            for k in range(n_matings):
                parent1 = X[0, k, :]  # Shape: (n_var,)
                parent2 = X[1, k, :]  # Shape: (n_var,)
                
                # Generate one child using our structure-aware crossover.
                child1 = self._crossover_pair(parent1, parent2, n_var, block_size, max_layers)
                child2 = self._crossover_pair(parent1, parent2, n_var, block_size, max_layers)
                # For demonstration, we create two identical offspring.
                # (You might later add variation to create two distinct offspring.)
                offspring1.append(child1)
                offspring2.append(child2)
            
            # Return array with shape (n_offsprings, n_matings, n_var)
            return np.array([offspring1, offspring2])
    
        def _crossover_pair(self, parent1, parent2, fixed_length, block_size, max_layers):
            """
            Create one offspring from two parents using layer-wise crossover.
            
            The new encoding is assumed to have:
            - Gene 0: number of layers.
            - Gene 1: hidden size.
            - Genes 2 ... : layer blocks (each block of length block_size).
            """
            child = []
            
            # 1. For gene 0 (number of layers): randomly choose from one parent.
            num_layers = parent1[0] if np.random.random() < 0.5 else parent2[0]
            child.append(num_layers)
            
            # 2. For gene 1 (hidden size): randomly choose from one parent.
            hidden_size = parent1[1] if np.random.random() < 0.5 else parent2[1]
            child.append(hidden_size)
            
            # 3. For each potential layer (from 1 to max_layers):
            for i in range(1, max_layers + 1):
                # Determine if each parent has an active layer at index i.
                # In the new encoding, parent's gene 0 holds the number of layers.
                p1_has = (parent1[0] >= i)
                p2_has = (parent2[0] >= i)
                
                # Compute block indices: layer blocks start at index 2.
                start = 2 + (i - 1) * block_size
                end = start + block_size
                
                if p1_has and p2_has:
                    # Both have the layer: pick the block from one parent randomly.
                    block = parent1[start:end] if np.random.random() < 0.5 else parent2[start:end]
                elif p1_has:
                    # Only parent1 has the layer.
                    block = parent1[start:end]
                elif p2_has:
                    # Only parent2 has the layer.
                    block = parent2[start:end]
                else:
                    # Neither parent has an active layer at this index: use padding.
                    block = np.full(block_size, -1)
                
                child.extend(block)
            
            # 4. Ensure the child genome has the expected fixed length.
            if len(child) < fixed_length:
                child.extend([-1] * (fixed_length - len(child)))
            elif len(child) > fixed_length:
                print ("overflow??")
                child = child[:fixed_length]
            
            return np.array(child)
        
    class custom_mutation(BaseArchitecture.Mutation):

        def __init__(self, mutation_rate=0.1, **kwargs):
            super().__init__(**kwargs)
            self.mutation_rate = mutation_rate


        def _do(self, problem, X, **kwargs):
            block_size = 5
            # Iterate over each candidate in the population.
            for i in range(X.shape[0]):
                candidate = X[i].copy()
                # Candidate's first gene gives the number of active layers.
                num_layers = int(candidate[0])
                # For each gene index:
                for j in range(X.shape[1]):
                    # Genes 0 and 1 are always active.
                    if j >= 2:
                        # Determine which layer block this gene belongs to.
                        # For j>=2, layer index = floor((j-2)/block_size)+1.
                        layer_index = (j - 2) // block_size + 1
                        # If this layer index is greater than the number of active layers, it's padding.
                        if layer_index > num_layers:
                            continue  # Skip mutation for padding genes.
                    # With mutation probability, mutate this gene.
                    if np.random.random() < self.mutation_rate:
                        # Select a new random value from the allowed range.
                        # Note: problem.xl and problem.xu are assumed to be arrays of length n_var.
                        new_val = np.random.randint(int(problem.xl[j]), int(problem.xu[j]) + 1)
                        candidate[j] = new_val
                # Update the candidate.
                X[i] = candidate
            return X

    class custom_duplicate_elimination(BaseArchitecture.ElementwiseDuplicateElimination):

        def is_equal(self, a, b):
            # Using numpy's array_equal to compare full genome vectors.
            return np.array_equal(a.X, b.X)