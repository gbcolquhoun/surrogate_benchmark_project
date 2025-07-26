# /home/graham/Documents/surrogate_benchmark_project/evo_search/searcher.py
import os
import numpy as np
import yaml


from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.util.termination.max_gen import MaximumGenerationTermination


from evo_search.architectures.flexibert_arch import *
from evo_search.architectures.hat_arch import hat_architecture


class evo_search:
    def __init__(self, design_space, search_params):
        

        self.parse_design_space(design_space)
        self.parse_search_params(search_params)
        print(self.problem_arch)

    def parse_design_space(self, design_space):
        """load the design space YAML. sets up the architecture object"""
        print(f"parsing design space: {design_space}")
        with open(design_space, 'r') as f:
            self.design_space = yaml.safe_load(f)
        arch_type = self.design_space.get('architecture')

        if arch_type == "flexibert":
            self.problem_arch = flexibert_architecture(self.design_space)
        elif arch_type == "hat":
            self.problem_arch = hat_architecture(self.design_space)
        else:
            raise ValueError("architecture not found")

        #print("Hidden sizes:", self.design_space.get('architecture', {}).get('hidden_size'))
        #print("Number of encoder layers:", self.design_space.get('architecture', {}).get('encoder_layers'))
        #print("Self-attention parameters:", self.design_space.get('architecture', {}).get('operation_parameters', {}).get('sa'))

    def parse_search_params(self, search_params):
        """load the search space YAML"""
        print(f"parsing search space: {search_params}")
        with open(search_params, 'r') as f:
            self.search_params = yaml.safe_load(f)

        self.evolution_iterations = self.search_params.get('evolution_iterations')
        self.population_size = self.search_params.get('population_size')

    def run_evo_search(self, graph=True, text_save=True):

        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=self.problem_arch.custom_sampling(),
            crossover=self.problem_arch.custom_crossover_importance(),
            mutation=self.problem_arch.custom_mutation_importance(),
            eliminate_duplicates=self.problem_arch.custom_duplicate_elimination(),
            termination = MaximumGenerationTermination(self.evolution_iterations)
        )
        
        print(f"termination: {algorithm.termination}")
        res = minimize(
            self.problem_arch, 
            algorithm,
            seed=1, 
            verbose=True,
            copy_algorithm=False
        )
        
        # Print the results.
        print("NSGA-II Optimization Completed")
        print("================================")
        print(f"Number of generations: {self.evolution_iterations}")
        print(f"Population size: {self.population_size}")
        print("\nPareto Front Objectives (F):")
        print(res.F)
        print("\nPareto Front Solutions (X):")
        print(res.X)
        print("\nAdditional Info:")
        print("Algorithm details:", res.algorithm)
        
        # Save results to a text file if text_save is True.
        if text_save:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Compose the text content
            text_content = []
            text_content.append("NSGA-II Optimization Completed")
            text_content.append("================================")
            text_content.append(f"Number of generations: {self.evolution_iterations}")
            text_content.append(f"Population size: {self.population_size}")
            text_content.append("\nPareto Front Objectives (F):")
            text_content.append(np.array2string(res.F, precision=4, separator=', '))
            text_content.append("\nPareto Front Solutions (X):")
            text_content.append(np.array2string(res.X, precision=4, separator=', '))
            text_content.append("\nAdditional Info:")
            text_content.append("Algorithm details: " + str(res.algorithm))
            text_content = "\n".join(text_content)
            
            # Create directory for results if it doesn't exist.
            os.makedirs("results", exist_ok=True)
            text_file_path = f"results/evo_search_{timestamp}.txt"
            with open(text_file_path, "w") as f:
                f.write(text_content)
            print(f"Results saved to {text_file_path}")

    def run_exhaustive_search(self):
        self.problem_arch.generate_true_POF()