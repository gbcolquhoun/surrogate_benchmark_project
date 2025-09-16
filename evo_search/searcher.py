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
        
        self.use_weight_magnitude_selection = self.search_params.get('use_weight_magnitude_selection', False)
        self.selection_metric = self.search_params.get('selection_metric', 'l2')
        self.use_importance_operators = self.search_params.get('use_importance_operators', False)

    def run_evo_search(self, graph=True, text_save=True):
    
        if self.use_weight_magnitude_selection:
            # turn ON magnitude-based selection (LinearSuper + MHA Super blocks)
            if hasattr(self.problem_arch, "enable_weight_magnitude"):
                self.problem_arch.enable_weight_magnitude(metric=self.selection_metric)
        else:
            # turn OFF (return to prefix)
            if hasattr(self.problem_arch, "disable_weight_magnitude"):
                self.problem_arch.disable_weight_magnitude()

         # pick EA operators
        if self.use_importance_operators:
            crossover_op = self.problem_arch.custom_crossover_importance()
            mutation_op  = self.problem_arch.custom_mutation_importance()
            ops_tag = "ops=importance"
        else:
            crossover_op = self.problem_arch.custom_crossover()
            mutation_op  = self.problem_arch.CustomMutation()
            ops_tag = "ops=vanilla"

        algorithm = NSGA2(
        pop_size=self.population_size,
        sampling=self.problem_arch.custom_sampling(),
        crossover=crossover_op,
        mutation=mutation_op,
        eliminate_duplicates=self.problem_arch.custom_duplicate_elimination(),
        termination=MaximumGenerationTermination(self.evolution_iterations),
        )
        
        print(f"termination: {algorithm.termination}")
        res = minimize(
            self.problem_arch, 
            algorithm,
            seed=2, 
            verbose=True,
            copy_algorithm=False
        )
        
        if text_save:
            import datetime, os, json
            import numpy as np
            import pandas as pd

            # --- identify run + modes for easy comparison filenames ---
            timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_tag    = "sel=magnitude" if getattr(self, "use_weight_magnitude_selection", False) else "sel=prefix"
            metric_tag  = f"metric={getattr(self, 'selection_metric', 'na')}" if getattr(self, "use_weight_magnitude_selection", False) else "metric=na"
            ops_tag     = "ops=importance" if getattr(self, "use_importance_operators", False) else "ops=vanilla"
            run_id      = f"{timestamp}_{mode_tag}_{metric_tag}_{ops_tag}"

            # --- create a dedicated run folder ---
            run_dir = os.path.join("results", run_id)
            os.makedirs(run_dir, exist_ok=True)

            # --- convenience helpers ---
            def _as_list(x):
                # Handle None or empty gracefully
                if x is None:
                    return []
                return x.tolist() if hasattr(x, "tolist") else list(x)

            # --- build headers for CSVs (genes + objectives) ---
            n_var = getattr(self.problem_arch, "n_var", None)
            X_cols = [f"gene_{i}" for i in range(n_var)] if n_var is not None else None
            # If you have named objectives, put them here in order:
            F_cols = ["latency_ms", "valid_loss"]

            # --- save combined front as CSV (one row per nondominated solution) ---
            X = res.X if res.X is not None else np.empty((0, 0))
            F = res.F if res.F is not None else np.empty((0, 0))
            if X_cols is None:  # fallback if n_var wasn't set
                X_cols = [f"gene_{i}" for i in range(X.shape[1])] if X.size else []

            combined = None
            if X.size and F.size and X.shape[0] == F.shape[0]:
                dfX = pd.DataFrame(X, columns=X_cols)
                dfF = pd.DataFrame(F, columns=F_cols[:F.shape[1]])
                combined = pd.concat([dfX, dfF], axis=1)
                combined_path = os.path.join(run_dir, "pareto_front.csv")
                combined.to_csv(combined_path, index=False)

            # --- save separate CSVs as well (sometimes handy) ---
            if X.size:
                pd.DataFrame(X, columns=X_cols).to_csv(os.path.join(run_dir, "pareto_X.csv"), index=False)
            if F.size:
                pd.DataFrame(F, columns=F_cols[:F.shape[1]]).to_csv(os.path.join(run_dir, "pareto_F.csv"), index=False)

            # --- also save raw arrays in NPZ (lossless, quick to reload) ---
            np.savez_compressed(os.path.join(run_dir, "pareto_front.npz"), X=X, F=F)

            # --- store meta + settings for reproducibility ---
            meta = {
                "timestamp": timestamp,
                "run_id": run_id,
                "modes": {
                    "use_weight_magnitude_selection": getattr(self, "use_weight_magnitude_selection", False),
                    "selection_metric": getattr(self, "selection_metric", None),
                    "use_importance_operators": getattr(self, "use_importance_operators", False),
                },
                "evolution": {
                    "population_size": self.population_size,
                    "generations": self.evolution_iterations,
                    "termination": str(res.algorithm.termination) if hasattr(res, "algorithm") else None,
                    "seed": getattr(self, "seed", None),
                },
                "algorithm": str(res.algorithm),
                "notes": "pareto_front.csv has genes+objectives combined; pareto_X.csv and pareto_F.csv are split; pareto_front.npz is raw arrays."
            }
            # If you want to also store the parsed YAMLs:
            try:
                meta["design_space"] = self.design_space
            except Exception:
                meta["design_space"] = "unavailable"
            try:
                meta["search_params"] = self.search_params
            except Exception:
                meta["search_params"] = "unavailable"

            with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # --- human-readable summary.txt (like your original, but richer) ---
            lines = []
            lines.append("NSGA-II Optimization Completed")
            lines.append("================================")
            lines.append(f"Run ID: {run_id}")
            lines.append(f"Population size: {self.population_size}")
            lines.append(f"Generations: {self.evolution_iterations}")
            lines.append(f"Selection mode: {'weight-magnitude' if meta['modes']['use_weight_magnitude_selection'] else 'prefix'}")
            lines.append(f"Selection metric: {meta['modes']['selection_metric']}")
            lines.append(f"Operators: {'importance-based' if meta['modes']['use_importance_operators'] else 'vanilla'}")
            lines.append("")
            lines.append("Algorithm:")
            lines.append(str(res.algorithm))
            lines.append("")
            lines.append("Pareto Front (F):")
            lines.append(np.array2string(F, precision=4, separator=', '))
            lines.append("")
            lines.append("Pareto Solutions (X):")
            lines.append(np.array2string(X, precision=4, separator=', '))
            lines.append("")
            lines.append("Files:")
            lines.append(f" - {os.path.join(run_dir, 'pareto_front.csv') if combined is not None else '(no pareto_front.csv)'}")
            lines.append(f" - {os.path.join(run_dir, 'pareto_X.csv') if X.size else '(no pareto_X.csv)'}")
            lines.append(f" - {os.path.join(run_dir, 'pareto_F.csv') if F.size else '(no pareto_F.csv)'}")
            lines.append(f" - {os.path.join(run_dir, 'pareto_front.npz')}")
            lines.append(f" - {os.path.join(run_dir, 'run_meta.json')}")

            with open(os.path.join(run_dir, "summary.txt"), "w") as f:
                f.write("\n".join(lines))

            print(f"Results saved under: {run_dir}")
    def run_exhaustive_search(self):
        self.problem_arch.generate_true_POF()