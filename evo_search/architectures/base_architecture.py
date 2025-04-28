# evo_search/architectures/base_architecture.py
from abc import ABC, abstractmethod
import numpy as np


# Import pymoo base classes for operators and problems
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination

'''
to maybe unify:
init() could be base-cased, would need to make:
- abstract unpack_csv()
- abstract build_bounds()

sampling could be base-cased, would need
- abstract generate_random_genome()
'''
class BaseArchitecture(Problem, ABC):
    Sampling = Sampling
    Crossover = Crossover
    Mutation = Mutation
    ElementwiseDuplicateElimination = ElementwiseDuplicateElimination
    
    @abstractmethod
    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu, vtype=None, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_ieq_constr, xl=xl, xu=xu, type_var=vtype, elementwise_evaluation=True, **kwargs)

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        return super()._evaluate(x, out, *args, **kwargs)
    
    class custom_sampling(Sampling):

        def _do(self, problem, n_samples, **kwargs):

            return None
        

    class custom_crossover(Crossover):

        def __init__(self, n_parents, n_offsprings, prob, **kwargs):
            super().__init__(n_parents, n_offsprings, prob, **kwargs)


        def _do(self, problem, X, **kwargs):
            return None
        

    class custom_mutation(Mutation):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)


        def _do(self, problem, X, **kwargs):

            return None


    class custom_duplicate_elimination(ElementwiseDuplicateElimination):

        def is_equal(self, a, b):
            return a.X[0] == b.X[0]

