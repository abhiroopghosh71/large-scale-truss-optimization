from pymoo.model.repair import Repair
import numpy as np


class MonotonicityRepair(Repair):
    def _do(self, problem, pop, **kwargs):
        x = pop.get("X")

        # set the design variables for the population
        pop.set("X", x)

        return pop
