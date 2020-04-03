from pymoo.model.repair import Repair
import numpy as np


class MonotonicityRepair(Repair):
    def __init__(self):
        # Choose to repair with a certain probability (Diversity preservation mechanism)
        self.repair_probability = 0.3
        super().__init__()

    def _do(self, problem, pop, **kwargs):
        # Do repair every 10 generations and after at least 20 generations have been done
        if kwargs['algorithm'].n_gen % 10 != 0 or problem.percent_rank_0 < 0.8:  # or kwargs['algorithm'].n_gen < 20:
            return pop

        print("==========================")
        print("Commencing repair sequence")
        print("==========================")

        x = pop.get("X")
        x_repaired = np.copy(x)  # Repaired offspring population

        z_avg = problem.z_avg  # The average shape in previous good solutions

        z_xl = problem.xl[-10:]
        z_xu = problem.xu[-10:]

        for i in range(x.shape[0]):
            # Repair with a certain probability
            if np.random.rand() < self.repair_probability:
                continue

            z = x[i, -10:]
            z_repaired = np.copy(z)  # Repaired z-coordinates of the offspring

            for j in range(len(z) - 1):

                if z_avg[j] < z_avg[j + 1] and z_repaired[j] > z_repaired[j + 1]:
                    # If z[j] > z[j + 1] randomly modify either variable to satisfy If z[j] < z[j + 1]
                    # r = np.random.rand()
                    # if r < 0.5:
                    #     z_repaired[j] = np.random.normal(loc=(z_avg[j] + z_avg[j + 1]) / 2, scale=1)
                    # else:
                    #     z_repaired[j + 1] = np.random.normal(loc=(z_avg[j] + z_avg[j + 1]) / 2, scale=1)

                    # If z[j] > z[j + 1] shift z[j + 1] above z[j]
                    diff = np.abs(z_repaired[j] - z_repaired[j + 1])
                    # Ensure after repair z[j + 1] is within variable limits
                    z_repaired[j + 1] = np.minimum(z_repaired[j] + diff, z_xu[j + 1])

                if z_avg[j] > z_avg[j + 1] and z_repaired[j] < z_repaired[j + 1]:
                    # Ensure after repair z[j + 1] is within variable limits
                    diff = np.abs(z_repaired[j] - z_repaired[j + 1])
                    z_repaired[j + 1] = np.maximum(z_repaired[j] - diff, z_xl[j + 1])

            x_repaired[i, -10:] = z_repaired  # Set the repaired z-coordinates for the offspring

            # A very crude repair where we preserve the average shape in good solutions among certain randomly
            # selected offsprings
            # repair_flag = False
            # for j in range(len(z) - 1):
            #     if z_avg[j] < z_avg[j + 1] and z[j] > z[j + 1]:
            #         repair_flag = True
            #         break
            #     if z_avg[j] > z_avg[j + 1] and z[j] < z[j + 1]:
            #         repair_flag = True
            #         break
            #
            # if repair_flag:
            #     z_repaired = np.copy(z_avg)

        # set the repaired design variables for the population
        pop.set("X", x_repaired)

        return pop
