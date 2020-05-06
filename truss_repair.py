from pymoo.model.repair import Repair
import numpy as np
import logging
from scipy.stats import trim_mean
import copy


logger = logging.getLogger(__name__)


class SimpleInequalityRepair(Repair):

    def __init__(self):
        super().__init__()
        # Choose to repair with a certain probability (Diversity preservation mechanism)
        self.repair_probability = 0.3

    @staticmethod
    def learn_rules(problem, x):
        problem.z_avg = np.average(x, axis=0)
        problem.z_std = np.std(x, axis=0)

    def _do(self, problem, x, **kwargs):
        # Do repair every 10 generations and after at least 20 generations have been done
        if kwargs['algorithm'].n_gen % 10 != 0 or problem.percent_rank_0 < 0.8:  # or kwargs['algorithm'].n_gen < 20:
            return x

        print("==========================")
        print("Commencing repair sequence")
        print("==========================")
        logger.info("Commencing repair sequence")

        # x = pop.get("X")
        x_repaired = np.copy(x)  # Repaired offspring population

        z_avg = problem.z_avg  # The average shape in previous good solutions

        z_xl = problem.xl[-problem.num_shape_vars:]
        z_xu = problem.xu[-problem.num_shape_vars:]

        for i in range(x.shape[0]):
            # Repair with a certain probability
            if np.random.rand() < self.repair_probability:
                continue

            z = x[i, -problem.num_shape_vars:]
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

            x_repaired[i, -problem.num_shape_vars:] = z_repaired  # Set the repaired z-coordinates for the offspring

        logger.info("Repair complete")

        # set the repaired design variables for the population
        # pop.set("X", x_repaired)

        return x_repaired


class ParameterlessInequalityRepair(Repair):

    @staticmethod
    def learn_rules(problem, x):
        """From the population after non-dominated sorting learn innovization rules."""
        z = x[:, -problem.num_shape_vars:]
        r = x[:, :problem.num_size_vars]

        # Remove 10% of the dataset from either end of the sorted z-values. Used to remove outliers
        # problem.z_avg = trim_mean(z, 0.1, axis=0)
        # problem.z_std = trim_mean(z, 0.1, axis=0)

        z_avg = np.mean(z, axis=0)
        z_std = np.std(z, axis=0)

        # Remove solutions with z_i not within 2 standard deviations of mean(z_i)
        z_without_outliers = [[] for _ in range(z.shape[1])]
        for indx in range(z.shape[1]):
            arr = z[:, indx]
            final_list = [x for x in arr if ((x > (z_avg[indx] - 2 * z_std[indx]))
                                             or (x < (z_avg[indx] + 2 * z_std[indx]))
                                             )]
            z_without_outliers[indx] = copy.copy(final_list)

        for indx in range(z.shape[1]):
            problem.z_avg[indx] = np.mean(z_without_outliers[indx])
            problem.z_std[indx] = np.std(z_without_outliers[indx])

        # Get scores of each learned rule of the shape vars
        for indx in range(z.shape[1] - 1):
            curr_sol_z_i = z[:, indx]  # Get z(i) values for all solutions
            curr_sol_z_i_plus_1 = z[:, indx + 1]  # Get z(i+1) values for all solutions
            problem.shape_rule_score[indx] = 0
            for sol_indx in range(z.shape[0]):
                # Check how many solutions follow the monotonicity relations of z_avg
                if ((problem.z_avg[indx] < problem.z_avg[indx + 1])
                        and (curr_sol_z_i[sol_indx] < curr_sol_z_i_plus_1[sol_indx]))\
                        or ((problem.z_avg[indx] > problem.z_avg[indx + 1])
                            and (curr_sol_z_i[sol_indx] > curr_sol_z_i_plus_1[sol_indx])):
                    problem.shape_rule_score[indx] += 1

        problem.shape_rule_score = problem.shape_rule_score / z.shape[0]  # Proportion of solutions following a rule

        # Monotonic relations between shape variables
        r_avg = np.mean(r, axis=0)
        r_std = np.std(r, axis=0)

        # Remove solutions with r_i not within 2 standard deviations of mean(r_i)
        r_without_outliers = [[] for _ in range(r.shape[1])]
        for indx in range(r.shape[1]):
            arr = r[:, indx]
            final_list = [x for x in arr if ((x > (r_avg[indx] - 2 * r_std[indx]))
                                             or (x < (r_avg[indx] + 2 * r_std[indx]))
                                             )]
            r_without_outliers[indx] = copy.copy(final_list)

        for indx in range(r.shape[1]):
            problem.r_avg[indx] = np.mean(r_without_outliers[indx])
            problem.r_std[indx] = np.std(r_without_outliers[indx])

        for indx, grp in enumerate(problem.grouped_size_vars):
            r_grp = r[:, grp]
            r_grp_avg = problem.r_avg[grp]
            problem.size_rule_score[indx] = np.zeros(problem.size_rule_score[indx].shape)
            for member_indx in range(r_grp.shape[1] - 1):
                curr_sol_r_i = r_grp[:, member_indx]  # Get z(i) values for all solutions
                curr_sol_r_i_plus_1 = r_grp[:, member_indx + 1]  # Get z(i+1) values for all solutions
                for sol_indx in range(r_grp.shape[0]):
                    # Check how many solutions follow the monotonicity relations of r_avg for each group of members
                    if ((r_grp_avg[member_indx] < r_grp_avg[member_indx + 1])
                        and (curr_sol_r_i[sol_indx] < curr_sol_r_i_plus_1[sol_indx])) \
                            or ((r_grp_avg[member_indx] > r_grp_avg[member_indx + 1])
                                and (curr_sol_r_i[sol_indx] > curr_sol_r_i_plus_1[sol_indx])):
                        problem.size_rule_score[indx][member_indx] += 1

            # Proportion of solutions following a rule
            problem.size_rule_score[indx] = problem.size_rule_score[indx] / r.shape[0]

        return problem.z_avg, problem.z_std, problem.shape_rule_score,\
            problem.r_avg, problem.r_std, problem.size_rule_score

    def _do(self, problem, x, **kwargs):
        """Perform repair on offsprings according to previously learned rules."""
        # Do repair every 10 generations and after at least 20 generations have been done
        if kwargs['algorithm'].n_gen % 10 != 0 or problem.percent_rank_0 < 0.8:
            return x

        print("========================================")
        print("Commencing parameterless repair sequence")
        print("========================================")
        logger.info("Commencing parameterless monotonicity repair sequence")

        # x = pop.get("X")
        x_repaired = np.copy(x)  # Repaired offspring population

        z_avg = problem.z_avg  # The average shape in previous good solutions
        r_avg = problem.r_avg

        z_xl = problem.xl[-problem.num_shape_vars:]
        z_xu = problem.xu[-problem.num_shape_vars:]
        r_xl = problem.xl[:problem.num_size_vars]
        r_xu = problem.xu[:problem.num_size_vars]

        for i in range(x.shape[0]):
            # Repair shape vars
            z = x[i, -problem.num_shape_vars:]
            z_repaired = np.copy(z)  # Repaired z-coordinates of the offspring

            for j in range(len(z) - 1):
                # Repair any solutions not conforming to the average with a probability proportional to the rule score
                diff = np.abs(z_repaired[j] - z_repaired[j + 1])
                rnd = np.random.rand()
                if (z_avg[j] < z_avg[j + 1]) and (z_repaired[j] > z_repaired[j + 1])\
                        and (rnd < problem.shape_rule_score[j]):
                    # If z[j] > z[j + 1] shift z[j + 1] above z[j]
                    # Ensure after repair z[j + 1] is within variable limits
                    z_repaired[j + 1] = np.minimum(z_repaired[j] + diff, z_xu[j + 1])

                if (z_avg[j] > z_avg[j + 1]) and (z_repaired[j] < z_repaired[j + 1]) \
                        and (rnd < problem.shape_rule_score[j]):
                    # If z[j] < z[j + 1] shift z[j + 1] below z[j]
                    # Ensure after repair z[j + 1] is within variable limits
                    z_repaired[j + 1] = np.maximum(z_repaired[j] - diff, z_xl[j + 1])
            x_repaired[i, -problem.num_shape_vars:] = z_repaired  # Set the repaired z-coordinates for the offspring

        for i in range(x.shape[0]):
            # Repair size vars
            r = x[i, :problem.num_size_vars]
            r_repaired = np.copy(r)

            for indx, grp in enumerate(problem.grouped_size_vars):
                for j in range(len(grp) - 1):
                    # Repair any solutions not conforming to the average with a probability proportional to the rule score
                    diff = np.abs(r_repaired[j] - r_repaired[j + 1])
                    rnd = np.random.rand()
                    if (r_avg[j] < r_avg[j + 1]) and (r_repaired[j] > r_repaired[j + 1]) \
                            and (rnd < problem.size_rule_score[indx][j]):
                        # If z[j] > z[j + 1] shift z[j + 1] above z[j]
                        # Ensure after repair z[j + 1] is within variable limits
                        r_repaired[j + 1] = np.minimum(r_repaired[j] + diff, r_xu[j + 1])

                    if (r_avg[j] > r_avg[j + 1]) and (r_repaired[j] < r_repaired[j + 1]) \
                            and (rnd < problem.size_rule_score[indx][j]):
                        # If z[j] < z[j + 1] shift z[j + 1] below z[j]
                        # Ensure after repair z[j + 1] is within variable limits
                        r_repaired[j + 1] = np.maximum(r_repaired[j] - diff, r_xl[j + 1])
            x_repaired[i, :problem.num_size_vars] = r_repaired  # Set the repaired z-coordinates for the offspring

        logger.info("Parameterless repair complete")

        # set the repaired design variables for the population
        # pop.set("X", x_repaired)

        return x_repaired
