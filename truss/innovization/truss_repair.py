import logging

import numpy as np
from pymoo.core.repair import Repair

logger = logging.getLogger(__name__)


class ParameterlessInequalityRepair(Repair):

    def __init__(self, momentum_coefficient=0.3):
        super().__init__()

        self.percent_of_pop_in_pf = 0.8  # How much of pop should be in pf before innovization is done
        self.repair_interval = 10  # Generation gap between repairs
        # Used to control the influence momentum has on deciding the reference shape used for innovization
        self.momentum_coefficient = momentum_coefficient

    @staticmethod
    def calc_mean_std_without_outliers(x):
        """Calculates mean by eliminating outliers. Uses an iterative process similar to Expectation-Maximization"""
        # mean_new = np.inf * np.ones(x.shape[1])
        mean_new = np.mean(x, axis=0)
        std_new = np.std(x, axis=0)

        for i in range(x.shape[1]):
            x_arr = x[:, i]
            mean_init = np.inf

            # Mean value converged
            while np.abs(mean_init - mean_new[i]) > 1e-6:
                # Remove data points beyond 2sd of the mean
                x_arr = np.delete(x_arr, np.where(x_arr > (mean_new[i] + 2*std_new[i])))
                x_arr = np.delete(x_arr, np.where(x_arr < (mean_new[i] - 2*std_new[i])))

                mean_init = mean_new[i]  # Store the old value of mean

                # Update the new mean values
                mean_new[i] = np.mean(x_arr)
                std_new[i] = np.std(x_arr)

        return mean_new, std_new

    def learn_rules(self, problem, x):
        """From the population after non-dominated sorting learn innovization rules."""

        if x.size == 0:
            return None
        z = x[:, -problem.num_shape_vars:]
        r = x[:, :problem.num_size_vars]

        problem.z_avg, problem.z_std = ParameterlessInequalityRepair.calc_mean_std_without_outliers(z)

        # Get scores of each learned rule of the shape vars
        for indx in range(z.shape[1] - 1):
            curr_sol_z_i = z[:, indx]  # Get z(i) values for all solutions
            curr_sol_z_i_plus_1 = z[:, indx + 1]  # Get z(i+1) values for all solutions
            problem.shape_rule_score[indx] = 0
            for sol_indx in range(z.shape[0]):
                # Check how many solutions follow the monotonicity relations of z_avg
                if (((problem.z_avg[indx] < problem.z_avg[indx + 1])
                        and (curr_sol_z_i[sol_indx] < curr_sol_z_i_plus_1[sol_indx])))\
                        or (((problem.z_avg[indx] > problem.z_avg[indx + 1])
                            and (curr_sol_z_i[sol_indx] > curr_sol_z_i_plus_1[sol_indx]))):
                    problem.shape_rule_score[indx] += 1

        problem.shape_rule_score = problem.shape_rule_score / z.shape[0]  # Proportion of solutions following a rule

        problem.r_avg, problem.r_std = ParameterlessInequalityRepair.calc_mean_std_without_outliers(r)

        # Monotonic relations between shape variables
        # r_avg = np.mean(r, axis=0)
        # r_std = np.std(r, axis=0)
        # Remove solutions with r_i not within 2 standard deviations of mean(r_i)
        # r_without_outliers = [[] for _ in range(r.shape[1])]
        # for indx in range(r.shape[1]):
        #     arr = r[:, indx]
        #     final_list = [x for x in arr if ((x > (r_avg[indx] - 2 * r_std[indx]))
        #                                      or (x < (r_avg[indx] + 2 * r_std[indx]))
        #                                      )]
        #     r_without_outliers[indx] = copy.copy(final_list)
        #
        # for indx in range(r.shape[1]):
        #     problem.r_avg[indx] = np.mean(r_without_outliers[indx])
        #     problem.r_std[indx] = np.std(r_without_outliers[indx])

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

        # Calculate ref. z vector used for innovization
        if len(problem.z_ref_history) > 1:
            z_avg_momentum = problem.z_ref_history[-1] - problem.z_ref_history[-2]
        else:
            z_avg_momentum = 0

        problem.z_ref = problem.z_avg + self.momentum_coefficient*z_avg_momentum
        problem.z_ref_history.append(problem.z_ref)

        # Calculate ref. r vector used for innovization
        if len(problem.r_ref_history) > 1:
            r_avg_momentum = problem.r_ref_history[-1] - problem.r_ref_history[-2]
        else:
            r_avg_momentum = 0

        problem.r_ref = problem.r_avg + self.momentum_coefficient*r_avg_momentum
        problem.r_ref_history.append(problem.r_ref)

        return problem

    def _do(self, problem, x, **kwargs):
        """Perform innovization on offsprings according to previously learned rules."""
        # Do innovization every 10 generations and after at least 20 generations have been done
        if kwargs['algorithm'].n_gen % self.repair_interval != 0 or problem.percent_rank_0 < self.percent_of_pop_in_pf:
            return x

        print("========================================")
        print("Commencing parameterless innovization sequence")
        print("========================================")
        logger.info("Commencing parameterless monotonicity innovization sequence")

        x_repaired = np.copy(x)  # Repaired offspring population

        # The reference shape and radii after merging average and momentum info
        z_ref = problem.z_ref
        r_ref = problem.r_ref

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
                if (z_ref[j] < z_ref[j + 1]) and (z_repaired[j] > z_repaired[j + 1])\
                        and (rnd <= problem.shape_rule_score[j]):
                    # If z[j] > z[j + 1] shift z[j + 1] above z[j]
                    # Ensure after innovization z[j + 1] is within variable limits
                    z_repaired[j + 1] = np.minimum(z_repaired[j] + diff, z_xu[j + 1])

                if (z_ref[j] > z_ref[j + 1]) and (z_repaired[j] < z_repaired[j + 1]) \
                        and (rnd <= problem.shape_rule_score[j]):
                    # If z[j] < z[j + 1] shift z[j + 1] below z[j]
                    # Ensure after innovization z[j + 1] is within variable limits
                    z_repaired[j + 1] = np.maximum(z_repaired[j] - diff, z_xl[j + 1])
            x_repaired[i, -problem.num_shape_vars:] = z_repaired  # Set the repaired z-coordinates for the offspring

            # Repair size vars
            r = x[i, :problem.num_size_vars]
            r_repaired = np.copy(r)

            for indx, grp in enumerate(problem.grouped_size_vars):
                for j in range(len(grp) - 1):
                    # Repair any solutions not conforming to the average with a probability proportional to the rule
                    # score
                    # FIXME: Replace j with grp[j]?
                    diff = np.abs(r_repaired[j] - r_repaired[j + 1])
                    rnd = np.random.rand()
                    if (r_ref[j] < r_ref[j + 1]) and (r_repaired[j] > r_repaired[j + 1]) \
                            and (rnd < problem.size_rule_score[indx][j]):
                        # If z[j] > z[j + 1] shift z[j + 1] above z[j]
                        # Ensure after innovization z[j + 1] is within variable limits
                        r_repaired[j + 1] = np.minimum(r_repaired[j] + diff, r_xu[j + 1])

                    if (r_ref[j] > r_ref[j + 1]) and (r_repaired[j] < r_repaired[j + 1]) \
                            and (rnd < problem.size_rule_score[indx][j]):
                        # If z[j] < z[j + 1] shift z[j + 1] below z[j]
                        # Ensure after innovization z[j + 1] is within variable limits
                        r_repaired[j + 1] = np.maximum(r_repaired[j] - diff, r_xl[j + 1])
            x_repaired[i, :problem.num_size_vars] = r_repaired  # Set the repaired z-coordinates for the offspring

        logger.info("Parameterless innovization complete")

        # set the repaired design variables for the population
        # pop.set("X", x_repaired)

        return x_repaired
