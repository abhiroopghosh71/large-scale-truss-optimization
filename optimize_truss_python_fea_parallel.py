import argparse
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from pymoo.util.display import MultiObjectiveDisplay

from truss.repair.truss_repair import ParameterlessInequalityRepair
from truss.truss_symmetric import TrussProblemSymmetric
from truss.truss_symmetric_shape_only import TrussProblem

save_folder = os.path.join('output', 'truss_optimization_nsga2')


class OptimizationDisplay(MultiObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

        f = algorithm.pop.get("F")
        f_min_weight = np.round(f[f[:, 0] == np.min(f[:, 0]), :], decimals=2).flatten()
        f_min_compliance = np.round(f[f[:, 1] == np.min(f[:, 1]), :], decimals=2).flatten()
        self.output.append("Min. weight solution", f_min_weight)
        self.output.append("Min. compliance solution", f_min_compliance)
        self.output.append("cv(min.)", np.min(algorithm.pop.get('CV')))
        self.output.append("cv(max.)", np.max(algorithm.pop.get('CV')))

        logging.info("===============================================================================================")
        logging.info("n_gen |  n_eval | Min. weight solution      | Min. compliance solution      |   cv(min.)   |   "
                     "cv(max.)  ")
        logging.info(f"{algorithm.n_gen}    | {algorithm.n_gen * algorithm.pop_size}      | "
                     f"{f_min_weight} | {f_min_compliance} |   {np.min(algorithm.pop.get('CV'))}   |   "
                     f"{np.max(algorithm.pop.get('CV'))}  ")
        logging.info("===============================================================================================")


def record_state(algorithm):
    x_pop = algorithm.pop.get('X')
    f_pop = algorithm.pop.get('F')
    rank_pop = algorithm.pop.get('rank')
    g_pop = None
    cv_pop = None

    if algorithm.problem.n_constr > 0:
        g_pop = algorithm.pop.get('G')
        cv_pop = algorithm.pop.get('CV')
    # algorithm.problem.z_monotonicity_matrix = get_monotonicity_pattern(x_pop, f_pop, rank_pop)

    if hasattr(algorithm, 'repair') and algorithm.repair is not None:
        # Calculate avarage z-coordinate across all non-dominated solutions
        if rank_pop[0] == np.inf:
            algorithm.repair.learn_rules(algorithm.problem, x_pop)
        else:
            algorithm.repair.learn_rules(algorithm.problem, x_pop[rank_pop == 0])

    # print(algorithm.problem.z_avg)
    algorithm.problem.percent_rank_0 = x_pop[rank_pop == 0].shape[0] / x_pop.shape[0]

    # TODO: Add max gen to hdf file
    if (algorithm.n_gen != 1) and (algorithm.n_gen % 10) != 0 and (algorithm.n_gen != algorithm.termination.n_max_gen):
        return

    with h5py.File(os.path.join(save_folder, 'optimization_history.hdf5'), 'a') as hf:
        g1 = hf.create_group(f'gen{algorithm.n_gen}')

        g1.create_dataset('X', data=x_pop)
        g1.create_dataset('F', data=f_pop)
        g1.create_dataset('rank', data=rank_pop)
        if algorithm.problem.n_constr > 0:
            g1.create_dataset('G', data=g_pop)
            g1.create_dataset('CV', data=cv_pop)

        num_members = algorithm.pop[0].data['stress'].shape[0]
        num_nodes = algorithm.pop[0].data['coordinates'].shape[0]
        stress_pop = np.zeros([algorithm.pop_size, num_members])
        strain_pop = np.zeros([algorithm.pop_size, num_members])
        u_pop = np.zeros([algorithm.pop_size, num_nodes * 6])
        x0_new_pop = np.zeros([algorithm.pop_size, num_nodes, 3])
        coordinates_pop = np.zeros([algorithm.pop_size, num_nodes, 3])
        connectivity_pop = np.zeros([algorithm.pop_size, num_members, 3])
        for pop_indx in range(algorithm.pop_size):
            stress_pop[pop_indx, :] = algorithm.pop[pop_indx].data['stress'].reshape(1, -1)
            strain_pop[pop_indx, :] = algorithm.pop[pop_indx].data['strain'].reshape(1, -1)
            u_pop[pop_indx, :] = algorithm.pop[pop_indx].data['u'].reshape(1, -1)
            x0_new_pop[pop_indx, :, :] = algorithm.pop[pop_indx].data['x0_new']
            coordinates_pop[pop_indx, :] = algorithm.pop[pop_indx].data['coordinates']
            connectivity_pop[pop_indx, :] = algorithm.pop[pop_indx].data['connectivity']

        g1.create_dataset('stress', data=stress_pop)
        g1.create_dataset('strain', data=strain_pop)
        g1.create_dataset('u', data=u_pop)
        g1.create_dataset('x0_new', data=x0_new_pop)
        g1.create_dataset('coordinates', data=coordinates_pop)
        g1.create_dataset('connectivity', data=connectivity_pop)
        g1.create_dataset('z_avg', data=algorithm.problem.z_avg)
        g1.create_dataset('z_ref', data=algorithm.problem.z_ref)
        g1.create_dataset('r_avg', data=algorithm.problem.r_avg)
        g1.create_dataset('r_ref', data=algorithm.problem.r_ref)

        with open(os.path.join(save_folder, 'z_ref'), 'a') as f:
            f.write(f"{algorithm.problem.z_ref}\n")
        with open(os.path.join(save_folder, 'z_avg'), 'a') as f:
            f.write(f"{algorithm.problem.z_avg}\n")
        with open(os.path.join(save_folder, 'r_avg'), 'a') as f:
            f.write(f"{algorithm.problem.r_avg}\n")
        with open(os.path.join(save_folder, 'r_ref'), 'a') as f:
            f.write(f"{algorithm.problem.r_ref}\n")

    # Save results
    np.savetxt(os.path.join(save_folder, 'f_current_gen'), f_pop[rank_pop == 0])
    np.savetxt(os.path.join(save_folder, 'x_current_gen'), x_pop[rank_pop == 0])

    np.savetxt(os.path.join(save_folder, 'f_pop_current_gen'), f_pop)
    np.savetxt(os.path.join(save_folder, 'x_pop_current_gen'), x_pop)

    # x_rank_0 = x_pop[rank_pop == 0]
    # f_rank_0 = f_pop[rank_pop == 0]

    # min_weight_sol_indx = np.where(np.min(f_rank_0[:, 0]))[0]
    # min_compliance_sol_indx = np.where(np.min(f_rank_0[:, 1]))[0]
    #
    # x_min_weight_solution = x_rank_0[min_weight_sol_indx, :]
    # x_min_compliance_solution = x_rank_0[min_compliance_sol_indx, :]
    # f_min_weight_solution = f_rank_0[min_weight_sol_indx, :]
    # f_min_compliance_solution = f_rank_0[min_compliance_sol_indx, :]
    #
    # np.savetxt(os.path.join(save_folder, 'x_extreme_current_gen'),
    #            np.append(x_min_weight_solution, x_min_compliance_solution, axis=0))
    # np.savetxt(os.path.join(save_folder, 'f_extreme_current_gen'),
    #            np.append(f_min_weight_solution, f_min_compliance_solution, axis=0))

    if algorithm.problem.n_constr > 0:
        np.savetxt(os.path.join(save_folder, 'g_current_gen'), g_pop[rank_pop == 0])
        np.savetxt(os.path.join(save_folder, 'cv_current_gen'), cv_pop[rank_pop == 0])
        np.savetxt(os.path.join(save_folder, 'g_pop_current_gen'), g_pop)
        np.savetxt(os.path.join(save_folder, 'cv_pop_current_gen'), cv_pop)

    np.savetxt(os.path.join(save_folder, 'rank_pop_current_gen'), rank_pop)
    np.savetxt(os.path.join(save_folder, 'stress_pop_current_gen'), stress_pop)
    np.savetxt(os.path.join(save_folder, 'strain_pop_current_gen'), strain_pop)
    np.savetxt(os.path.join(save_folder, 'u_pop_current_gen'), u_pop)
    np.save(os.path.join(save_folder, 'x0_new_pop_current_gen'), x0_new_pop)
    pickle.dump(algorithm, open(os.path.join(save_folder, 'pymoo_algorithm_current_gen.pickle'), 'wb'))

    with open(os.path.join(save_folder, 'repair_details_current_gen'), 'w') as f:
        for z in algorithm.problem.z_avg:
            f.write(f"{z} ")
        f.write("\n")
        for s in algorithm.problem.shape_rule_score:
            f.write(f"{s} ")
        f.write("\n")
        for r in algorithm.problem.r_avg:
            f.write(f"{r} ")
        f.write("\n")
        for g_indx, grp in enumerate(algorithm.problem.grouped_size_vars):
            for g in grp:
                f.write(f"{g} ")
            f.write("\n")
            for score in algorithm.problem.size_rule_score[g_indx]:
                f.write(f"{score} ")
            f.write("\n")


def parse_args(args):
    """Defines and parses the command line arguments that can be supplied by the user.

    Args:
        args (list): Command line arguments supplied by the user.

    """
    # Command line args accepted by the program
    parser = argparse.ArgumentParser(description='Large Scale Truss Design Optimization')

    # Truss parameters
    parser.add_argument('--nshapevar', type=int, default=10, help='Random seed')
    parser.add_argument('--symmetric', action='store_true', default=False, help='Enforce symmetricity of trusses')

    # Optimization parameters
    parser.add_argument('--seed', type=int, default=184716924, help='Random seed')
    parser.add_argument('--ngen', type=int, default=200, help='Maximum number of generations')
    parser.add_argument('--popsize', type=int, default=100, help='Population size')

    # Innovization
    parser.add_argument('--repair', action='store_true', default=False, help='Apply custom repair operator')
    parser.add_argument('--momentum', type=float, default=0.3, help='Value of momentum coefficient')

    # Logging parameters
    parser.add_argument('--logging', action='store_true', default=True, help='Enable/disable logging')
    parser.add_argument('--save_folder', type=str, help='Experiment name')

    # Parallelization
    parser.add_argument('--ncores', default=mp.cpu_count() // 4,
                        help='How many cores to use for population members to be evaluated in parallel')

    # Not yet operational
    parser.add_argument('--report-freq', type=float, default=10, help='Default logging frequency in generations')
    parser.add_argument('--crossover', default='real_sbx', help='Choose crossover operator')
    parser.add_argument('--mutation-eta', default=20, help='Define mutation parameter eta')
    parser.add_argument('--mutation-prob', default=0.005, help='Define mutation parameter eta')

    return parser.parse_args(args)


def setup_logging(out_to_console=False, log_file=None):
    # logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if out_to_console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)


if __name__ == '__main__':
    t0 = time.time()

    cmd_args = parse_args(sys.argv[1:])
    # arg_str = f'--ngen 60 --popsize 50 --report-freq 20 --target-thrust baseline --seed 0 --mutation-eta 3 ' \
    #           f'--mutation-prob 0.05 --crossover two_point --save nsga2-test-{time.strftime("%Y%m%d-%H%M%S")}'
    # cmd_args = parse_args(arg_str.split(' '))

    # seed_list = np.loadtxt('random_seed_list', dtype=np.int32)
    # seed = seed_list[0]

    if cmd_args.symmetric:
        truss_problem = TrussProblemSymmetric(num_shape_vars=cmd_args.nshapevar)
    else:
        truss_problem = TrussProblem(num_shape_vars=cmd_args.nshapevar)
    truss_optimizer = NSGA2(
        pop_size=cmd_args.popsize,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=3),
        mutation=get_mutation("real_pm", eta=3),
        eliminate_duplicates=True,
        callback=record_state,
        display=OptimizationDisplay()
    )
    termination = get_termination("n_gen", cmd_args.ngen)

    save_folder = os.path.join('output', f'truss_nsga2_parallel_seed{cmd_args.seed}_{time.strftime("%Y%m%d-%H%M%S")}')

    if cmd_args.repair:
        if truss_optimizer.pop_size < 50:
            warnings.warn("Population size might be too low to learn innovization rules")
            logging.warning("Population size might be too low to learn innovization rules")
        # truss_optimizer.repair = MonotonicityRepairV1()

        truss_optimizer.repair = ParameterlessInequalityRepair(momentum_coefficient=cmd_args.momentum)
        save_folder = os.path.join('output',
                                   f'truss_nsga2_repair_0.8pf_seed{cmd_args.seed}_{time.strftime("%Y%m%d-%H%M%S")}')
        print("======================")
        print("Repair operator active")
        print("======================")

    if cmd_args.save_folder is not None:
        save_folder = os.path.join('output', cmd_args.save_folder)

    os.makedirs(save_folder)

    if cmd_args.logging:
        setup_logging(log_file=os.path.join(save_folder, 'log.txt'))

    logging.info(f"User-supplied arguments: {sys.argv[1:]}")
    logging.info(f"All arguments after parsing: {cmd_args}")
    logging.info(f"Population size = {truss_optimizer.pop_size}, Max. generations = {termination.n_max_gen}")
    logging.info(f"Vars = {truss_problem.n_var}, "
                 f"Objectives = {truss_problem.n_obj}, Constraints = {truss_problem.n_constr}")
    logging.info(f"Range of decision variables:\nX_L=\n{truss_problem.xl}\nX_U=\n{truss_problem.xu}\n")
    logging.info(f"Size variables = {truss_problem.num_size_vars}")
    logging.info(f"Shape variables = {truss_problem.num_shape_vars}")
    logging.info(f"Fixed nodes:\n{truss_problem.fixed_nodes}")
    logging.info(f"Load nodes:\n{truss_problem.load_nodes}")
    logging.info(f"Force:\n{truss_problem.force}")
    if cmd_args.repair:
        logging.info("Members grouped together for repair:\ntruss_problem.grouped_members")
    else:
        logging.info("Repair not active")

    logging.info("Beginning optimization")
    res = minimize(truss_problem,
                   truss_optimizer,
                   termination,
                   seed=cmd_args.seed,
                   save_history=False,
                   verbose=True)

    logging.info("Optimization complete. Writing data")
    print(res.F)

    # Save results
    # For final PF
    np.savetxt(os.path.join(save_folder, 'f_max_gen'), res.F)
    np.savetxt(os.path.join(save_folder, 'x_max_gen'), res.X)
    if truss_problem.n_constr > 0:
        np.savetxt(os.path.join(save_folder, 'g_max_gen'), res.G)
        np.savetxt(os.path.join(save_folder, 'cv_max_gen'), res.CV)

    # For final pop
    np.savetxt(os.path.join(save_folder, 'f_pop_max_gen'), res.pop.get('F'))
    np.savetxt(os.path.join(save_folder, 'x_pop_max_gen'), res.pop.get('X'))
    if truss_problem.n_constr > 0:
        np.savetxt(os.path.join(save_folder, 'g_pop_max_gen'), res.pop.get('G'))
        np.savetxt(os.path.join(save_folder, 'cv_pop_max_gen'), res.pop.get('CV'))
    np.savetxt(os.path.join(save_folder, 'rank_pop_max_gen'), res.pop.get('rank'))

    # Additional data for final pop
    num_members = res.pop[0].data['stress'].shape[0]
    num_nodes = res.pop[0].data['coordinates'].shape[0]
    stress_final_pop = np.zeros([truss_optimizer.pop_size, num_members])
    strain_final_pop = np.zeros([truss_optimizer.pop_size, num_members])
    u_final_pop = np.zeros([truss_optimizer.pop_size, num_nodes * 6])
    x0_new_final_pop = np.zeros([truss_optimizer.pop_size, num_nodes, 3])
    for indx in range(truss_optimizer.pop_size):
        stress_final_pop[indx, :] = res.pop[indx].data['stress'].reshape(1, -1)
        strain_final_pop[indx, :] = res.pop[indx].data['strain'].reshape(1, -1)
        u_final_pop[indx, :] = res.pop[indx].data['u'].reshape(1, -1)
        x0_new_final_pop[indx, :, :] = res.pop[indx].data['x0_new']
    np.savetxt(os.path.join(save_folder, 'stress_pop_max_gen'), stress_final_pop)
    np.savetxt(os.path.join(save_folder, 'strain_pop_max_gen'), strain_final_pop)
    np.savetxt(os.path.join(save_folder, 'u_pop_max_gen'), u_final_pop)
    np.save(os.path.join(save_folder, 'x0_new_pop_max_gen'), x0_new_final_pop)

    # Save pymoo result object
    pickle.dump(res, open(os.path.join(save_folder, 'pymoo_result.pickle'), 'wb'))

    t1 = time.time()
    total_execution_time = t1 - t0
    print(f"Total execution time {total_execution_time}")  # Seconds elapsed
    logging.info(f"Total execution time {total_execution_time}")

    # Plot results
    plt.scatter(res.F[:, 0], res.F[:, 1])
    plt.xlabel("Weight (kg)")
    plt.ylabel("Compliance (m/N)")
    plt.savefig(os.path.join(save_folder, 'pf.png'))
    plt.show()
