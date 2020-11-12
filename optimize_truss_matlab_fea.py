import argparse
import logging
import os
import pickle
import sys
import time
import warnings

import h5py
import matlab
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.display import Display

from truss.innovization.truss_repair import ParameterlessInequalityRepair

matlab_engine = matlab.engine.start_matlab()
save_folder = os.path.join('output', 'truss_optimization_nsga2')


class OptimizationDisplay(Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

        f = algorithm.pop.get("F")
        f_min_weight = np.round(f[f[:, 0] == np.min(f[:, 0]), :], decimals=2).flatten()
        f_min_compliance = np.round(f[f[:, 1] == np.min(f[:, 1]), :], decimals=2).flatten()
        self.output.append("Min. weight solution", f_min_weight)
        self.output.append("Min. compliance solution", f_min_compliance)
        self.output.append("cv(min.)", np.min(algorithm.pop.get('CV')))
        self.output.append("cv(max.)", np.max(algorithm.pop.get('CV')))


class TrussProblem(Problem):

    def __init__(self):
        # Truss parameters
        self.density = 7121.4  # kg/m3
        self.elastic_modulus = 200e9  # Pa
        self.yield_stress = 248.2e6  # Pa
        self.max_allowable_displacement = 0.025  # Max displacements of all nodes in x, y, and z directions
        self.num_shape_vars = 10
        self.num_size_vars = 260

        coordinates_file = 'truss/sample_input/coord_iscso.csv'
        connectivity_file = 'truss/sample_input/connect_iscso.csv'
        fixednodes_file = 'truss/sample_input/fixn_iscso.csv'
        loadn_file = 'truss/sample_input/loadn_iscso.csv'
        force_file = 'truss/sample_input/force_iscso.csv'

        self.coordinates = np.loadtxt(coordinates_file, delimiter=',')
        self.connectivity = np.loadtxt(connectivity_file, delimiter=',')
        self.fixed_nodes = np.loadtxt(fixednodes_file).reshape(-1, 1)
        self.load_nodes = np.loadtxt(loadn_file).reshape(-1, 1)
        self.force = np.loadtxt(force_file, delimiter=',')

        # Innovization parameters (shape)
        self.z_avg = -1e16 * np.ones(self.num_shape_vars)
        self.z_std = -1e16 * np.ones(self.num_shape_vars)
        self.shape_rule_score = np.zeros(self.num_shape_vars - 1)  # A score given to a rule between 0 and 1

        # Innovization parameters (size)
        # Contains indices of member size decision variables divided into groups. For example, members along the x-axis
        # on top of the truss are considered a group
        self.grouped_members = [np.arange(0, 18), np.arange(18, 36), np.arange(36, 54), np.arange(54, 72),
                                ]
        self.r_avg = -1e16 * np.ones(self.num_size_vars)
        self.r_std = -1e16 * np.ones(self.num_size_vars)
        self.size_rule_score = []  # A score given to a rule between 0 and 1
        for grp in self.grouped_members:
            self.size_rule_score.append(np.zeros(len(grp) - 1))
        self.percent_rank_0 = None

        super().__init__(n_var=270,
                         n_obj=2,
                         n_constr=2,
                         xl=np.concatenate((0.005 * np.ones(self.num_size_vars), -25 * np.ones(self.num_shape_vars))),
                         xu=np.concatenate((0.100 * np.ones(self.num_size_vars), 3.5 * np.ones(self.num_shape_vars))))

    def _evaluate(self, x_in, out, *args, **kwargs):
        x = np.copy(x_in)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if hasattr(kwargs['algorithm'], 'innovization') and kwargs['algorithm'].repair is not None:
            x = kwargs['algorithm'].repair.do(self, np.copy(x), **kwargs)

        n = x.shape[0]
        g1 = np.zeros(n)
        g2 = np.zeros(n)
        g3 = np.zeros(n)

        # Create a list of coordinate and connectivity matrices for all population members
        coordinates_list_matlab = [None for _ in range(n)]
        connectivity_list_matlab = [None for _ in range(n)]
        coordinates_array = np.zeros([n, self.coordinates.shape[0], self.coordinates.shape[1]])
        connectivity_array = np.zeros([n, self.connectivity.shape[0], self.connectivity.shape[1]])
        for i in range(n):
            r = x[i, :260]  # Radius of each element
            z = x[i, 260:]  # Z-coordinate of bottom members

            coordinates = np.copy(self.coordinates)
            connectivity = np.copy(self.connectivity)

            connectivity[:, 2] = r
            coordinates[0:10, 2] = z
            coordinates[38:48, 2] = z
            coordinates[10:19, 2] = np.flip(z[:-1])
            coordinates[48:57, 2] = np.flip(z[:-1])

            coordinates_list_matlab[i] = matlab.double(coordinates.tolist())
            connectivity_list_matlab[i] = matlab.double(connectivity.tolist())
            coordinates_array[i] = coordinates
            connectivity_array[i] = connectivity

        weight_pop, compliance_pop, stress_pop, strain_pop, u_pop, x0_new_pop = \
            matlab_engine.run_fea_parallel(coordinates_list_matlab,
                                           connectivity_list_matlab,
                                           matlab.double(self.fixed_nodes.tolist()),
                                           matlab.double(self.load_nodes.tolist()),
                                           matlab.double(self.force.tolist()),
                                           matlab.double([self.density]),
                                           matlab.double([self.elastic_modulus]),
                                           nargout=6)

        for i in range(n):
            g1[i] = matlab_engine.max(matlab_engine.abs(stress_pop[i])) - self.yield_stress
            g2[i] = matlab_engine.max(matlab_engine.abs(u_pop[i])) - self.max_allowable_displacement
            del_coord = np.array(x0_new_pop)[i] - coordinates_array[i]
            if np.max(del_coord[:, 2]) > 0:
                g3[i] = np.max(del_coord[:, 2])
            else:
                g3[i] = -1

        out['stress'] = np.array(stress_pop)
        out['strain'] = np.array(strain_pop)
        out['u'] = np.array(u_pop)
        out['x0_new'] = np.array(x0_new_pop)
        out['coordinates'] = coordinates_array
        out['connectivity'] = connectivity_array

        out['F'] = np.column_stack([np.array(weight_pop), np.array(compliance_pop)])
        # out['F'] = np.column_stack([np.array(weight_pop), np.max(out['stress'], axis=1)])
        if self.n_constr == 2:
            out['G'] = np.column_stack([g1, g2])
        elif self.n_constr == 3:
            out['G'] = np.column_stack([g1, g2, g3])


def record_state(algorithm):
    x_pop = algorithm.pop.get('X')
    f_pop = algorithm.pop.get('F')
    rank_pop = algorithm.pop.get('rank')
    g_pop = algorithm.pop.get('G')
    cv_pop = algorithm.pop.get('CV')
    # algorithm.problem.z_monotonicity_matrix = get_monotonicity_pattern(x_pop, f_pop, rank_pop)

    if hasattr(algorithm, 'innovization') and algorithm.repair is not None:
        # Calculate avarage z-coordinate across all non-dominated solutions
        # algorithm.problem.z_avg = np.average(x_pop[rank_pop == 0][:, -10:], axis=0)
        # algorithm.problem.z_std = np.std(x_pop[rank_pop == 0][:, -10:], axis=0)
        algorithm.repair.learn_rules(algorithm.problem, x_pop[rank_pop == 0])

    algorithm.problem.percent_rank_0 = x_pop[rank_pop == 0].shape[0] / x_pop.shape[0]

    # TODO: Add max gen to hdf file
    if (algorithm.n_gen != 1) and (algorithm.n_gen % 10) != 0 and (algorithm.n_gen != algorithm.termination.n_max_gen):
        return

    with h5py.File(os.path.join(save_folder, 'optimization_history.hdf5'), 'a') as hf:
        # if algorithm.n_gen == 1:
        #     hf.create_dataset('max_gen', data=algorithm.max_gen)
        g1 = hf.create_group(f'gen{algorithm.n_gen}')

        # for p in range(algorithm.pop_size):
        # g2 = g1.create_group(f'sol{p + 1}')
        g1.create_dataset('X', data=x_pop)
        g1.create_dataset('F', data=f_pop)
        g1.create_dataset('rank', data=rank_pop)
        g1.create_dataset('G', data=g_pop)
        g1.create_dataset('CV', data=cv_pop)

        num_members = algorithm.pop[0].data['stress'].shape[0]
        num_nodes = algorithm.pop[0].data['coordinates'].shape[0]
        stress_pop = np.zeros([algorithm.pop_size, num_members])
        strain_pop = np.zeros([algorithm.pop_size, num_members])
        u_pop = np.zeros([algorithm.pop_size, num_nodes * 6])
        x0_new_pop = np.zeros([algorithm.pop_size, num_nodes, 3])
        for indx in range(algorithm.pop_size):
            stress_pop[indx, :] = algorithm.pop[indx].data['stress'].reshape(1, -1)
            strain_pop[indx, :] = algorithm.pop[indx].data['strain'].reshape(1, -1)
            u_pop[indx, :] = algorithm.pop[indx].data['u'].reshape(1, -1)
            x0_new_pop[indx, :, :] = algorithm.pop[indx].data['x0_new']

        g1.create_dataset('stress', data=stress_pop)
        g1.create_dataset('strain', data=strain_pop)
        g1.create_dataset('u', data=u_pop)
        g1.create_dataset('x0_new', data=x0_new_pop)

        # Save results
        np.savetxt(os.path.join(save_folder, 'f_current_gen'), f_pop[rank_pop == 0])
        np.savetxt(os.path.join(save_folder, 'x_current_gen'), x_pop[rank_pop == 0])
        # np.savetxt(os.path.join(save_folder, 'g_current_gen'), g_pop[rank_pop == 0])
        # np.savetxt(os.path.join(save_folder, 'cv_current_gen'), cv_pop[rank_pop == 0])
        np.savetxt(os.path.join(save_folder, 'f_pop_current_gen'), f_pop)
        np.savetxt(os.path.join(save_folder, 'x_pop_current_gen'), x_pop)
        np.savetxt(os.path.join(save_folder, 'g_pop_current_gen'), g_pop)
        np.savetxt(os.path.join(save_folder, 'cv_pop_current_gen'), cv_pop)
        np.savetxt(os.path.join(save_folder, 'rank_pop_current_gen'), rank_pop)
        np.savetxt(os.path.join(save_folder, 'stress_pop_current_gen'), stress_pop)
        np.savetxt(os.path.join(save_folder, 'strain_pop_current_gen'), strain_pop)
        np.savetxt(os.path.join(save_folder, 'u_pop_current_gen'), u_pop)
        np.save(os.path.join(save_folder, 'x0_new_pop_current_gen'), x0_new_pop)
        pickle.dump(algorithm, open('pymoo_algorithm_current_gen.pickle', 'wb'))


def parse_args(args):
    """Defines and parses the command line arguments that can be supplied by the user.

    Args:
        args (dict): Command line arguments supplied by the user.

    """
    # Command line args accepted by the program
    parser = argparse.ArgumentParser(description='Large Scale Truss Design Optimization')

    # Optimization parameters
    parser.add_argument('--seed', type=int, default=184716924, help='Random seed')
    parser.add_argument('--ngen', type=int, default=200, help='Maximum number of generations')
    parser.add_argument('--popsize', type=int, default=100, help='Population size')
    parser.add_argument('--innovization', action='store_true', default=False, help='Apply custom innovization operator')

    # Logging parameters
    parser.add_argument('--logging', action='store_true', default=True, help='Enable/disable logging')
    parser.add_argument('--save_folder', type=str, help='Experiment name')

    # Not yet operational
    parser.add_argument('--report-freq', type=float, default=10, help='Default logging frequency in generations')
    parser.add_argument('--crossover', default='real_sbx', help='Choose crossover operator')
    parser.add_argument('--mutation-eta', default=20, help='Define mutation parameter eta')
    parser.add_argument('--mutation-prob', default=0.005, help='Define mutation parameter eta')
    parser.add_argument('--ncores', default=None, help='How many cores to use for parallel evaluator execution')

    return parser.parse_args(args)


def setup_logging(log_file=None):
    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)


if __name__ == '__main__':
    t0 = time.time()
    matlab_engine.parpool(2)

    cmd_args = parse_args(sys.argv[1:])
    # arg_str = f'--ngen 60 --popsize 50 --report-freq 20 --target-thrust baseline --seed 0 --mutation-eta 3 ' \
    #           f'--mutation-prob 0.05 --crossover two_point --save nsga2-test-{time.strftime("%Y%m%d-%H%M%S")}'
    # cmd_args = parse_args(arg_str.split(' '))

    # seed_list = np.loadtxt('random_seed_list', dtype=np.int32)
    # seed = seed_list[0]

    truss_problem = TrussProblem()
    truss_optimizer = NSGA2(
        pop_size=cmd_args.popsize,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=30),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
        callback=record_state,
        display=OptimizationDisplay()
    )

    save_folder = os.path.join('output', f'truss_nsga2_seed{cmd_args.seed}_{time.strftime("%Y%m%d-%H%M%S")}')

    if cmd_args.repair:
        if truss_optimizer.pop_size < 50:
            warnings.warn("Population size might be too low to learn innovization rules")
            logging.warning("Population size might be too low to learn innovization rules")
        # truss_optimizer.innovization = MonotonicityRepairV1()
        truss_optimizer.repair = ParameterlessInequalityRepair()
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

    termination = get_termination("n_gen", cmd_args.ngen)

    res = minimize(truss_problem,
                   truss_optimizer,
                   termination,
                   seed=cmd_args.seed,
                   save_history=False,
                   verbose=True)

    print(res.F)

    matlab_engine.quit()

    # Save results
    # For final PF
    np.savetxt(os.path.join(save_folder, 'f_max_gen'), res.F)
    np.savetxt(os.path.join(save_folder, 'x_max_gen'), res.X)
    np.savetxt(os.path.join(save_folder, 'g_max_gen'), res.G)
    np.savetxt(os.path.join(save_folder, 'cv_max_gen'), res.CV)

    # For final pop
    np.savetxt(os.path.join(save_folder, 'f_pop_max_gen'), res.pop.get('F'))
    np.savetxt(os.path.join(save_folder, 'x_pop_max_gen'), res.pop.get('X'))
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
    pickle.dump(res, open('pymoo_result.pickle', 'wb'))

    t1 = time.time()
    print("Total execution time: ", t1 - t0)  # Seconds elapsed

    # Plot results
    plt.scatter(res.F[:, 0], res.F[:, 1])
    plt.xlabel("Weight (kg)")
    plt.ylabel("Compliance (m/N)")
    plt.show()
