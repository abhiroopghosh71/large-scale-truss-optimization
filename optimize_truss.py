from pymoo.model.problem import Problem
import numpy as np
import matlab
import matlab.engine
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
import os
import time
import argparse

from truss_repair import MonotonicityRepair
from obj_eval import calc_obj


matlab_engine = matlab.engine.start_matlab()
matlab_engine.parpool()
save_file = os.path.join('output', 'truss_optimization_nsga2')


class TrussProblem(Problem):
    def __init__(self):
        self.density = 7121.4  # kg/m3
        self.elastic_modulus = 200e9  # Pa
        self.yield_stress = 248.2e6  # Pa
        self.max_allowable_displacement = 0.025  # Max displacements of all nodes in x, y, and z directions

        coordinates_file = 'truss/sample_input/coord_iscso.csv'
        connectivity_file = 'truss/sample_input/connect_iscso.csv'
        fixednodes_file = 'truss/sample_input/fixn_iscso.csv'
        loadn_file = 'truss/sample_input/loadn_iscso.csv'
        force_file = 'truss/sample_input/force_iscso.csv'

        self.coordinates = np.loadtxt(coordinates_file, delimiter=',')
        self.connectivity = np.loadtxt(connectivity_file, delimiter=',')
        self.fixed_nodes = np.loadtxt(fixednodes_file).reshape(-1, 1)
        self.load_nodes = np.loadtxt(loadn_file).reshape(-1, 1)
        self.force = np.loadtxt(force_file, delimiter=',').reshape(-1, 1)

        # For every pair of z-coordinates z1 and z2, store the number of good solutions where z1 >, < or = z2
        self.z_monotonicity_matrix = np.zeros([3, 10, 10])
        self.z_avg = np.zeros(10)
        self.percent_rank_0 = None

        # self.matlab_engine = matlab.engine.start_matlab()

        super().__init__(n_var=270,
                         n_obj=2,
                         n_constr=2,
                         xl=np.concatenate((0.005 * np.ones(260), -25 * np.ones(10))),
                         xu=np.concatenate((0.100 * np.ones(260), 3.5 * np.ones(10))))

    def _evaluate(self, x_in, out, *args, **kwargs):
        x = np.copy(x_in)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n = x.shape[0]
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        g1 = np.zeros(n)
        g2 = np.zeros(n)

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
            f1[i] = weight_pop[i][0]
            f2[i] = compliance_pop[i][0]
            g1[i] = matlab_engine.max(matlab_engine.abs(stress_pop[i])) - self.yield_stress
            g2[i] = matlab_engine.max(matlab_engine.abs(u_pop[i])) - self.max_allowable_displacement
            kwargs['individuals'][i].data['stress'] = np.array(stress_pop[i]).flatten()
            kwargs['individuals'][i].data['strain'] = np.array(strain_pop[i]).flatten()
            kwargs['individuals'][i].data['u'] = np.array(u_pop[i]).flatten()
            kwargs['individuals'][i].data['x0_new'] = np.array(x0_new_pop[i])
            kwargs['individuals'][i].data['coordinates'] = coordinates_array[i]
            kwargs['individuals'][i].data['connectivity'] = connectivity_array[i]

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


def get_monotonicity_pattern(x, f, rank):
    monotonicity_matrix = np.zeros([3, 10, 10])
    x_non_dominated = x[rank == 0]

    return 0


def record_state(algorithm):
    x_pop = algorithm.pop.get('X')
    f_pop = algorithm.pop.get('F')
    rank_pop = algorithm.pop.get('rank')
    # algorithm.problem.z_monotonicity_matrix = get_monotonicity_pattern(x_pop, f_pop, rank_pop)

    # Calculate avarage z-coordinate across all non-dominated solutions
    algorithm.problem.z_avg = np.average(x_pop[rank_pop == 0][:, -10:], axis=0)
    algorithm.problem.percent_rank_0 = x_pop[rank_pop == 0].shape[0] / x_pop.shape[0]
    # TODO: Add max gen to hdf file
    if (algorithm.n_gen != 1) and (algorithm.n_gen % 10) != 0:
        return

    with h5py.File(os.path.join(save_file, 'optimization_history.hdf5'), 'a') as hf:
        g1 = hf.create_group(f'gen{algorithm.n_gen}')

        # for p in range(algorithm.pop_size):
        # g2 = g1.create_group(f'sol{p + 1}')
        g1.create_dataset('X', data=x_pop)
        g1.create_dataset('F', data=f_pop)

        num_members = algorithm.pop[0].data['stress'].shape[0]
        num_nodes = algorithm.pop[0].data['coordinates'].shape[0]
        stress = np.zeros([algorithm.pop_size, num_members])
        strain = np.zeros([algorithm.pop_size, num_members])
        u = np.zeros([algorithm.pop_size, num_nodes * 6])
        x0_new = np.zeros([algorithm.pop_size, num_nodes, 3])
        for indx in range(algorithm.pop_size):
            stress[indx, :] = algorithm.pop[indx].data['stress']
            strain[indx, :] = algorithm.pop[indx].data['strain']
            u[indx, :] = algorithm.pop[indx].data['u']
            x0_new[indx, :] = algorithm.pop[indx].data['x0_new']

        g1.create_dataset('stress', data=stress)
        g1.create_dataset('strain', data=strain)


def parse_args(args):
    """Defines and parses the command line arguments that can be supplied by the user.

    Args:
        args (dict): Command line arguments supplied by the user.

    """
    # Command line args accepted by the program
    parser = argparse.ArgumentParser(description='Large Scale Truss Design Optimization')
    parser.add_argument('--seed', type=int, default=184716924, help='Random seed')
    parser.add_argument('--ngen', type=int, default=200, help='Maximum number of generations')
    parser.add_argument('--popsize', type=int, default=100, help='Population size')
    parser.add_argument('--report-freq', type=float, default=10, help='Default logging frequency in generations')
    parser.add_argument('--repair', action='store_true', default=False, help='Apply custom repair operator')
    parser.add_argument('--save', type=str, help='Experiment name')
    parser.add_argument('--crossover', default='real_sbx', help='Choose crossover operator')
    parser.add_argument('--mutation-eta', default=20, help='Define mutation parameter eta')
    parser.add_argument('--mutation-prob', default=0.005, help='Define mutation parameter eta')
    parser.add_argument('--ncores', default=None, help='How many cores to use for parallel evaluator execution')

    return parser.parse_args(args)


if __name__ == '__main__':
    t0 = time.time()

    seed_list = np.loadtxt('random_seed_list', dtype=np.int32)
    seed = seed_list[0]

    problem = TrussProblem()
    truss_optimizer = NSGA2(
        pop_size=500,
        # n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=30),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
        callback=record_state
    )
    save_file = os.path.join('output', f'truss_nsga2_seed{seed}_{time.strftime("%Y%m%d-%H%M%S")}')

    # truss_optimizer.repair = MonotonicityRepair()
    if truss_optimizer.repair is not None:
        save_file = os.path.join('output', f'truss_nsga2_repair_0.8pf_seed{seed}_{time.strftime("%Y%m%d-%H%M%S")}')
        print("======================")
        print("Repair operator active")
        print("======================")

    os.makedirs(save_file)
    termination = get_termination("n_gen", 2000)

    res = minimize(problem,
                   truss_optimizer,
                   termination,
                   seed=seed,
                   save_history=False,
                   verbose=True)

    print(res.F)

    matlab_engine.quit()

    np.savetxt(os.path.join(save_file, 'f_max_gen'), res.F)
    np.savetxt(os.path.join(save_file, 'x_max_gen'), res.X)

    t1 = time.time()
    print("Total execution time: ", t1 - t0)  # Seconds elapsed

    # Plot results
    plt.scatter(res.F[:, 0], res.F[:, 1])
    plt.xlabel("Weight (kg)")
    plt.ylabel("Compliance (m/N)")
    plt.show()

    # pool.close()
