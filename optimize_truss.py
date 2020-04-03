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

from truss_repair import MonotonicityRepair
from obj_eval import calc_obj


matlab_engine = matlab.engine.start_matlab()
save_file = os.path.join('output', 'truss_optimization_nsga2')

# pool = mp.Pool(processes=4)
# pool = mp.Pool(mp.cpu_count())


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

    # def _evaluate(self, x, out, *args, **kwargs):
    #     n = x.shape[0]
    #     # f1 = np.zeros(n)
    #     # f2 = np.zeros(n)
    #
    #     coordinates = np.copy(self.coordinates)
    #     connectivity = np.copy(self.connectivity)
    #     fixed_nodes = np.copy(self.fixed_nodes)
    #     load_nodes = np.copy(self.load_nodes)
    #     force = np.copy(self.force)
    #
    #     pool = mp.Pool(mp.cpu_count())
    #
    #     results = np.array([pool.apply(calc_obj,
    #                                    args=(x_indiv, connectivity, coordinates, fixed_nodes, load_nodes, force,
    #                                          self.density, self.elastic_modulus, matlab_engine)) for x_indiv in x])
    #     # results = pool.map(calc_obj, [for x_indiv in x])
    #
    #     # Step 3: Don't forget to close
    #     pool.close()
    #
    #     out["F"] = np.copy(results)
    #     # out["G"] = anp.column_stack([g1, g2])

    def _evaluate(self, x_in, out, *args, **kwargs):
        x = np.copy(x_in)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n = x.shape[0]
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        g1 = np.zeros(n)
        g2 = np.zeros(n)
        for i in range(n):
            r = x[i, :260]  # Radius of each element
            z = x[i, 260:]  # Z-coordinate of bottom members

            coordinates = np.copy(self.coordinates)
            connectivity = np.copy(self.connectivity)
            fixed_nodes = np.copy(self.fixed_nodes)
            load_nodes = np.copy(self.load_nodes)
            force = np.copy(self.force)

            connectivity[:, 2] = r
            coordinates[0:10, 2] = z
            coordinates[38:48, 2] = z
            coordinates[10:19, 2] = np.flip(z[:-1])
            coordinates[48:57, 2] = np.flip(z[:-1])

            weight, compliance, stress, strain, u, x0_new =\
                matlab_engine.run_fea(matlab.double(coordinates.tolist()),
                                      matlab.double(connectivity.tolist()),
                                      matlab.double(fixed_nodes.tolist()),
                                      matlab.double(load_nodes.tolist()),
                                      matlab.double(force.tolist()),
                                      matlab.double([self.density]),
                                      matlab.double([self.elastic_modulus]),
                                      nargout=6)

            f1[i] = weight
            f2[i] = compliance
            g1[i] = matlab_engine.max(matlab_engine.abs(stress)) - self.yield_stress
            g2[i] = matlab_engine.max(matlab_engine.abs(u)) - self.max_allowable_displacement
            kwargs['individuals'][i].data['stress'] = np.array(stress).flatten()
            kwargs['individuals'][i].data['strain'] = np.array(strain).flatten()

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
        stress = np.zeros([algorithm.pop_size, num_members])
        strain = np.zeros([algorithm.pop_size, num_members])
        for indx in range(algorithm.pop_size):
            stress[indx, :] = algorithm.pop[indx].data['stress']
            strain[indx, :] = algorithm.pop[indx].data['strain']

        g1.create_dataset('stress', data=stress)
        g1.create_dataset('strain', data=strain)


if __name__ == '__main__':
    seed_list = np.loadtxt('random_seed_list', dtype=np.int32)
    seed = seed_list[0]

    problem = TrussProblem()
    truss_optimizer = NSGA2(
        pop_size=10,
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
    termination = get_termination("n_gen", 20)

    res = minimize(problem,
                   truss_optimizer,
                   termination,
                   seed=seed,
                   save_history=False,
                   verbose=True)

    print(res.F)

    # Plot results
    plt.scatter(res.F[:, 0], res.F[:, 1])
    plt.xlabel("Weight (kg)")
    plt.ylabel("Compliance (m/N)")
    plt.show()

    # pool.close()
