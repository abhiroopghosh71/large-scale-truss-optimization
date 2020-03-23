from pymoo.model.problem import Problem
import numpy as np
import matlab
import matlab.engine
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle
import h5py
import os
import time

from obj_eval import calc_obj


matlab_engine = matlab.engine.start_matlab()
save_file = os.path.join('output', 'truss_optimization_nsga2')

# pool = mp.Pool(processes=4)


class TrussProblem(Problem):
    def __init__(self):
        self.density = 7121.4  # kg/m3
        self.elastic_modulus = 200e9  # Pa
        self.yield_stress = 248.2e6  # Pa

        coordinates_file = 'truss/sample_input/coord_iscso.csv'
        connectivity_file = 'truss/sample_input/connect_iscso.csv'
        fixednodes_file = 'truss/sample_input/fixn_iscso.csv'
        loadn_file = 'truss/sample_input/loadn_iscso.csv'
        force_file = 'truss/sample_input/force_iscso.csv'

        # coordinates = matlab.double(np.loadtxt(coordinates_file, delimiter=',').tolist())
        # connectivity = matlab.double(np.loadtxt(connectivity_file, delimiter=',').tolist())
        # fixednodes = matlab.double(np.loadtxt(fixednodes_file, delimiter=',').reshape(-1, 1).tolist())
        # loadn = matlab.double(np.loadtxt(loadn_file, delimiter=',').reshape(-1, 1).tolist())
        # force = matlab.double(np.loadtxt(force_file, delimiter=',').reshape(-1, 1).tolist())

        self.coordinates = np.loadtxt(coordinates_file, delimiter=',')
        self.connectivity = np.loadtxt(connectivity_file, delimiter=',')
        self.fixed_nodes = np.loadtxt(fixednodes_file, delimiter=',').reshape(-1, 1)
        self.load_nodes = np.loadtxt(loadn_file, delimiter=',').reshape(-1, 1)
        self.force = np.loadtxt(force_file, delimiter=',').reshape(-1, 1)

        # self.matlab_engine = matlab.engine.start_matlab()

        super().__init__(n_var=197,
                         n_obj=2,
                         n_constr=1,
                         xl=np.concatenate((0.005 * np.ones(187), -25 * np.ones(10))),
                         xu=np.concatenate((0.066 * np.ones(187), 3.5 * np.ones(10))))

    def _evaluate(self, x_in, out, *args, **kwargs):
        x = np.copy(x_in)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n = x.shape[0]
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        g = np.zeros(n)
        for i in range(n):
            r = x[i, :187]  # Radius of each element
            z = x[i, 187:]  # Z-coordinate of bottom members

            coordinates = np.copy(self.coordinates)
            connectivity = np.copy(self.connectivity)
            fixed_nodes = np.copy(self.fixed_nodes)
            load_nodes = np.copy(self.load_nodes)
            force = np.copy(self.force)

            # connectivity[:, 2] = r
            # Horizontal elements in x-axis
            connectivity[:18, 2] = np.copy(r[:18])
            connectivity[36:54, 2] = np.copy(connectivity[:18, 2])
            connectivity[18:36, 2] = np.copy(r[18:36])
            connectivity[54:72, 2] = np.copy(connectivity[18:36, 2])

            # Cross elements at the end
            connectivity[72:76, 2] = np.copy(r[36:40])

            # For elements in y and z axis
            connectivity[76:114, 2] = np.copy(r[40:78])
            for e in range(76, 114):
                # Repeat connectivity in y-axis elements
                if (e % 2) == 0:
                    connectivity[114 + (e-76), 2] = connectivity[e, 2]
                # Repeat connectivity in y-axis elements
                else:
                    connectivity[114 + (e-76), 2] = r[int(78 + (e-76-1) / 2)]

            # Bottom and top cross elements
            connectivity[152:224, 2] = r[97:169]

            # y-axis sloped elements
            connectivity[224:242, 2] = r[169:187]
            connectivity[242:260, 2] = np.copy(connectivity[224:242, 2])

            coordinates[0:10, 2] = np.copy(z)
            coordinates[38:48, 2] = np.copy(z)
            coordinates[10:19, 2] = np.flip(z[:-1])
            coordinates[48:57, 2] = np.flip(z[:-1])

            # density = matlab.double([self.density])
            # elastic_modulus = matlab.double([self.elastic_modulus])
            weight, compliance, stress, strain = matlab_engine.run_fea(matlab.double(coordinates.tolist()),
                                                                       matlab.double(connectivity.tolist()),
                                                                       matlab.double(fixed_nodes.tolist()),
                                                                       matlab.double(load_nodes.tolist()),
                                                                       matlab.double(force.tolist()),
                                                                       matlab.double([self.density]),
                                                                       matlab.double([self.elastic_modulus]),
                                                                       nargout=4)

            f1[i] = weight
            f2[i] = compliance
            g[i] = matlab_engine.max(matlab_engine.abs(stress)) - self.yield_stress
            kwargs['individuals'][i].data['stress'] = np.array(stress).flatten()
            kwargs['individuals'][i].data['strain'] = np.array(strain).flatten()

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.copy(g)


def record_state(algorithm):
    # TODO: Add max gen to hdf file
    if (algorithm.n_gen != 1) and (algorithm.n_gen % 10) != 0:
        return
    with h5py.File(os.path.join(save_file, 'optimization_history.hdf5'), 'a') as hf:
        g1 = hf.create_group(f'gen{algorithm.n_gen}')

        X = algorithm.pop.get('X')
        F = algorithm.pop.get('F')
        # for p in range(algorithm.pop_size):
        # g2 = g1.create_group(f'sol{p + 1}')
        g1.create_dataset('X', data=X)
        g1.create_dataset('F', data=F)

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
    save_file = os.path.join('output', f'truss_symmetric_nsga2_seed{seed}_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(save_file)
    problem = TrussProblem()
    algorithm = NSGA2(
        pop_size=500,
        # n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=30),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
        callback=record_state
    )

    termination = get_termination("n_gen", 2000)
    res = minimize(problem,
                   algorithm,
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

    # array([[6.76379445e+04, 2.72801086e+00],
    #        [6.47451475e+04, 3.00652541e+00],
    #        [6.62875302e+04, 2.78385282e+00],
    #        [6.59075983e+04, 2.79265365e+00]])
