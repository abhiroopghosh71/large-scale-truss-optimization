import logging
import multiprocessing as mp

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import NoRepair

from truss.fea.run_fea import run_fea
from truss.generate_truss import gen_truss

logger = logging.getLogger(__name__)

class TrussProblem(Problem):

    def __init__(self, num_shape_vars=10, n_cores=mp.cpu_count() // 4):
        # Truss material properties
        self.density = 7121.4  # kg/m3
        self.elastic_modulus = 200e9  # Pa
        self.yield_stress = 248.2e6  # Pa

        # Constraints
        self.max_allowable_displacement = 0.025  # Max displacements of all nodes in x, y, and z directions
        self.num_shape_vars = num_shape_vars

        # coordinates_file = 'truss/sample_input/coord_iscso.csv'
        # connectivity_file = 'truss/sample_input/connect_iscso.csv'
        # fixednodes_file = 'truss/sample_input/fixn_iscso.csv'
        # loadn_file = 'truss/sample_input/loadn_iscso.csv'
        # force_file = 'truss/sample_input/force_iscso.csv'

        # self.coordinates = np.loadtxt(coordinates_file, delimiter=',')
        # self.connectivity = np.loadtxt(connectivity_file, delimiter=',')
        # self.fixed_nodes = np.loadtxt(fixednodes_file).reshape(-1, 1)
        # self.load_nodes = np.loadtxt(loadn_file).reshape(-1, 1)
        # self.load_nodes = np.array(np.append(np.arange(2, 19), np.arange(40, 57))).reshape(-1, 1)
        # # self.load_nodes = np.array([[10], [48]])
        # warnings.warn(f"Load nodes changed by user!! {self.load_nodes}")
        # self.force = np.loadtxt(force_file, delimiter=',')
        self.force = np.array([0, 0, -5000])

        self.coordinates, self.connectivity, self.fixed_nodes, self.load_nodes, self.member_groups\
            = gen_truss(n_shape_nodes=2*self.num_shape_vars - 1)
        self.num_size_vars = self.connectivity.shape[0]
        self.fixed_nodes = self.fixed_nodes.reshape(-1, 1)
        self.load_nodes = self.load_nodes.reshape(-1, 1)

        print(f"No. of shape vars = {self.num_shape_vars}")
        print(self.force)

        # Innovization parameters (general)
        self.percent_rank_0 = None

        # Innovization parameters (shape)
        self.z_avg = np.nan * np.ones(self.num_shape_vars)
        self.z_std = np.nan * np.ones(self.num_shape_vars)
        self.shape_rule_score = np.zeros(self.num_shape_vars - 1)  # A score given to a rule between 0 and 1
        self.z_ref = np.nan * np.ones(self.num_shape_vars)  # The reference shape to use for innovization
        self.z_ref_history = []

        # Innovization parameters (size)
        # TODO: Replace with groups returned from gen_truss
        self.grouped_size_vars = [np.arange(0, 2 * self.num_shape_vars - 2),
                                  np.arange(2*self.num_shape_vars - 2, 4*self.num_shape_vars - 4),
                                  np.arange(4*self.num_shape_vars - 4, 6*self.num_shape_vars - 6),
                                  np.arange(6*self.num_shape_vars - 6, 8*self.num_shape_vars - 8),
                                  ]
        self.r_avg = -1e16 * np.ones(self.num_size_vars)
        self.r_std = -1e16 * np.ones(self.num_size_vars)
        self.size_rule_score = []  # A score given to a rule between 0 and 1
        for grp in self.grouped_size_vars:
            self.size_rule_score.append(np.zeros(len(grp) - 1))
        self.percent_rank_0 = None

        # Parallelization
        self.n_cores = n_cores
        if n_cores > mp.cpu_count():
            self.n_cores = mp.cpu_count()

            # TODO: Make n_constr a user parameter
        super().__init__(n_var=self.num_shape_vars + self.num_size_vars,
                         n_obj=2,
                         n_constr=2,
                         xl=np.concatenate((0.005 * np.ones(self.num_size_vars), -25 * np.ones(self.num_shape_vars))),
                         xu=np.concatenate((0.100 * np.ones(self.num_size_vars), 3.5 * np.ones(self.num_shape_vars))))

        print(f"Number of constraints = {self.n_constr}")

    @staticmethod
    def calc_obj(i, x, coordinates, connectivity, fixed_nodes, load_nodes, force, density, elastic_modulus,
                 yield_stress, max_allowable_displacement, num_shape_vars, structure_type='truss'):
        r = np.copy(x[:-num_shape_vars])  # Radius of each element
        z = np.copy(x[-num_shape_vars:])  # Z-coordinate of bottom members

        connectivity[:, 2] = r
        # coordinates[0:10, 2] = z
        # coordinates[38:48, 2] = z
        # coordinates[10:19, 2] = np.flip(z[:-1])
        # coordinates[48:57, 2] = np.flip(z[:-1])
        coordinates[0:num_shape_vars, 2] = z
        coordinates[(2*num_shape_vars - 1) * 2:(2*num_shape_vars - 1) * 2 + num_shape_vars, 2] = z
        coordinates[num_shape_vars:2*num_shape_vars - 1, 2] = np.flip(z[:-1])
        coordinates[(2*num_shape_vars - 1) * 2 + num_shape_vars:(2*num_shape_vars - 1) * 2 + 2*num_shape_vars - 1, 2] = np.flip(z[:-1])

        weight, compliance, stress, strain, u, x0_new = run_fea(np.copy(coordinates),
                                                                np.copy(connectivity), fixed_nodes,
                                                                load_nodes, force, density,
                                                                elastic_modulus, structure_type=structure_type)
        del_coord = np.array(x0_new) - coordinates

        f = np.array([weight, compliance])
        # f = np.array([weight, np.max(np.abs(u))])
        # f2 = np.max(np.abs(del_x[:, 2]))

        # Allow displacement only in -z direction
        if np.max(del_coord[:, 2]) > 0:
            g3 = np.max(del_coord[:, 2])
        else:
            g3 = -1
        g = np.array([np.max(np.abs(stress)) - yield_stress, np.max(np.abs(u)) - max_allowable_displacement, g3])

        return i, f, g, stress, strain, u, x0_new, coordinates, connectivity

    def _evaluate(self, x_in, out, *args, **kwargs):
        # TODO: Parallelize obj eval
        x = np.copy(x_in)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if hasattr(kwargs['algorithm'], 'innovization') and kwargs['algorithm'].repair is not None and type(kwargs['algorithm'].repair) != NoRepair:
            x = kwargs['algorithm'].repair.do(self, np.copy(x), **kwargs)

        pool = mp.Pool(self.n_cores)
        logging.debug(f"Multiprocessing pool opened. CPU count = {mp.cpu_count()}, Pool Size = {self.n_cores}")

        # Call apply_async() for asynchronous evaluation of each population member
        result_objects = [pool.apply_async(TrussProblem.calc_obj, args=(i, row, np.copy(self.coordinates),
                                                                        np.copy(self.connectivity), self.fixed_nodes,
                                                                        self.load_nodes, self.force,
                                                                        self.density, self.elastic_modulus,
                                                                        self.yield_stress,
                                                                        self.max_allowable_displacement,
                                                                        self.num_shape_vars,
                                                                        'truss'))
                          for i, row in enumerate(x)]

        pool.close()  # Need to close the pool to prevent spawning too many processes
        pool.join()
        logging.debug("Parallel objective evaluation complete. Pool closed.")

        # Result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]

        # apply_async() might return results in a different order
        results.sort(key=lambda r: r[0])

        out['F'] = np.array([[r[1][0], r[1][1]] for r in results])
        out['G'] = np.array([[r[2][c] for c in range(self.n_constr)] for r in results])

        out['stress'] = np.array([r[3] for r in results])
        out['strain'] = np.array([r[4] for r in results])
        out['u'] = np.array([r[5] for r in results])
        out['x0_new'] = np.array([r[6] for r in results])
        out['coordinates'] = np.array([r[7] for r in results])
        out['connectivity'] = np.array([r[8] for r in results])