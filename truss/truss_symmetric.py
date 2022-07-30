import logging
import multiprocessing as mp

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import NoRepair

from truss.fea.run_fea import run_fea
from truss.generate_truss import gen_truss

logger = logging.getLogger(__name__)


class TrussProblemSymmetric(Problem):

    def __init__(self, num_shape_vars=10, n_cores=mp.cpu_count() // 4):
        # Truss parameters
        self.density = 7121.4  # kg/m3
        self.elastic_modulus = 200e9  # Pa
        self.yield_stress = 248.2e6  # Pa
        self.max_allowable_displacement = 0.025  # Max displacements of all nodes in x, y, and z directions
        self.num_shape_vars = num_shape_vars

        self.force = np.array([0, 0, -5000])

        self.coordinates, self.connectivity, self.fixed_nodes, self.load_nodes, self.member_groups\
            = gen_truss(n_shape_nodes=2*self.num_shape_vars - 1)
        self.num_members = self.connectivity.shape[0]
        self.num_nodes = self.coordinates.shape[0]

        self.num_size_vars = 0
        self.grouped_size_vars = []

        self.grouped_size_vars.append(np.arange(0, len(self.member_groups['straight_x'][0])//2 * 2))
        self.num_size_vars += len(self.member_groups['straight_x'][0])//2 * 2

        self.grouped_size_vars.append(np.arange(self.num_size_vars,
                                                self.num_size_vars + len(self.member_groups['straight_xz'][0])//2 + 1))
        self.num_size_vars += len(self.member_groups['straight_xz'][0])//2 + 1

        self.grouped_size_vars.append(np.arange(self.num_size_vars,
                                                self.num_size_vars
                                                + (len(self.member_groups['straight_xy'][0])//2 + 1) * 2))
        self.num_size_vars += (len(self.member_groups['straight_xy'][0])//2 + 1) * 2

        self.grouped_size_vars.append(np.arange(self.num_size_vars,
                                                self.num_size_vars + len(self.member_groups['slanted_xz'][0])))
        self.num_size_vars += len(self.member_groups['slanted_xz'][0])

        self.grouped_size_vars.append(np.arange(self.num_size_vars,
                                                self.num_size_vars + len(self.member_groups['cross_yz_end'][0]) // 2))
        self.num_size_vars += len(self.member_groups['cross_yz_end'][0]) // 2

        self.grouped_size_vars.append(np.arange(self.num_size_vars,
                                                self.num_size_vars + len(self.member_groups['cross_xy'][0])//2 * 2))
        self.num_size_vars += len(self.member_groups['cross_xy'][0])//2 * 2

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
        self.r_avg = np.nan * np.ones(self.num_size_vars)
        self.r_std = np.nan * np.ones(self.num_size_vars)
        self.size_rule_score = []  # A score given to a rule between 0 and 1
        for grp in self.grouped_size_vars:
            self.size_rule_score.append(np.zeros(len(grp) - 1))
        self.r_ref = np.nan * np.ones(self.num_size_vars)  # The reference shape to use for innovization
        self.r_ref_history = []

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
    def set_conectivity_matrix(connectivity, r, member_groups):
        r_indx = 0
        m = member_groups['straight_x']
        connectivity[m[0][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # Bottom
        connectivity[m[0][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])
        connectivity[m[2][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # Bottom
        connectivity[m[2][:len(m[0]) // 2], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])
        r_indx += len(m[0]) // 2

        connectivity[m[1][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # Bottom
        connectivity[m[1][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])
        connectivity[m[3][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # Bottom
        connectivity[m[3][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])
        r_indx += len(m[0]) // 2

        m = member_groups['straight_xz']
        connectivity[m[0][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # y = 0
        connectivity[m[1][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # y = 4
        connectivity[m[0][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])
        connectivity[m[1][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])
        r_indx += len(m[0]) // 2 + 1

        m = member_groups['straight_xy']
        connectivity[m[0][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # z = 0
        connectivity[m[1][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # z = 4
        connectivity[m[0][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])  # z = 0
        connectivity[m[1][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])  # z = 4
        r_indx += len(m[0]) // 2 + 1

        m = member_groups['slanted_xz']
        connectivity[m[0], 2] = r[r_indx:r_indx + len(m[0])]
        connectivity[m[1], 2] = np.flip(r[r_indx:r_indx + len(m[0])])
        connectivity[m[2], 2] = r[r_indx:r_indx + len(m[0])]
        connectivity[m[3], 2] = np.flip(r[r_indx:r_indx + len(m[0])])
        r_indx += len(m[0])

        m = member_groups['cross_yz_end']
        connectivity[m[0], 2] = np.array(r[r_indx], r[r_indx])  # x = 0
        connectivity[m[1], 2] = np.array(r[r_indx], r[r_indx])  # x = 72
        r_indx += 1

        m = member_groups['cross_xy']
        connectivity[m[0][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 0
        connectivity[m[0][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 0

        connectivity[m[2][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 0
        connectivity[m[2][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 0
        r_indx += len(m[0]) // 2

        connectivity[m[1][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 4
        connectivity[m[1][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 4

        connectivity[m[3][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 4
        connectivity[m[3][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 4

        return connectivity

    @staticmethod
    def set_coordinate_matrix(coordinates, z, num_shape_vars):
        # Change node coordinates according to the shape decision variables
        coordinates[0:num_shape_vars, 2] = z
        coordinates[(2*num_shape_vars - 1) * 2:(2*num_shape_vars - 1) * 2 + num_shape_vars, 2] = z
        coordinates[num_shape_vars:2*num_shape_vars - 1, 2] = np.flip(z[:-1])
        coordinates[(2*num_shape_vars - 1) * 2 + num_shape_vars:(2*num_shape_vars - 1) * 2 + 2*num_shape_vars - 1, 2] \
            = np.flip(z[:-1])

        return coordinates

    @staticmethod
    def calc_obj(x_row_indx, x, coordinates, connectivity, member_groups, fixed_nodes, load_nodes, force, density,
                 elastic_modulus, yield_stress, max_allowable_displacement, num_shape_vars, structure_type='truss'):
        r = np.copy(x[:-num_shape_vars])  # Radius of each element
        z = np.copy(x[-num_shape_vars:])  # Z-coordinate of bottom members

        connectivity = TrussProblemSymmetric.set_conectivity_matrix(connectivity=connectivity, r=r,
                                                                    member_groups=member_groups)

        coordinates = TrussProblemSymmetric.set_coordinate_matrix(coordinates=coordinates, z=z,
                                                                  num_shape_vars=num_shape_vars)

        weight, compliance, stress, strain, u, x0_new = run_fea(np.copy(coordinates), np.copy(connectivity),
                                                                fixed_nodes,
                                                                load_nodes, force, density,
                                                                elastic_modulus, structure_type=structure_type)
        del_coord = np.array(x0_new) - coordinates

        f = np.array([weight, compliance])

        # Allow displacement only in -z direction
        if np.max(del_coord[:, 2]) > 0:
            g3 = np.max(del_coord[:, 2])
        else:
            g3 = -1
        g = np.array([np.max(np.abs(stress)) - yield_stress, np.max(np.abs(u)) - max_allowable_displacement, g3])

        return x_row_indx, f, g, stress, strain, u, x0_new, coordinates, connectivity

    def _evaluate(self, x_in, out, *args, **kwargs):
        # TODO: Parallelize obj eval
        x = np.copy(x_in)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # KLUGE: Force smoothen shape
        # x[:, -self.num_shape_vars:] = np.flip(np.sort(x[:, -self.num_shape_vars:], axis=1), axis=1)

        if hasattr(kwargs['algorithm'], 'innovization') and kwargs['algorithm'].repair is not None and type(kwargs['algorithm'].repair) != NoRepair:
            x = kwargs['algorithm'].repair.do(self, np.copy(x), **kwargs)

        if self.n_cores > 1:
            pool = mp.Pool(self.n_cores)
            logging.debug(f"Multiprocessing pool opened. CPU count = {mp.cpu_count()}, Pool Size = {self.n_cores}")
            print(f"Multiprocessing pool opened. CPU count = {mp.cpu_count()}, Pool Size = {self.n_cores}")

            # Call apply_async() for asynchronous evaluation of each population member
            result_objects = [pool.apply_async(TrussProblemSymmetric.calc_obj, args=(i, row, np.copy(self.coordinates),
                                                                                     np.copy(self.connectivity),
                                                                                     self.member_groups,
                                                                                     self.fixed_nodes,
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

            if x_in.ndim == 1:
                out['X'] = x.flatten()
            else:
                out['X'] = np.copy(x)
            out['F'] = np.array([[r[1][0], r[1][1]] for r in results])
            out['G'] = np.array([[r[2][c] for c in range(self.n_constr)] for r in results])

            out['stress'] = np.array([r[3] for r in results])
            out['strain'] = np.array([r[4] for r in results])
            out['u'] = np.array([r[5] for r in results])
            out['x0_new'] = np.array([r[6] for r in results])
            out['coordinates'] = np.array([r[7] for r in results])
            out['connectivity'] = np.array([r[8] for r in results])
        else:
            print("Sequential execution.")
            result_objects = map(TrussProblemSymmetric.calc_obj,
                                 np.arange(x.shape[0]),
                                 x,
                                 [np.copy(self.coordinates) for _ in range(x.shape[0])],
                                 [np.copy(self.connectivity) for _ in range(x.shape[0])],
                                 [self.member_groups for _ in range(x.shape[0])],
                                 [self.fixed_nodes for _ in range(x.shape[0])],
                                 [self.load_nodes for _ in range(x.shape[0])],
                                 [self.force for _ in range(x.shape[0])],
                                 [self.density for _ in range(x.shape[0])],
                                 [self.elastic_modulus for _ in range(x.shape[0])],
                                 [self.yield_stress for _ in range(x.shape[0])],
                                 [self.max_allowable_displacement for _ in range(x.shape[0])],
                                 [self.num_shape_vars for _ in range(x.shape[0])],
                                 ['truss' for _ in range(x.shape[0])])

            logging.debug("Objective evaluation complete.")

            # Result_objects is a list of pool.ApplyResult objects
            results = list(result_objects)
            # results = result_objects

            if x_in.ndim == 1:
                out['X'] = x.flatten()
            else:
                out['X'] = np.copy(x)
            out['F'] = np.array([r[1] for r in results])
            out['G'] = np.array([r[2] for r in results])

            out['stress'] = np.array([r[3] for r in results])
            out['strain'] = np.array([r[4] for r in results])
            out['u'] = np.array([r[5] for r in results])
            out['x0_new'] = np.array([r[6] for r in results])
            out['coordinates'] = np.array([r[7] for r in results])
            out['connectivity'] = np.array([r[8] for r in results])
