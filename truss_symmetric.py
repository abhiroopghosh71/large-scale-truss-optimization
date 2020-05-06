import multiprocessing as mp
from pymoo.model.problem import Problem
import numpy as np
import logging

from utils.generate_truss import gen_truss
from run_fea import run_fea


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
        # for key in self.member_groups.keys():
        #     members = self.member_groups[key]
        #     if key == 'straight_x':
        #         for m in members:
        #             self.num_size_vars += len(m) // 2
        #     elif key == 'straight_xz':
        #         for m in members:
        #             self.num_size_vars += len(m)//2 + 1
        #     elif key == 'straight_xy':
        #         for m in members:
        #             self.num_size_vars += len(m)//2 + 1
        #     elif key == 'slanted_xz':
        #         self.num_size_vars += len(members[0]) * 4

        self.fixed_nodes = self.fixed_nodes.reshape(-1, 1)
        self.load_nodes = self.load_nodes.reshape(-1, 1)

        print(f"No. of shape vars = {self.num_shape_vars}")
        print(self.force)

        # Innovization parameters (shape)
        self.z_avg = -1e16 * np.ones(self.num_shape_vars)
        self.z_std = -1e16 * np.ones(self.num_shape_vars)
        self.shape_rule_score = np.zeros(self.num_shape_vars - 1)  # A score given to a rule between 0 and 1

        # Innovization parameters (size)
        # TODO: Replace with groups returned by gen_truss                      ]
        # self.grouped_size_vars = [np.arange(0, 2 * self.num_shape_vars - 2),
        #                           np.arange(2*self.num_shape_vars - 2, 4*self.num_shape_vars - 4),
        #                           np.arange(4*self.num_shape_vars - 4, 6*self.num_shape_vars - 6),
        #                           np.arange(6*self.num_shape_vars - 6, 8*self.num_shape_vars - 8),
        #                           ]
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
    def calc_obj(i, x, coordinates, connectivity, member_groups, fixed_nodes, load_nodes, force, density, elastic_modulus,
                 yield_stress, max_allowable_displacement, num_shape_vars, structure_type='truss'):
        r = np.copy(x[:-num_shape_vars])  # Radius of each element
        z = np.copy(x[-num_shape_vars:])  # Z-coordinate of bottom members

        # for key in member_groups:
        #     # Take each member type
        #     grp = member_groups[key]
        #     # Among each member type there are subtypes
        #     for m in grp:
        #         m_length = len(m)
        #         connectivity[m[:len(m)//2], 2] = x[x_indx:x_indx + len(m)//2]
        #         connectivity[m[len(m)//2:], 2] = np.flip(x[x_indx:x_indx + len(m)//2])
        #         x_indx += len(m)//2 + 1
        # num_size_vars += len(member_groups['straight_x'][0])//2 * 2
        # num_size_vars += len(member_groups['straight_xz'][0])//2 + 1
        # num_size_vars += len(member_groups['straight_xy'][0])//2 + 1
        # num_size_vars += len(member_groups['slanted_xz'][0]) * 2
        # num_size_vars += len(member_groups['cross_yz_end'][0]) // 2
        # num_size_vars += len(member_groups['cross_xy'][0])//2 * 2

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

        # Change node coordinates according to the shape decision variables
        coordinates[0:num_shape_vars, 2] = z
        coordinates[(2*num_shape_vars - 1) * 2:(2*num_shape_vars - 1) * 2 + num_shape_vars, 2] = z
        coordinates[num_shape_vars:2*num_shape_vars - 1, 2] = np.flip(z[:-1])
        coordinates[(2*num_shape_vars - 1) * 2 + num_shape_vars:(2*num_shape_vars - 1) * 2 + 2*num_shape_vars - 1, 2]\
            = np.flip(z[:-1])

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

        # KLUGE: Force smoothen shape
        # x[:, -self.num_shape_vars:] = np.flip(np.sort(x[:, -self.num_shape_vars:], axis=1), axis=1)

        if hasattr(kwargs['algorithm'], 'repair') and kwargs['algorithm'].repair is not None:
            x = kwargs['algorithm'].repair.do(self, np.copy(x), **kwargs)

        pool = mp.Pool(self.n_cores)
        logging.debug(f"Multiprocessing pool opened. CPU count = {mp.cpu_count()}, Pool Size = {self.n_cores}")

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