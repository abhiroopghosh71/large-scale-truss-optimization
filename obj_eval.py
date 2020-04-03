import matlab
import numpy as np


def calc_obj(x, indx, coordinates, connectivity, fixed_nodes, load_nodes, force, density, elastic_modulus, yield_stress,
             max_allowable_displacement, matlab_engine):
    r = x[:260]  # Radius of each element
    z = x[260:]  # Z-coordinate of bottom members

    connectivity[:, 2] = r
    coordinates[0:10, 2] = z
    coordinates[38:48, 2] = z
    coordinates[10:19, 2] = np.flip(z[:-1])
    coordinates[48:57, 2] = np.flip(z[:-1])

    weight, compliance, stress, strain, u, x0_new = \
        matlab_engine.run_fea(matlab.double(coordinates.tolist()),
                              matlab.double(connectivity.tolist()),
                              matlab.double(fixed_nodes.tolist()),
                              matlab.double(load_nodes.tolist()),
                              matlab.double(force.tolist()),
                              matlab.double([density]),
                              matlab.double([elastic_modulus]),
                              nargout=6)

    return [indx, weight, compliance, stress, strain, u, x0_new,
            matlab_engine.max(matlab_engine.abs(stress)) - yield_stress,
            matlab_engine.max(matlab_engine.abs(u)) - max_allowable_displacement]
