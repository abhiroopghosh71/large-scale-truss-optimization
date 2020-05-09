import os
import time

import matlab
import matlab.engine
import numpy as np


def test_truss_evaluator():
    os.chdir('..')
    eng = matlab.engine.start_matlab()

    # ISCSO example files
    coordinates_file = 'tests/test_truss_input_output/coord_iscso.csv'
    connectivity_file = 'tests/test_truss_input_output/connect_iscso.csv'
    fixednodes_file = 'tests/test_truss_input_output/fixn_iscso.csv'
    loadn_file = 'tests/test_truss_input_output/loadn_iscso.csv'
    force_file = 'tests/test_truss_input_output/force_iscso.csv'

    coordinates = matlab.double(np.loadtxt(coordinates_file, delimiter=',').tolist())
    connectivity = matlab.double(np.loadtxt(connectivity_file, delimiter=',').tolist())
    fixednodes = matlab.double(np.loadtxt(fixednodes_file).reshape(-1, 1).tolist())
    loadn = matlab.double(np.loadtxt(loadn_file).reshape(-1, 1).tolist())
    force = matlab.double(np.loadtxt(force_file, delimiter=',').tolist())

    density = matlab.double([7121.4])
    elastic_modulus = matlab.double([200e9])
    # draw_truss(Coordinates, Connectivity, fixednodes, loadn, force)

    t0 = time.time()
    weight_truss, compliance_truss, stress_truss, strain_truss, u_truss, x0_new_truss = eng.run_fea(
        coordinates, connectivity, fixednodes, loadn, force, density, elastic_modulus, nargout=6)
    print(f"Execution Time = {time.time() - t0} seconds")

    print(f"Weight = {weight_truss}")
    print(f"Compliance = {compliance_truss}")

    # For truss setting
    assert np.isclose(weight_truss, 6169.28834420487)
    assert np.isclose(compliance_truss, 21.9414649459514)

    # Kelvin's example files
    # coordinates_file = 'truss/sample_input/Coordinates.csv'
    # connectivity_file = 'truss/sample_input/Connectivity.csv'
    # fixednodes_file = 'truss/sample_input/fixnodes.csv'
    # loadn_file = 'truss/sample_input/loadnodes.csv'
    # force_file = 'truss/sample_input/force.csv'

    # For frame setting
    # assert np.isclose(weight_truss, 0.000230381701330425)
    # assert np.isclose(compliance_truss, 4.15494884447509)
