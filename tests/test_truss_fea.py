import pytest
import numpy as np
import matlab.engine
import matlab
import os


def test_truss_evaluator():
    os.chdir('..')
    eng = matlab.engine.start_matlab()

    # Kelvin's example files
    # coordinates_file = 'truss/sample_input/Coordinates.csv'
    # connectivity_file = 'truss/sample_input/Connectivity.csv'
    # fixednodes_file = 'truss/sample_input/fixnodes.csv'
    # loadn_file = 'truss/sample_input/loadnodes.csv'
    # force_file = 'truss/sample_input/force.csv'

    # ISCSO example files
    coordinates_file = 'tests/test_truss_input/coord_iscso.csv'
    connectivity_file = 'tests/test_truss_input/connect_iscso.csv'
    fixednodes_file = 'tests/test_truss_input/fixn_iscso.csv'
    loadn_file = 'tests/test_truss_input/loadn_iscso.csv'
    force_file = 'tests/test_truss_input/force_iscso.csv'

    coordinates = matlab.double(np.loadtxt(coordinates_file, delimiter=',').tolist())
    connectivity = matlab.double(np.loadtxt(connectivity_file, delimiter=',').tolist())
    fixednodes = matlab.double(np.loadtxt(fixednodes_file).reshape(-1, 1).tolist())
    loadn = matlab.double(np.loadtxt(loadn_file).reshape(-1, 1).tolist())
    force = matlab.double(np.loadtxt(force_file, delimiter=',').reshape(-1, 1).tolist())

    density = matlab.double([7121.4])
    elastic_modulus = matlab.double([200e9])
    # draw_truss(Coordinates, Connectivity, fixednodes, loadn, force)

    weight, compliance, stress, strain, U, x0_new = eng.run_fea(
        coordinates, connectivity, fixednodes, loadn, force, density, elastic_modulus, nargout=6)

    print(f"Weight = {weight}")
    print(f"Compliance = {compliance}")

    # For frame setting
    # assert np.isclose(weight, 0.000230381701330425)
    # assert np.isclose(compliance, 4.15494884447509)

    # For truss setting
    assert np.isclose(weight, 6169.28834420487)
    assert np.isclose(compliance, 21.9414649459514)
