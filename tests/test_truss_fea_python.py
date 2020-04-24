import pytest
import numpy as np
import os
import time

from run_fea import run_fea


def test_truss_evaluator_python():
    os.chdir('..')

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

    coordinates = np.loadtxt(coordinates_file, delimiter=',')
    connectivity = np.loadtxt(connectivity_file, delimiter=',')
    fixednodes = np.loadtxt(fixednodes_file).reshape(-1, 1)
    loadn = np.loadtxt(loadn_file).reshape(-1, 1)
    force = np.loadtxt(force_file, delimiter=',')

    density = 7121.4
    elastic_modulus = 200e9
    # draw_truss(Coordinates, Connectivity, fixednodes, loadn, force)

    t0 = time.time()
    weight, compliance, stress, strain, U, x0_new = run_fea(
        coordinates, connectivity, fixednodes, loadn, force, density, elastic_modulus)
    print(f"Execution Time = {time.time() - t0} seconds")

    print(f"Weight = {weight}")
    print(f"Compliance = {compliance}")

    # For frame setting
    # assert np.isclose(weight, 0.000230381701330425)
    # assert np.isclose(compliance, 4.15494884447509)

    # For truss setting
    assert np.isclose(weight, 6169.28834420487)
    assert np.isclose(compliance, 21.9414649459514)
