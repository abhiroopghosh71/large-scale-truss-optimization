import pytest
import numpy as np
import matlab.engine
import matlab
import os


def test_truss_evaluator():
    os.chdir('..')
    eng = matlab.engine.start_matlab()

    coordinates_file = 'truss/sample_input/Coordinates.csv'
    connectivity_file = 'truss/sample_input/Connectivity.csv'
    fixednodes_file = 'truss/sample_input/fixnodes.csv'
    loadn_file = 'truss/sample_input/loadnodes.csv'
    force_file = 'truss/sample_input/force.csv'

    # coordinates_file = 'coord_iscso.csv';
    # connectivity_file = 'connect_iscso.csv';
    # fixednodes_file = 'fixn_iscso.csv';
    # loadn_file = 'loadn_iscso.csv';
    # force_file = 'force.csv';

    # coordinates = eng.load(coordinates_file)
    # connectivity = eng.load(connectivity_file)
    # fixednodes = eng.load(fixednodes_file)
    # loadn = eng.load(loadn_file)
    # force = eng.load(force_file)

    coordinates = matlab.double(np.loadtxt(coordinates_file, delimiter=',').tolist())
    connectivity = matlab.double(np.loadtxt(connectivity_file, delimiter=',').tolist())
    fixednodes = matlab.double(np.loadtxt(fixednodes_file, delimiter=',').reshape(-1, 1).tolist())
    loadn = matlab.double(np.loadtxt(loadn_file, delimiter=',').reshape(-1, 1).tolist())
    force = matlab.double(np.loadtxt(force_file, delimiter=',').reshape(-1, 1).tolist())

    density = matlab.double([7.12140021e-6])
    elastic_modulus = matlab.double([2850.0])
    # draw_truss(Coordinates, Connectivity, fixednodes, loadn, force)
    weight, compliance, stress, strain = eng.run_fea(coordinates, connectivity, fixednodes, loadn, force, density,
                                                     elastic_modulus, nargout=4)

    assert np.isclose(weight, 0.000230381701330425)
    assert np.isclose(compliance, 4.15494884447509)
