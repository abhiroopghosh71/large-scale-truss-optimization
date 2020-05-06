import pytest
import numpy as np
import os
import time

from run_fea import run_fea


def test_truss_evaluator_python():
    os.chdir('..')

    # ISCSO example files
    coordinates_file = 'tests/test_truss_input_output/coord_iscso.csv'
    connectivity_file = 'tests/test_truss_input_output/connect_iscso.csv'
    fixed_nodes_file = 'tests/test_truss_input_output/fixn_iscso.csv'
    load_nodes_file = 'tests/test_truss_input_output/loadn_iscso.csv'
    force_file = 'tests/test_truss_input_output/force_iscso.csv'

    coordinates = np.loadtxt(coordinates_file, delimiter=',')
    connectivity = np.loadtxt(connectivity_file, delimiter=',')
    fixed_nodes = np.loadtxt(fixed_nodes_file).reshape(-1, 1)
    load_nodes = np.loadtxt(load_nodes_file).reshape(-1, 1)
    force = np.loadtxt(force_file, delimiter=',')

    density = 7121.4
    elastic_modulus = 200e9

    t0 = time.time()
    weight_truss, compliance_truss, stress_truss, strain_truss, u_truss, x0_new_truss = run_fea(
        coordinates=coordinates, connectivity=connectivity, fixed_nodes=fixed_nodes, load_nodes=load_nodes, force=force,
        density=density, elastic_modulus=elastic_modulus, structure_type='truss')
    print(f"\nExecution Time = {time.time() - t0} seconds")

    print(f"Weight = {weight_truss}")
    print(f"Volume = {weight_truss/density}")
    print(f"Compliance = {compliance_truss}")

    # For truss setting
    assert np.isclose(weight_truss, 6169.28834420487)
    assert np.isclose(compliance_truss, 21.9414649459514)

    # Kelvin's example files
    coordinates_file = 'truss/sample_input/Coordinates.csv'
    connectivity_file = 'truss/sample_input/Connectivity.csv'
    fixed_nodes_file = 'truss/sample_input/fixnodes.csv'
    load_nodes_file = 'truss/sample_input/loadnodes.csv'
    force_file = 'truss/sample_input/force.csv'

    t0 = time.time()
    coordinates = np.loadtxt(coordinates_file, delimiter=',')
    connectivity = np.loadtxt(connectivity_file, delimiter=',')
    fixed_nodes = np.loadtxt(fixed_nodes_file).reshape(-1, 1)
    load_nodes = np.loadtxt(load_nodes_file).reshape(-1, 1)
    force = np.loadtxt(force_file, delimiter=',')

    density = 7121.4
    elastic_modulus = 2850

    weight_frame, compliance_frame, stress_frame, strain_frame, u_frame, x0_new_frame = run_fea(
        coordinates=coordinates, connectivity=connectivity, fixed_nodes=fixed_nodes, load_nodes=load_nodes, force=force,
        density=density, elastic_modulus=elastic_modulus, structure_type='frame')
    print(f"\nExecution Time = {time.time() - t0} seconds")

    print(f"Weight = {weight_frame}")
    print(f"Volume = {weight_frame/density}")
    print(f"Compliance = {compliance_frame}")

    # For frame setting
    assert np.isclose(weight_frame, 230381.69453679543)
    assert np.isclose(compliance_frame, 4.154949024003706)
