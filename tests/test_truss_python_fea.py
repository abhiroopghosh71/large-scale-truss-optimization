import os
import time

import numpy as np

from truss.fea.run_fea import run_fea


def test_truss_evaluator_python():
    os.chdir('..')

    # ISCSO example files
    coordinates_file = 'tests/test_truss_input_output/coord_iscso.csv'
    connectivity_file = 'tests/test_truss_input_output/connect_iscso.csv'
    fixed_nodes_file = 'tests/test_truss_input_output/fixn_iscso.csv'
    load_nodes_file = 'tests/test_truss_input_output/loadn_iscso.csv'
    force_file = 'tests/test_truss_input_output/force_iscso.csv'

    # ISCSO correct outputs
    weight_file = 'tests/test_truss_input_output/weight_truss_iscso'
    compliance_file = 'tests/test_truss_input_output/compliance_truss_iscso'
    stress_file = 'tests/test_truss_input_output/stress_truss_iscso'
    strain_file = 'tests/test_truss_input_output/strain_truss_iscso'
    u_file = 'tests/test_truss_input_output/u_truss_iscso'
    x0_new_file = 'tests/test_truss_input_output/x0_new_truss_iscso'

    weight_correct_iscso = np.loadtxt(weight_file)
    compliance_correct_iscso = np.loadtxt(compliance_file)
    stress_correct_iscso = np.loadtxt(stress_file)
    strain_correct_iscso = np.loadtxt(strain_file)
    u_correct_iscso = np.loadtxt(u_file)
    x0_new_correct_iscso = np.loadtxt(x0_new_file)

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
    assert np.allclose(weight_truss, weight_correct_iscso)
    assert np.allclose(compliance_truss, compliance_correct_iscso)
    assert np.allclose(stress_truss, stress_correct_iscso)
    assert np.allclose(strain_truss, strain_correct_iscso)
    assert np.allclose(u_truss, u_correct_iscso)
    assert np.allclose(x0_new_truss, x0_new_correct_iscso)

    # Kelvin's example files
    coordinates_file = 'truss/sample_input/Coordinates.csv'
    connectivity_file = 'truss/sample_input/Connectivity.csv'
    fixed_nodes_file = 'truss/sample_input/fixnodes.csv'
    load_nodes_file = 'truss/sample_input/loadnodes.csv'
    force_file = 'truss/sample_input/force.csv'

    # Kelvin correct outputs
    weight_file = 'tests/test_truss_input_output/weight_frame'
    compliance_file = 'tests/test_truss_input_output/compliance_frame'
    stress_file = 'tests/test_truss_input_output/stress_frame'
    strain_file = 'tests/test_truss_input_output/strain_frame'
    u_file = 'tests/test_truss_input_output/u_frame'
    x0_new_file = 'tests/test_truss_input_output/x0_new_frame'

    weight_correct_frame = np.loadtxt(weight_file)
    compliance_correct_frame = np.loadtxt(compliance_file)
    stress_correct_frame = np.loadtxt(stress_file)
    strain_correct_frame = np.loadtxt(strain_file)
    u_correct_frame = np.loadtxt(u_file)
    x0_new_correct_frame = np.loadtxt(x0_new_file)

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
    assert np.isclose(weight_frame, weight_correct_frame)
    assert np.isclose(compliance_frame, compliance_correct_frame)
    assert np.allclose(stress_frame, stress_correct_frame)
    assert np.allclose(strain_frame, strain_correct_frame)
    assert np.allclose(u_frame, u_correct_frame)
    assert np.allclose(x0_new_frame, x0_new_correct_frame)
