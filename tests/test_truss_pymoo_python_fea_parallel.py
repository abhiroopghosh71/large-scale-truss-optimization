import pytest
import os
import numpy as np
import time
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize


import optimize_truss_python_fea_parallel
from truss.truss_symmetric_shape_only import TrussProblem


def test_truss_optimizer_python_fea():
    t0 = time.time()
    os.chdir('..')
    seed = 184716924
    optimize_truss_python_fea_parallel.save_folder = os.path.join(os.getcwd(), 'output',
                                                                  f'truss_nsga2_test_seed{seed}_'
                                                                  f'{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(optimize_truss_python_fea_parallel.save_folder)
    problem = TrussProblem()

    coordinates_file = 'tests/test_truss_input_output/coord_iscso.csv'
    connectivity_file = 'tests/test_truss_input_output/connect_iscso.csv'
    fixednodes_file = 'tests/test_truss_input_output/fixn_iscso.csv'
    loadn_file = 'tests/test_truss_input_output/loadn_iscso.csv'
    force_file = 'tests/test_truss_input_output/force_iscso.csv'

    problem.coordinates = np.loadtxt(coordinates_file, delimiter=',')
    problem.connectivity = np.loadtxt(connectivity_file, delimiter=',')
    problem.fixed_nodes = np.loadtxt(fixednodes_file).reshape(-1, 1)
    problem.load_nodes = np.loadtxt(loadn_file).reshape(-1, 1)
    problem.force = np.loadtxt(force_file, delimiter=',')

    algorithm = NSGA2(
        pop_size=20,
        # n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=30),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
        callback=optimize_truss_python_fea_parallel.record_state,
        display=optimize_truss_python_fea_parallel.OptimizationDisplay()
    )

    termination = get_termination("n_gen", 50)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=False,
                   verbose=True)

    print(f"Total execution time = {time.time() - t0} seconds")

    np.allclose(res.F,
                np.array([[9.13229410e+04, 1.94409845e+00],
                          [1.08504089e+05, 1.33373249e+00],
                          [9.17889667e+04, 1.63208874e+00],
                          [9.15334147e+04, 1.93918346e+00],
                          [9.70089217e+04, 1.44792709e+00],
                          [9.61897260e+04, 1.57189272e+00],
                          [9.51286696e+04, 1.61464174e+00],
                          [1.05929941e+05, 1.36121166e+00],
                          [1.02619727e+05, 1.36470401e+00],
                          [9.21607549e+04, 1.61505579e+00],
                          [9.91146873e+04, 1.43466841e+00],
                          [1.01507655e+05, 1.36751474e+00],
                          [9.53681942e+04, 1.58883931e+00],
                          [1.07101278e+05, 1.34553784e+00],
                          [9.95689951e+04, 1.39073567e+00],
                          [1.07801835e+05, 1.34135548e+00],
                          [9.92677206e+04, 1.41917837e+00],
                          [1.06148148e+05, 1.36108686e+00],
                          [1.00480161e+05, 1.38404339e+00],
                          [1.02508059e+05, 1.36489444e+00]]))
