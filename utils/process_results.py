import h5py
import os
import numpy as np
from pathlib import Path
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
import matlab
import matlab.engine

import optimize_truss


def get_x_and_f(out_file_path):
    current_dir = os.getcwd()
    os.chdir(Path(out_file_path).parent)
    hf = h5py.File(out_file_path, 'r')

    max_gen = 0
    max_gen_key = None
    for key in hf.keys():
        gen_no = int(key[3:])
        if gen_no > max_gen:
            max_gen = gen_no
            max_gen_key = key

    x_max_gen = np.array(hf[max_gen_key]['X'])
    f_max_gen = np.array(hf[max_gen_key]['F'])

    fronts_max_gen = NonDominatedSorting().do(f_max_gen)

    print("Files not saved")
    # np.savetxt('x_max_gen', x_max_gen)
    # np.savetxt('f_max_gen', f_max_gen)
    # np.savetxt('pf_max_gen', f_max_gen[fronts_max_gen[0], :])
    hf.close()
    os.chdir(current_dir)

    return x_max_gen, f_max_gen, fronts_max_gen


def plot_obj(f_obj):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.scatter(f_obj[:, 0], f_obj[:, 1], c='blue', alpha=0.5)
    ax1.set_xlabel('Weight (kg)')
    ax1.set_ylabel('Compliance (m/N)')

    return fig1, ax1


def plot_truss(matlab_eng, truss, plot_title=None):
    # Gather enough amount of each type of energy before blasting the matlab engine
    coordinates_matlab = matlab.double(truss.coordinates.tolist())
    connectivity_matlab = matlab.double(truss.connectivity.tolist())
    fixed_nodes_matlab = matlab.double(truss.fixed_nodes.tolist())
    load_nodes_matlab = matlab.double(truss.load_nodes.tolist())
    force_matlab = matlab.double(truss.force.tolist())

    # Fire!! HAAAAAA!!!!
    matlab_eng.draw_truss(coordinates_matlab, connectivity_matlab, fixed_nodes_matlab, load_nodes_matlab,
                          force_matlab, nargout=0)
    if plot_title is not None:
        matlab_eng.title(plot_title)


def convert_x_to_truss_params(x_pop, truss):
    # x = np.copy(x_pop)
    # if x.ndim == 1:
    #     x = x.reshape(1, -1)
    #
    # connectivity = np.zeros([x.shape[0], truss.connectivity.shape[0], truss.connectivity.shape[1]])
    # coordinates = np.zeros([x.shape[0], truss.coordinates.shape[0], truss.coordinates.shape[1]])
    # for indx in range(x.shape[0]):
    r = x_pop[:260]  # Radius of each element
    z = x_pop[260:]  # Z-coordinate of bottom members

    # connectivity = np.copy(truss.connectivity)
    # coordinates = np.copy(truss.coordinates)
    truss.connectivity[:, 2] = r
    truss.coordinates[0:10, 2] = z
    truss.coordinates[38:48, 2] = z
    truss.coordinates[10:19, 2] = np.flip(z[:-1])
    truss.coordinates[48:57, 2] = np.flip(z[:-1])

    weight, compliance, stress, strain, u, x0_new = \
        optimize_truss.matlab_engine.run_fea(matlab.double(truss.coordinates.tolist()),
                              matlab.double(truss.connectivity.tolist()),
                              matlab.double(truss.fixed_nodes.tolist()),
                              matlab.double(truss.load_nodes.tolist()),
                              matlab.double(truss.force.tolist()),
                              matlab.double([truss.density]),
                              matlab.double([truss.elastic_modulus]),
                              nargout=6)
    return truss


if __name__ == '__main__':
    curr_dir = os.getcwd()
    os.chdir('..')

    # Summon MATLAB, one of the Great Old Ones. Using the incantations in the matlab,engine package, we can summon
    # MATLAB powers through a Python individual. Note, however, that any external divine power takes longer to charge.
    # So in combat situations requiring quick response times, this might put you at a disadvantage.
    # matlab_engine = matlab.engine.start_matlab()
    optimize_truss.matlab_engine.addpath(
        r'/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/'
        r'iscso_based_truss_optimization/large_scale_truss_optimization', nargout=0)

    # Base optimization
    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/' \
                  'CP3/TrussResults/' \
                  '20200326_truss_nsga2_unsupported node/truss_nsga2_seed184716924_20200326-001556/' \
                  'optimization_history.hdf5'

    # Repair 0.8 percent pop in pf
    # output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/' \
    #               'CP3/TrussResults/' \
    #               '20200326_truss_nsga2_unsupported node/truss_nsga2_repair_0.8pf_seed184716924_20200329-191058/' \
    #               'optimization_history.hdf5'

    [x, f, fronts_indx] = get_x_and_f(output_file)
    plot_obj(f)

    # Plot the min weight and compliance trusses
    min_obj = np.min(f, axis=0)
    min_obj_indx = np.argmin(f, axis=0)

    truss_min_weight = optimize_truss.TrussProblem()
    convert_x_to_truss_params(x[min_obj_indx[0], :], truss_min_weight)
    plot_truss(optimize_truss.matlab_engine, truss_min_weight,
               plot_title=f'Min. weight Truss\n'
                          f'Weight = {np.around(f[min_obj_indx[0], 0], decimals=2)} kg, '
                          f'Compliance = {np.around(f[min_obj_indx[0], 1], decimals=2)} m/N')

    truss_min_compliance = optimize_truss.TrussProblem()
    convert_x_to_truss_params(x[min_obj_indx[1], :], truss_min_compliance)
    plot_truss(optimize_truss.matlab_engine, truss_min_compliance,
               plot_title='Min. compliance\n'
                          f'Weight = {np.around(f[min_obj_indx[1], 0], decimals=2)} kg, '
                          f'Compliance = {np.around(f[min_obj_indx[1], 1], decimals=2)} m/N')

    # output_file_symm = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/' \
    #                    'Code/CP3/TrussResults/20200326_truss_nsga2_unsupported node/' \
    #                    'truss_symmetric_nsga2_seed184716924_20200326-003009/optimization_history.hdf5'
    #
    # [x_symm, f_symm, fronts_indx_symm] = write_data_to_txt(output_file_symm)
    # plot_obj(f_symm)

    os.chdir(curr_dir)
    plt.show()
