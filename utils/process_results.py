import h5py
import os
import numpy as np
from pathlib import Path
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt


def write_data_to_txt(out_file_path):
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

    np.savetxt('x_max_gen', x_max_gen)
    np.savetxt('f_max_gen', f_max_gen)
    np.savetxt('pf_max_gen', f_max_gen[fronts_max_gen[0], :])
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


if __name__ == '__main__':
    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/' \
                  'CP3/TrussResults/20200322_nsga2_truss/truss_nsga2_seed184716924_20200322-015313/' \
                  'optimization_history.hdf5'

    [x, f, fronts_indx] = write_data_to_txt(output_file)
    plot_obj(f)

    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/' \
                  'TrussResults/20200322_nsga2_truss/truss_symmetric_nsga2_seed184716924_20200322-015318/' \
                  'optimization_history.hdf5'

    [x, f, fronts_indx] = write_data_to_txt(output_file)
    plot_obj(f)

    plt.show()
