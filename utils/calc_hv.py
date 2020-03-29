from pymoo.factory import get_performance_indicator
import numpy as np
import h5py
import matplotlib.pyplot as plt


def calc_hv(out_file):
    hv_indicator = get_performance_indicator("hv", ref_point=np.array([1.1, 1.1]))
    hf = h5py.File(out_file, 'r')
    # hv_all_gen = np.zeros([len(hf.keys()), 2])
    hv_all_gen = []
    for i, key in enumerate(hf.keys()):
        gen_no = int(key[3:])
        f_current_gen = np.array(hf[key]['F'])
        f_min_point = np.array([0, 0])
        f_max_point = np.array([30000, 0.1])
        f_current_gen_normalized = (f_current_gen - f_min_point) / (f_max_point - f_min_point)
        hv_current_gen = hv_indicator.calc(f_current_gen_normalized)
        hv_all_gen.append([gen_no, hv_current_gen])

    hv_all_gen = np.array(hv_all_gen)
    hv_all_gen = hv_all_gen[np.argsort(hv_all_gen[:, 0]), :]
    hf.close()

    return hv_all_gen


if __name__ == '__main__':
    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/' \
                  'CP3/TrussResults/20200322_nsga2_truss/truss_nsga2_seed184716924_20200322-015313/' \
                  'optimization_history.hdf5'

    hv_base = calc_hv(output_file)

    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/' \
                  'TrussResults/20200322_nsga2_truss/truss_symmetric_nsga2_seed184716924_20200322-015318/' \
                  'optimization_history.hdf5'

    hv_symmetric = calc_hv(output_file)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # ax1.scatter(hv_base[:, 0], hv_base[:, 1], c='blue', alpha=0.5, label='Base optimization')
    # ax1.scatter(hv_symmetric[:, 0], hv_symmetric[:, 1], c='red', alpha=0.5, label='Symmetric Truss')
    ax1.plot(hv_base[:, 0], hv_base[:, 1], c='blue', alpha=0.5, label='Base optimization')
    ax1.plot(hv_symmetric[:, 0], hv_symmetric[:, 1], c='red', alpha=0.5, label='Symmetric Truss')
    ax1.set_ylim(0, 1.2)
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Hypervolume')
    ax1.legend()
    ax1.grid()
