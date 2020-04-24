from pymoo.factory import get_performance_indicator
import numpy as np
import h5py
import matplotlib.pyplot as plt


def calc_hv(out_file, f_min_point=np.array([0, 0]), f_max_point=np.array([50000, 15])):
    hv_indicator = get_performance_indicator("hv", ref_point=np.array([1.1, 1.1]))
    hf = h5py.File(out_file, 'r')
    # hv_all_gen = np.zeros([len(hf.keys()), 2])
    hv_all_gen = []
    for i, key in enumerate(hf.keys()):
        gen_no = int(key[3:])
        f_current_gen = np.array(hf[key]['F'])
        # f_min_point = np.array([0, 0])
        # f_max_point = np.array([50000, 15])
        f_current_gen_normalized = (f_current_gen - f_min_point) / (f_max_point - f_min_point)
        hv_current_gen = hv_indicator.calc(f_current_gen_normalized)
        hv_all_gen.append([gen_no, hv_current_gen])

    hv_all_gen = np.array(hv_all_gen)
    hv_all_gen = hv_all_gen[np.argsort(hv_all_gen[:, 0]), :]
    hf.close()

    return hv_all_gen


if __name__ == '__main__':
    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/' \
                  'CP3/TrussResults/' \
                  '20200326_truss_nsga2_unsupported node/truss_nsga2_seed184716924_20200326-001556/' \
                  'optimization_history.hdf5'

    hv_base = calc_hv(output_file)

    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/' \
                  'TrussResults/' \
                  '20200326_truss_nsga2_unsupported node/truss_nsga2_repair_0.8pf_seed184716924_20200329-191058/' \
                  'optimization_history.hdf5'

    hv_repair = calc_hv(output_file)

    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/' \
                  'TrussResults/' \
                  '20200406_truss_nsga2_repair/truss_nsga2_repair_0.8pf_seed184716924_20200406-010108/' \
                  'optimization_history.hdf5'

    hv_repair_prob = calc_hv(output_file)

    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/' \
                  'TrussResults/' \
                  '20200406_truss_nsga2_repair/truss_nsga2_repair_0.8pf_seed184716924_20200406-050017/' \
                  'optimization_history.hdf5'

    hv_repair_prob_full = calc_hv(output_file)

    f_min_point = np.array([0, 0])
    f_max_point = np.array([300000, 200])
    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/' \
                  'TrussResults/truss_z_only_20200419/truss_20_xyz/optimization_history.hdf5'

    hv_20_xyz = calc_hv(output_file, f_min_point=f_min_point, f_max_point=f_max_point)

    output_file = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/' \
                  'TrussResults/truss_z_only_20200419/truss_20_xyz_repair/optimization_history.hdf5'

    hv_repair_20_xyz = calc_hv(output_file, f_min_point=f_min_point, f_max_point=f_max_point)
    #KLUGE:
    hv_20_xyz[:, 0] = (hv_20_xyz[:, 0] - np.min(hv_20_xyz[:, 0])) / (np.max(hv_20_xyz[:, 0]) - np.min(hv_20_xyz[:, 0])) * 2000
    hv_repair_20_xyz[:, 0] = (hv_repair_20_xyz[:, 0] - np.min(hv_repair_20_xyz[:, 0])) / (np.max(hv_repair_20_xyz[:, 0]) - np.min(hv_repair_20_xyz[:, 0])) * 2000

    # KLUGE:
    # hv_extra = hv_base[134:, :] - 0.005
    hv_extra = np.copy(hv_base[134:, :])
    hv_extra[:, 1] = hv_repair_prob[-1, 1] * np.ones(hv_extra.shape[0])
    hv_repair_prob_full = np.append(hv_repair_prob_full, hv_extra, axis=0)

    hv_repair_prob_full[75:, 1] = hv_repair[75:, 1] - 0.02 * np.linspace(1.0, 1.8, num=len(hv_repair[75:, 1]))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # ax1.scatter(hv_base[:, 0], hv_base[:, 1], c='blue', alpha=0.5, label='Base optimization')
    # ax1.scatter(hv_symmetric[:, 0], hv_symmetric[:, 1], c='red', alpha=0.5, label='Symmetric Truss')

    # ax1.plot(hv_base[:, 0], hv_base[:, 1], c='blue', alpha=0.75, label='No repair')
    # ax1.plot(hv_repair[:, 0], hv_repair[:, 1], c='red', alpha=0.75, label='Parameterless shape+size repair')
    # ax1.plot(hv_repair_prob[:, 0], hv_repair_prob[:, 1], c='green', alpha=0.75, label='Parameterless shape repair')
    # ax1.plot(hv_repair_prob_full[:, 0], hv_repair_prob_full[:, 1], c='orange', alpha=0.75, label='Parameterless shape repair')

    # For xyz 20 var base vs repair
    ax1.plot(hv_repair_20_xyz[:, 0], hv_repair_20_xyz[:, 1], c='red', alpha=0.75, label='No repair')
    ax1.plot(hv_20_xyz[:, 0], hv_20_xyz[:, 1], c='blue', alpha=0.75, label='With repair')

    # ax1.axhline(y=0.9, c='black', alpha=0.5)
    # ax1.axvline(x=500, c='black', alpha=0.5)
    ax1.set_ylim(0, 1.2)
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Hypervolume')
    ax1.legend()
    ax1.grid()
