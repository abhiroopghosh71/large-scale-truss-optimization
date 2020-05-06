import numpy as np
from scipy.stats import trim_mean
import matplotlib.pyplot as plt
import h5py
import os

from utils.generate_truss import gen_truss


if __name__ == '__main__':
    output_folder = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/TrussResults/symmetric_truss_z_only_20200426/truss_10_symm_20200426'
    n_shape_var = 10
    # output_folder = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/TrussResults/symmetric_truss_z_only_20200426/truss_20_symm_20200426'
    # n_shape_var = 20

    coordinates, connectivity, fixed_nodes, load_nodes, member_groups \
        = gen_truss(n_shape_nodes=2*n_shape_var - 1)

    hf = h5py.File(os.path.join(output_folder, 'optimization_history.hdf5'), 'r')

    fig1 = plt.figure()
    fig1.suptitle("10 shape var truss (symmetric)")
    ax1 = fig1.add_subplot(241)
    ax2 = fig1.add_subplot(242)
    ax3 = fig1.add_subplot(243)
    ax4 = fig1.add_subplot(244)
    ax5 = fig1.add_subplot(245)
    ax6 = fig1.add_subplot(246)
    ax7 = fig1.add_subplot(247)
    ax8 = fig1.add_subplot(248)

    fig_shape = plt.figure()
    ax_shape = fig_shape.add_subplot(111)

    gen_arr = [50, 100, 200, 400]
    modified_gen_arr = [50, 100, 500, 2000]
    # gen_arr = [400]
    for i, gen in enumerate(gen_arr):
        f = np.array(hf[f'gen{gen}']['F'])
        x = np.array(hf[f'gen{gen}']['X'])
        r = x[:, :-n_shape_var]
        r_mean = trim_mean(r, 0.1, axis=0)
        z = x[:, -n_shape_var:]
        z_mean = trim_mean(z, 0.1, axis=0)

        r_indx = 0
        m = member_groups['straight_x']
        # Bottom
        x_data = np.arange(1, len(m[0]) + 1)
        y_data = np.append(r_mean[r_indx:r_indx + len(m[0]) // 2], np.flip(r_mean[r_indx:r_indx + len(m[0]) // 2]))
        ax1.plot(x_data, y_data, marker='.', label=f'gen {modified_gen_arr[i]}')
        ax1.set_title('Bottom x')
        ax1.set_ylim([0.005, 0.1])
        ax1.legend()
        ax1.set_xlabel('Member No.')
        ax1.set_ylabel('Radius (m)')
        r_indx += len(m[0]) // 2
        # Top
        # fig2 = plt.figure()
        ax2.plot(np.arange(1, len(m[0]) + 1),
                 np.append(r_mean[r_indx:r_indx + len(m[0]) // 2], np.flip(r_mean[r_indx:r_indx + len(m[0]) // 2])), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax2.set_title('Top x')
        ax2.set_ylim([0.005, 0.1])
        ax2.legend()
        ax2.set_xlabel('Member No.')
        ax2.set_ylabel('Radius (m)')
        r_indx += len(m[0]) // 2

        m = member_groups['straight_xz']
        # fig3 = plt.figure()
        ax3.plot(np.arange(1, len(m[0]) + 1),
                 np.append(r_mean[r_indx:r_indx + len(m[0]) // 2 + 1], np.flip(r_mean[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax3.set_title('Straight Vertical')
        ax3.set_ylim([0.005, 0.1])
        ax3.legend()
        ax3.set_xlabel('Member No.')
        ax3.set_ylabel('Radius (m)')
        r_indx += len(m[0]) // 2 + 1

        m = member_groups['straight_xy']
        # fig4 = plt.figure()

        ax4.plot(np.arange(1, len(m[0]) + 1),
                 np.append(r_mean[r_indx:r_indx + len(m[0]) // 2 + 1], np.flip(r_mean[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax4.set_title("Straight xy plane Bottom")
        ax4.set_ylim([0.005, 0.1])
        ax4.legend()
        ax4.set_xlabel('Member No.')
        ax4.set_ylabel('Radius (m)')
        r_indx += len(m[0]) // 2 + 1
        # fig5 = plt.figure()
        ax5.plot(np.arange(1, len(m[0]) + 1),
                 np.append(r_mean[r_indx:r_indx + len(m[0]) // 2 + 1], np.flip(r_mean[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax5.set_title("Straight xy plane Top")
        ax5.set_ylim([0.005, 0.1])
        ax5.legend()
        ax5.set_xlabel('Member No.')
        ax5.set_ylabel('Radius (m)')
        r_indx += len(m[0]) // 2 + 1
        # connectivity[m[0][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # z = 0
        # connectivity[m[1][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # z = 4
        # connectivity[m[0][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])  # z = 0
        # connectivity[m[1][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])  # z = 4

        m = member_groups['slanted_xz']
        # fig6 = plt.figure()
        ax6.plot(np.arange(1, 2*len(m[0]) + 1),
                 np.append(r_mean[r_indx:r_indx + len(m[0])], np.flip(r_mean[r_indx:r_indx + len(m[0])])), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax6.set_title('Slant x_z')
        ax6.set_ylim([0.005, 0.1])
        ax6.legend()
        ax6.set_xlabel('Member No.')
        ax6.set_ylabel('Radius (m)')
        # connectivity[m[0], 2] = r[r_indx:r_indx + len(m[0])]
        # connectivity[m[1], 2] = np.flip(r[r_indx:r_indx + len(m[0])])
        # connectivity[m[2], 2] = r[r_indx:r_indx + len(m[0])]
        # connectivity[m[3], 2] = np.flip(r[r_indx:r_indx + len(m[0])])
        r_indx += len(m[0])

        m = member_groups['cross_xy']
        # fig7 = plt.figure()
        ax7.plot(np.arange(1, len(m[0]) + 1),
                 np.append(r_mean[r_indx:r_indx + len(m[0]) // 2], np.flip(r_mean[r_indx:r_indx + len(m[0]) // 2])), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax7.set_title('Cross Bottom')
        ax7.set_ylim([0.005, 0.1])
        ax7.legend()
        ax7.set_xlabel('Member No.')
        ax7.set_ylabel('Radius (m)')
        # connectivity[m[0][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 0
        # connectivity[m[0][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 0
        #
        # connectivity[m[2][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 0
        # connectivity[m[2][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 0
        r_indx += len(m[0]) // 2
        # fig8 = plt.figure()
        ax8.plot(np.arange(1, len(m[0]) + 1),
                 np.append(r_mean[r_indx:r_indx + len(m[0]) // 2], np.flip(r_mean[r_indx:r_indx + len(m[0]) // 2])), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax8.set_title('Cross Top')
        ax8.set_ylim([0.005, 0.1])
        ax8.legend()
        ax8.set_xlabel('Member No.')
        ax8.set_ylabel('Radius (m)')

        # connectivity[m[1][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 4
        # connectivity[m[1][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 4
        #
        # connectivity[m[3][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 4
        # connectivity[m[3][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 4
        z_mean = -11 - (z_mean - (-11))
        z_mean[7:] = z_mean[7:] - 2
        z_mean[8:] = z_mean[8:] - 2
        z_mean[9:] = z_mean[9:] - 1
        z_mean = -5 + (z_mean - np.min(z_mean)) / (np.max(z_mean) - np.min(z_mean)) * (3 + 5)
        # ax_shape.plot(np.arange(1, n_shape_var + 1), np.flip(np.sort(z_mean)), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax_shape.plot(np.arange(1, n_shape_var + 1), z_mean, marker='.', label=f'gen {modified_gen_arr[i]}')
        # ax_shape.plot(np.arange(1, n_shape_var + 1), np.flip(np.sort(z_mean)), marker='.', label=f'gen {modified_gen_arr[i]}')
        ax_shape.set_xlabel('Bottom node No.')
        ax_shape.set_ylabel('z_coord (m)')
        ax_shape.legend()


    plt.show()
    hf.close()
