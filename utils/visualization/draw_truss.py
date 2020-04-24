import numpy as np
import matplotlib.pyplot as plt
from utils.generate_truss import gen_truss
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.markers import MarkerStyle


def draw_truss(node_coordinates, element_connectivity, fixed_nodes, load_nodes):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plot nodes
    ax1.scatter(node_coordinates[:, 0], node_coordinates[:, 1], node_coordinates[:, 2],
                marker=MarkerStyle(marker='o'), c='b')

    # Plot fixed nodes
    ax1.scatter(node_coordinates[fixed_nodes - 1, 0],
                node_coordinates[fixed_nodes - 1, 1],
                node_coordinates[fixed_nodes - 1, 2], marker=MarkerStyle(marker='s', fillstyle='none'), s=200, c='r')

    # Plot elements
    for nodes in element_connectivity:
        ax1.plot([coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
                 [coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
                 [coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]], c='k')

    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.set_zlabel('z', fontsize=10)
    ax1.set_xlim(-5, np.max(node_coordinates[:, 0] + 20))
    ax1.set_ylim(-5, np.max(node_coordinates[:, 1] + 10))
    ax1.set_zlim(np.min(node_coordinates[:, 2]) - 5, np.max(node_coordinates[:, 2] + 5))


if __name__ == '__main__':
    # coordinates, connectivity, fixed_nodes, load_nodes = gen_truss()
    # draw_truss(coordinates, connectivity, fixed_nodes, load_nodes)
    #
    # coordinates, connectivity, fixed_nodes, load_nodes = gen_truss(n_shape_nodes=39)
    # draw_truss(coordinates, connectivity, fixed_nodes, load_nodes)

    # For truss_20_xyz - 20 shape var, 3 force
    num_shape_vars = 20
    coordinates, connectivity, fixed_nodes, load_nodes = gen_truss(n_shape_nodes=num_shape_vars*2 -1)
    # x = np.loadtxt('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/'
    #                'TrussResults/truss_z_only_20200419/truss_20_xyz/x_max_gen')
    # f = np.loadtxt('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/'
    #                'TrussResults/truss_z_only_20200419/truss_20_xyz/f_max_gen')
    # sol_indx = 0
    # r = np.copy(x[sol_indx, :-num_shape_vars])  # Radius of each element
    # z = np.copy(x[sol_indx, -num_shape_vars:])  # Z-coordinate of bottom members
    # z[:13] = np.arange(3, z[13], (z[13] - 3) / 13)  # For min weight
    # z[:13] = np.arange(-3, z[13], (z[13] + 3) / 13)  # For min compliance
    # z[19] = z[18]
    x = np.loadtxt('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/'
                   'TrussResults/truss_z_only_20200419/truss_20_xyz_repair/x_max_gen')
    f = np.loadtxt('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/'
                   'TrussResults/truss_z_only_20200419/truss_20_xyz_repair/f_max_gen')
    sol_indx = 1
    r = np.copy(x[sol_indx, :-num_shape_vars])  # Radius of each element
    z = np.copy(x[sol_indx, -num_shape_vars:])  # Z-coordinate of bottom members
    # z[:13] = np.arange(2.5, z[13], (z[13] - 2.5) / 13)  # For min weight
    z[:14] = np.arange(-3, z[14], (z[14] + 3) / 14)  # For min compliance
    # z[19] = z[18]

    connectivity[:, 2] = r
    # coordinates[0:10, 2] = z
    # coordinates[38:48, 2] = z
    # coordinates[10:19, 2] = np.flip(z[:-1])
    # coordinates[48:57, 2] = np.flip(z[:-1])
    coordinates[0:num_shape_vars, 2] = z
    coordinates[(2*num_shape_vars - 1) * 2:(2*num_shape_vars - 1) * 2 + num_shape_vars, 2] = z
    coordinates[num_shape_vars:2*num_shape_vars - 1, 2] = np.flip(z[:-1])
    coordinates[(2*num_shape_vars - 1) * 2 + num_shape_vars:(2*num_shape_vars - 1) * 2 + 2*num_shape_vars - 1, 2] = np.flip(z[:-1])
    draw_truss(coordinates, connectivity, fixed_nodes, load_nodes)

    plt.show()
