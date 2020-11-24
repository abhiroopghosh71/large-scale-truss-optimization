import warnings

import numpy as np

from utils.visualization.draw_truss import draw_truss


def gen_truss(n_shape_nodes=19, member_radius=0.015, member_length_xyz=np.array([4, 4, 4]), density=7121.4,
              elastic_modulus=200e9, yield_stress=248.2e6, force=np.array([0, 0, -5000])):
    """Generate trusses of the type given in ISCSO 2019"""
    if n_shape_nodes % 2 == 0:
        warnings.warn(f"Number of shape nodes (={n_shape_nodes}) should be odd")
    # n_shape_nodes include all nodes at the bottom on each side
    n_nodes = n_shape_nodes * 4

    all_nodes = np.arange(1, n_nodes + 1)

    # The bottom corner nodes of truss have fixed supports
    fixed_nodes = np.array([1, n_shape_nodes, 2 * n_shape_nodes + 1, 3 * n_shape_nodes])

    # All unsupported nodes carry a load
    load_nodes = np.copy(all_nodes)
    for node in fixed_nodes:
        load_nodes = np.delete(load_nodes, np.argwhere(load_nodes == node))
    load_nodes = np.delete(load_nodes, np.argwhere(load_nodes <= n_shape_nodes))
    for node in range(2*n_shape_nodes + 1, 3 * n_shape_nodes + 1):
        load_nodes = np.delete(load_nodes, np.argwhere(load_nodes == node))

    node_coordinates = np.zeros([n_nodes, 3])

    # For nodes along x-axis
    # x_axis_member_length = truss_length / (n_shape_nodes - 1)
    for i in range(n_shape_nodes):
        node_coordinates[i, :] = [i * member_length_xyz[0], 0, 0]

    for i in range(n_shape_nodes, 2 * n_shape_nodes):
        node_coordinates[i, :] = [(i - n_shape_nodes) * member_length_xyz[0], 0, member_length_xyz[2]]

    for i in range(2 * n_shape_nodes, 3 * n_shape_nodes):
        node_coordinates[i, :] = [(i - 2*n_shape_nodes) * member_length_xyz[0], member_length_xyz[1], 0]

    for i in range(3 * n_shape_nodes, 4 * n_shape_nodes):
        node_coordinates[i, :] = [(i - 3*n_shape_nodes) * member_length_xyz[0],
                                  member_length_xyz[1],
                                  member_length_xyz[2]]

    element_connectivity = []
    current_member_indx = 0  # The row index of the connectivity matrix represents the member number
    # Different types of members: Straight, slanted, cross-connected. The axis labels at the end denote the axis or
    # plane on which the members lie
    member_groups = {'straight_x': [], 'straight_xz': [], 'straight_xy': [], 'slanted_xz': [],
                     'cross_yz_end': [], 'cross_xy': []}

    # For all members aligned along x-axis
    member_groups['straight_x'].append([])
    # Straight members along x-axis y = 0, z = 0
    for i in range(n_shape_nodes - 1):
        element_connectivity.append([i, i+1, member_radius])
        member_groups['straight_x'][0].append(current_member_indx)
        current_member_indx += 1

    member_groups['straight_x'].append([])
    # Straight members along x-axis y = 0, z = 4
    for i in range(n_shape_nodes, 2*n_shape_nodes - 1):
        element_connectivity.append([i, i+1, member_radius])
        member_groups['straight_x'][1].append(current_member_indx)
        current_member_indx += 1

    member_groups['straight_x'].append([])
    # Straight members along x-axis y = 4, z = 0
    for i in range(2 * n_shape_nodes, 3*n_shape_nodes - 1):
        element_connectivity.append([i, i+1, member_radius])
        member_groups['straight_x'][2].append(current_member_indx)
        current_member_indx += 1

    member_groups['straight_x'].append([])
    # Straight members along x-axis y = 4, z = 4
    for i in range(3 * n_shape_nodes, 4*n_shape_nodes - 1):
        element_connectivity.append([i, i+1, member_radius])
        member_groups['straight_x'][3].append(current_member_indx)
        current_member_indx += 1

    member_groups['cross_yz_end'].append([])
    # For all the cross connected members along the y-z plane at each end
    # Along y-z plane and x = 0
    element_connectivity.append([0, 3*n_shape_nodes, member_radius])
    member_groups['cross_yz_end'][0].append(current_member_indx)
    current_member_indx += 1
    element_connectivity.append([n_shape_nodes, 2 * n_shape_nodes, member_radius])
    member_groups['cross_yz_end'][0].append(current_member_indx)
    current_member_indx += 1

    member_groups['cross_yz_end'].append([])
    # Along y-z plane and x = 72
    element_connectivity.append([n_shape_nodes - 1, 4*n_shape_nodes - 1, member_radius])
    member_groups['cross_yz_end'][1].append(current_member_indx)
    current_member_indx += 1
    element_connectivity.append([2*n_shape_nodes - 1, 3*n_shape_nodes - 1, member_radius])
    member_groups['cross_yz_end'][1].append(current_member_indx)
    current_member_indx += 1

    # For all connections along y and z axes
    member_groups['straight_xz'].append([])
    member_groups['straight_xy'].append([])
    for i in range(n_shape_nodes):
        element_connectivity.append([i, i + n_shape_nodes, member_radius])
        member_groups['straight_xz'][0].append(current_member_indx)
        current_member_indx += 1
        element_connectivity.append([i, i + 2*n_shape_nodes, member_radius])
        member_groups['straight_xy'][0].append(current_member_indx)
        current_member_indx += 1
    member_groups['straight_xz'].append([])
    member_groups['straight_xy'].append([])
    for i in range(3 * n_shape_nodes, 4 * n_shape_nodes):
        element_connectivity.append([i, i - n_shape_nodes, member_radius])
        member_groups['straight_xz'][1].append(current_member_indx)
        current_member_indx += 1
        element_connectivity.append([i, i - 2*n_shape_nodes, member_radius])
        member_groups['straight_xy'][1].append(current_member_indx)
        current_member_indx += 1

    # For all members cross-connected along x-axis
    member_groups['cross_xy'].append([])
    for i in range(n_shape_nodes - 1):
        element_connectivity.append([i, i + 2*n_shape_nodes + 1, member_radius])
        member_groups['cross_xy'][0].append(current_member_indx)
        current_member_indx += 1

    member_groups['cross_xy'].append([])
    for i in range(n_shape_nodes, 2*n_shape_nodes - 1):
        element_connectivity.append([i, i + 2*n_shape_nodes + 1, member_radius])
        member_groups['cross_xy'][1].append(current_member_indx)
        current_member_indx += 1

    member_groups['cross_xy'].append([])
    for i in range(2 * n_shape_nodes, 3*n_shape_nodes - 1):
        element_connectivity.append([i, i - 2*n_shape_nodes + 1, member_radius])
        member_groups['cross_xy'][2].append(current_member_indx)
        current_member_indx += 1

    member_groups['cross_xy'].append([])
    for i in range(3 * n_shape_nodes, 4 * n_shape_nodes - 1):
        element_connectivity.append([i, i - 2*n_shape_nodes + 1, member_radius])
        member_groups['cross_xy'][3].append(current_member_indx)
        current_member_indx += 1

    # For all members cross-connected along z-axis
    member_groups['slanted_xz'].append([])
    for i in range(n_shape_nodes, n_shape_nodes + n_shape_nodes//2):
        element_connectivity.append([i - n_shape_nodes + 1, i, member_radius])
        member_groups['slanted_xz'][0].append(current_member_indx)
        current_member_indx += 1

    member_groups['slanted_xz'].append([])
    for i in range(n_shape_nodes + n_shape_nodes//2 + 1, 2*n_shape_nodes):
        element_connectivity.append([i - n_shape_nodes - 1, i, member_radius])
        member_groups['slanted_xz'][1].append(current_member_indx)
        current_member_indx += 1

    member_groups['slanted_xz'].append([])
    for i in range(3 * n_shape_nodes, 3 * n_shape_nodes + n_shape_nodes//2):
        element_connectivity.append([i - n_shape_nodes + 1, i, member_radius])
        member_groups['slanted_xz'][2].append(current_member_indx)
        current_member_indx += 1

    member_groups['slanted_xz'].append([])
    for i in range(3 * n_shape_nodes + n_shape_nodes//2 + 1, 4 * n_shape_nodes):
        element_connectivity.append([i - n_shape_nodes - 1, i, member_radius])
        member_groups['slanted_xz'][3].append(current_member_indx)
        current_member_indx += 1

    element_connectivity = np.array(element_connectivity)
    # Uncomment following line to have beam numbers start from 1 instead of 0
    element_connectivity[:, :2] = element_connectivity[:, :2] + 1

    return node_coordinates, element_connectivity, fixed_nodes, load_nodes, member_groups


if __name__ == '__main__':
    coordinates, connectivity, fixed_nodes, load_nodes, member_groups = gen_truss(n_shape_nodes=19)
    draw_truss(coordinates, connectivity, fixed_nodes, load_nodes)
