import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.markers import MarkerStyle
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height)
    return p


def draw_truss(node_coordinates, element_connectivity, fixed_nodes, load_nodes, suppress_axis=False, grid=True,
               force_arrows=True, highlight_group=False):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plot nodes
    node_plt = ax1.scatter(node_coordinates[:, 0], node_coordinates[:, 1], node_coordinates[:, 2],
                marker=MarkerStyle(marker='o'), c='b', s=15, label="Node", alpha=0.5)

    # Plot fixed nodes
    # ax1.scatter(node_coordinates[fixed_nodes - 1, 0],
    #             node_coordinates[fixed_nodes - 1, 1],
    #             node_coordinates[fixed_nodes - 1, 2], marker=MarkerStyle(marker='s', fillstyle='none'), s=200, c='r')
    fixnode_plt = ax1.scatter(node_coordinates[fixed_nodes - 1, 0],
                node_coordinates[fixed_nodes - 1, 1],
                node_coordinates[fixed_nodes - 1, 2] - 0.3,
                marker=MarkerStyle(marker='^', fillstyle='full'), s=100, c='b', label="Fixed support")

    # Plot force
    max_z = np.max(node_coordinates[:, 2])
    xy_nodes = node_coordinates[:, :2]
    # ax1.quiver(xy_nodes[:, 0], xy_nodes[:, 1], max_z*np.ones_like(xy_nodes[:, 0]) + 3,
    #            np.zeros_like(xy_nodes[:, 0]), np.zeros_like(xy_nodes[:, 0]), -4 * np.ones_like(xy_nodes[:, 0]),
    #            length=2.8, normalize=True, arrow_length_ratio=0.3, color='r', linewidth=2, label="Force")
    if force_arrows:
        for i, c in enumerate(node_coordinates):
            arw = Arrow3D([c[0], c[0]], [c[1], c[1]], [max_z + 3, max_z + 0.2], arrowstyle="-|>",
                          color="red", lw=1,
                          mutation_scale=25, zorder=3, label="Force", alpha=1)
            # if i == (len(node_coordinates) - 1):
            arw.set_label("Force (10 kN)")
            ax1.add_artist(arw)
            # fig1.patches.append(arw)

    # Plot elements
    for i, nodes in enumerate(element_connectivity):
        if i == len(element_connectivity) - 1:
            l = "Ungrouped members"
        else:
            l = None
        linewidth = 1
        c = 'k'

        zorder = 1
        elem_plt = ax1.plot([node_coordinates[int(nodes[0] - 1), 0], node_coordinates[int(nodes[1] - 1), 0]],
                            [node_coordinates[int(nodes[0] - 1), 1], node_coordinates[int(nodes[1] - 1), 1]],
                            [node_coordinates[int(nodes[0] - 1), 2], node_coordinates[int(nodes[1] - 1), 2]],
                            c=c, linewidth=linewidth, label=l, alpha=1)

    # KLUGE
    if highlight_group and len(element_connectivity) == 260:
        for i in range(len(element_connectivity) - 1, -1, -1):
            nodes = element_connectivity[i]
            l = None
            if (76 <= i < 114 and i % 2 == 0) or (114 <= i < 152 and i % 2 == 0):
                linewidth = 4
                c = 'm'
                if i == 76:
                    l = "Vertical member ($G_1, G_4$)"
            elif 18 <= i < 36 or 54 <= i < 72:
                linewidth = 4
                c = 'c'
                if i == 18:
                    l = "Top longitudinal member ($G_2$)"
            elif 0 <= i < 18 or 36 <= i < 54:
                linewidth = 4
                c = 'y'
                if i == 0:
                    l = "Bottom longitudinal member ($G_3$)"
            else:
                continue
            zorder = 3
            grp_plt = ax1.plot([node_coordinates[int(nodes[0] - 1), 0], node_coordinates[int(nodes[1] - 1), 0]],
                               [node_coordinates[int(nodes[0] - 1), 1], node_coordinates[int(nodes[1] - 1), 1]],
                               [node_coordinates[int(nodes[0] - 1), 2], node_coordinates[int(nodes[1] - 1), 2]],
                               c=c, linewidth=linewidth, label=l, alpha=0.7)

    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.set_zlabel('z', fontsize=10)
    # ax1.set_xlim(-5, np.max(node_coordinates[:, 0] + 20))
    ax1.set_xlim(-0, np.max(node_coordinates[:, 0] + 1))
    ax1.set_ylim(-5, np.max(node_coordinates[:, 1] + 10))
    ax1.set_zlim(np.min(node_coordinates[:, 2]) - 5, np.max(node_coordinates[:, 2] + 5))

    if not grid:
        ax1.grid(False)

    if suppress_axis:
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.set_axis_off()

    h, l = ax1.get_legend_handles_labels()
    handler_arr, label_arr = [], []
    # handler_arr = [node_plt, elem_plt[0], fixnode_plt]
    # label_arr = ["Node", "Members", "Fixed support"]
    handler_arr += h
    label_arr += l
    if force_arrows:
        handler_arr.append(arw)
        label_arr.append("Force")
    ax1.legend(handler_arr, label_arr,
               handler_map={Arrow3D: HandlerPatch(patch_func=make_legend_arrow),
                            },
               fontsize=17
               )
    # ax1.legend([plt.arrow()], ["arr"])
    fig1.tight_layout()

    return fig1, ax1


if __name__ == '__main__':
    from truss.generate_truss import gen_truss
    # coordinates, connectivity, fixed_nodes, load_nodes = gen_truss()
    # draw_truss(coordinates, connectivity, fixed_nodes, load_nodes)
    #
    # coordinates, connectivity, fixed_nodes, load_nodes = gen_truss(n_shape_nodes=39)
    # draw_truss(coordinates, connectivity, fixed_nodes, load_nodes)

    # For truss_20_xyz - 20 shape var, 3 force
    num_shape_vars = 20
    node_coordinates, connectivity, fixed_nodes, load_nodes, member_groups = gen_truss(n_shape_nodes=num_shape_vars * 2 - 1)
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
    node_coordinates[0:num_shape_vars, 2] = z
    node_coordinates[(2 * num_shape_vars - 1) * 2:(2 * num_shape_vars - 1) * 2 + num_shape_vars, 2] = z
    node_coordinates[num_shape_vars:2 * num_shape_vars - 1, 2] = np.flip(z[:-1])
    node_coordinates[(2 * num_shape_vars - 1) * 2 + num_shape_vars:(2 * num_shape_vars - 1) * 2 + 2 * num_shape_vars - 1, 2] = np.flip(z[:-1])
    truss_fig, truss_ax = draw_truss(node_coordinates, connectivity, fixed_nodes, load_nodes, suppress_axis=False)

    plt.show()
