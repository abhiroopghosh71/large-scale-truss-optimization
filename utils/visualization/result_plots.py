import matplotlib.pyplot as plt
import numpy as np


def plot_obj(pf_list, axes_limits=None, title=None, plot_labels=None, axis_labels=("f1", "f2"), show_grid=False):
    """Plot multiple populations in the objective space."""
    # TODO: Extend function to 3 obj
    fig_obj = plt.figure()
    ax1_obj = fig_obj.add_subplot(111)

    if title is not None:
        fig_obj.suptitle(title)
    ax1_obj.set_xlabel(axis_labels[0])
    ax1_obj.set_ylabel(axis_labels[1])

    min_limit = [np.inf, np.inf]
    max_limit = [-np.inf, -np.inf]
    for i, pf in enumerate(pf_list):
        for j in range(pf.shape[1]):
            if np.min(pf[:, j]) < min_limit[j]:
                min_limit[j] = np.min(pf[:, j])
            if np.max(pf[:, j]) > max_limit[j]:
                max_limit[j] = np.max(pf[:, j])
        if (plot_labels is not None) and i < len(plot_labels):
            label = plot_labels[i]
        else:
            label = f"data {i + 1}"
        ax1_obj.scatter(pf[:, 0], pf[:, 1], label=label)

    if axes_limits is not None:
        ax1_obj.set_xlim([axes_limits[0, 0], axes_limits[1, 0]])
        ax1_obj.set_ylim([axes_limits[0, 1], axes_limits[1, 1]])
    ax1_obj.legend()
    if show_grid:
        ax1_obj.grid()

    return fig_obj, ax1_obj


def plot_hv(hv_list, axes_limits=None, title=None, plot_labels=None,
            axis_labels=("Function Evaluations", "Hypervolume (HV)"), show_grid=False):
    """Plot Hypervolume (HV) plots for multiple experiments."""

    fig_hv = plt.figure()
    ax_hv = fig_hv.add_subplot(111)

    if title is not None:
        fig_hv.suptitle(title)
    ax_hv.set_xlabel(axis_labels[0])
    ax_hv.set_ylabel(axis_labels[1])

    for i, hv in enumerate(hv_list):
        if (plot_labels is not None) and i < len(plot_labels):
            label = plot_labels[i]
        else:
            label = f"data {i + 1}"
        ax_hv.plot(hv[:, 0], hv[:, 1], label=label)

    if axes_limits is not None:
        ax_hv.set_xlim([axes_limits[0, 0], axes_limits[1, 0]])
        ax_hv.set_ylim([axes_limits[0, 1], axes_limits[1, 1]])
    ax_hv.legend()
    if show_grid:
        ax_hv.grid()

    return fig_hv, ax_hv


if __name__ == '__main__':
    test_pf_list = np.array([np.array([[3, 4], [2, 5], [1, 6]]),
                            np.array([[7, 8], [5, 9], [2, 7]]),
                            np.array([[3, 5], [2, 6], [3, 8]])])

    plot_obj(test_pf_list, axes_limits=np.array([[0, 0], [10, 10]]), title="Sample plot",
             plot_labels=["Result 1", "Result 2"], axis_labels=("Weight (kg)", "Compliance (m/N)"))

    test_hv_list = np.array([np.array([[10, 0.1], [20, 0.3], [30, 0.5], [40, 0.9], [50, 1.05]]),
                             np.array([[10, 0.3], [20, 0.5], [30, 0.8], [40, 0.8]]),
                             np.array([[5, 0.2], [37, 0.7], [49, 0.85], [88, 1.13]])])

    plot_hv(test_hv_list, axes_limits=np.array([[0, 0], [100, 1.5]]), title="Sample HV plot",
            plot_labels=["Result 1", "Result 2"], axis_labels=("Generations", "Hypervolume (HV)"))
