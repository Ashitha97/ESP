"""
Various fucntions for visualizations
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def dyna_plot(poly_arr, name, fill=False, color=None, fig_labels=None, set_axis=None, save=None, show=True, legend=True):
    """
    Plots Dynamometer cards
    :param poly_arr: An array of polygons that need to be plotted
    :param name: Name of the plot (If being used will be used as the file name)
    :param fill: Fill the plots(Boolean, Default: False)
    :param color: Array of color codes of the cards specified in poly_arr. (Default: None, will generate random colors)
    :param fig_labels: Labels of the cards being plotted in poly_arr. (Default: None, will generate 0,1,2..)
    :param set_axis: Set a predetermined axis(eg: (-1000, 1000)). Default: None
    :param save: Path where the plots need to be saved. IF not specified will not save the cards
    :param show: Show the cards(Boolean, Default: False)
    :param legend: Show a legend(Boolean, Default: True)
    :return: None
    """

    if color is None:
        cmap = plt.cm.get_cmap('tab20b', len(poly_arr))  # Gets a listed cmap with standard mpl cmap name
        color = cmap.colors

    if fig_labels is None:  # if fig_labels not provided
        fig_labels = range(len(poly_arr))

    fig_clean, ax_clean = plt.subplots()  # set up the figure

    for i in range(len(poly_arr)):  # Iterate over each of the polygons in the poly arr
        xy = np.asarray(poly_arr[i].exterior.coords)  # Get co-ordinates

        if fill:
            ax_clean.fill(xy[:, 0], xy[:, 1], facecolor=color[i], label=str(fig_labels[i]))
        else:
            ax_clean.plot(xy[:, 0], xy[:, 1], c=color[i], label=str(fig_labels[i]))

    if set_axis:
        ax_clean.set_ylim(set_axis)

    #     plt.axis('off')
    fig_clean.set_size_inches(12, 8)
    fig_clean.patch.set_facecolor('w')
    plt.title(name)

    if legend:
        plt.legend(loc='best')

    if save:
        save_name = os.path.join(save, name + ".png")
        plt.savefig(save_name, dpi=50, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()