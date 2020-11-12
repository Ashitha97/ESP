"""
Various functions visualizations
"""
import matplotlib.pyplot as plt
import numpy as np
import os
# plotly imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def plot_features(df, well_name, fail_col, feature_cols, zero_label='Normal', mov_avg=None):
    """
    Plots the features and failures of a specific well
    :param df: The data frame we need to use
    :param well_name: Name of the well
    :param fail_col: Failure Column to be considered
    :param feature_cols: Columns to plot as features (Should be numerical)
    :param zero_label: The label which shows normal working condition, wont be plotted
    :param mov_avg: Plot Moving Averages if needed (Default: None)
    """

    # get the specific well
    df_well = df[df.NodeID == well_name].reset_index(drop=True)

    # get all the unique failures from the failure col
    fail = df_well[fail_col].unique()
    fail = fail[fail != zero_label]

    # Get only features (for mov_averages)
    if mov_avg is not None:
        df_feature = df_well.set_index('Date')[feature_cols].rolling(mov_avg).mean()
    else:
        df_feature = df_well.set_index("Date")

    # set up the figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])  # secondary y_axis for failures

    # plot features
    for c in feature_cols:
        fig.add_trace(go.Scatter(x=df_feature.index, y=df_feature[c], mode='lines', name=c), secondary_y=False)

    # Plot failures
    for f in fail:
        temp_fail = df_well[fail_col].map(lambda x: 1 if x == f else 0)
        fig.add_trace(go.Scatter(x=df_well.Date,
                                 y=temp_fail,
                                 line={
                                     'width': 0,
                                     'shape': 'hv'
                                 },
                                 fill='tozeroy',
                                 name=f), secondary_y=True)

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(template="seaborn", title=well_name + " with MA of :" + str(mov_avg), autosize=True)
    fig.update_yaxes(title_text="Features (KPI)", secondary_y=False)
    fig.update_yaxes(title_text="Failure", secondary_y=True)

    return fig.show()
