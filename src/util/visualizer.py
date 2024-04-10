"""
This module contains methods to visualize different kinds of output data from the JUSTICE model.
"""

import matplotlib
import numpy as np

from ema_workbench.analysis import plotting, Density, parcoords
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
import matplotlib.cm as cm
from matplotlib.colors import to_rgb


def process_input_data(input_data, path_to_data):
    data = pd.DataFrame()
    output_file_name = ""

    data_length = np.zeros(len(input_data))
    for index, file in enumerate(input_data):
        data = pd.concat([data, pd.read_csv(path_to_data + "/" + file)])
        output_file_name = output_file_name + file.split(".")[0] + "_"
        data_length[index] = int(data.shape[0])
    return output_file_name


def visualize_tradeoffs(
    figsize=(15, 10),
    set_style="whitegrid",
    font_scale=1.8,
    number_of_objectives=4,
    colourmap="bright",
    linewidth=0.4,
    alpha=0.05,
    path_to_data=None,
    path_to_output="./data/plots",
    output_file_name="",
    objective_of_interest=0,
    column_labels=None,
    legend_labels=None,
    axis_rotation=45,
    fontsize=15,
    **kwargs,
):
    """
    Visualize the tradeoffs between different objectives in the model.
    """
    sns.set_theme(font_scale=font_scale)
    sns.set_style(set_style)
    sns.set_theme(rc={"figure.figsize": figsize})

    # Check kwargs for input data or multiple data files
    input_data = kwargs.get("input_data", None)
    if input_data is None:
        raise ValueError("No input data provided for visualization.")

    if path_to_data is not None and input_data is not None:
        # Repeat for multiple data files and concatenate them into one dataframe
        if isinstance(input_data, list):
            data = pd.DataFrame()
            data_length = np.zeros(len(input_data))
            for index, file in enumerate(input_data):
                data = pd.concat([data, pd.read_csv(path_to_data + "/" + file)])
                output_file_name = output_file_name + file.split(".")[0] + "_"
                data_length[index] = int(data.shape[0])

        else:
            data = pd.read_csv(path_to_data + "/" + input_data)

            output_file_name = input_data

    # Redundant
    for index, file in enumerate(input_data):
        output_file_name = output_file_name + file.split(".")[0] + "_"

    # Keep only the last objective columns
    data = data.iloc[:, -number_of_objectives:]

    if column_labels is not None:
        # Check if the number of column labels is equal to the number of objectives
        if len(column_labels) != number_of_objectives:
            raise ValueError(
                "Number of column labels provided does not match the number of objectives."
            )
        data.columns = column_labels

    limits = parcoords.get_limits(data)
    axes = parcoords.ParallelAxes(limits, rot=axis_rotation, fontsize=fontsize)

    # Getting the matplotlib axes object from Parcoords for annotation
    matplotlib_ax = axes.axes[0]

    # Add annotation text on the plot. write on the left side of the plot: "Direction of Preference -->"
    matplotlib_ax.annotate(
        "$\\leftarrow$ Direction of Preference",  # $\\rightarrow$"
        xy=(0.0, 0.5),
        xytext=(-0.05, 0.5),  # Adjust the xytext to move the text more to the left
        xycoords="axes fraction",
        fontsize=15,
        ha="center",
        va="center",
        color="black",
        rotation=90,
        # arrowprops=dict(arrowstyle="<-", color="black", lw=1.5),
    )

    color_palette = sns.color_palette(colourmap)

    # Plot the data and save the figure
    start_index = 0
    best_solutions = np.zeros((len(input_data), number_of_objectives))
    for i in range(len(data_length)):

        end_index = int(data_length[i])

        _sliced_data = data.iloc[start_index:end_index]
        best_solutions_indices = _sliced_data.iloc[:, objective_of_interest].idxmin()
        labels = []
        if legend_labels is not None:
            labels = legend_labels
        else:
            labels = input_data

        axes.plot(
            _sliced_data,
            color=color_palette[i],
            linewidth=linewidth,
            alpha=alpha,
            label=labels[i].split(".")[0],  # Splitting in case of file extension
        )
        best_solutions[i, :] = _sliced_data.iloc[best_solutions_indices]

        start_index = end_index + 1

    best_solutions_df = pd.DataFrame(best_solutions, columns=data.columns)

    # Loop through the best solutions dataframe and plot them
    for j in range(best_solutions_df.shape[0]):
        axes.plot(
            best_solutions_df.iloc[j],
            color=color_palette[j],
            linewidth=2,
            alpha=0.8,
        )

    # Add the legend
    axes.legend()

    # Save the figure
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    plt.savefig(path_to_output + "/" + output_file_name, dpi=300)
