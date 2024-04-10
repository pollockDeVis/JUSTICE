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
from matplotlib.ticker import MaxNLocator

from matplotlib.colors import to_hex


def preprocess_data(
    data, number_of_objectives, data_length, direction="min", objective_of_interest=0
):
    """
    Preprocess the data to be visualized.
    """

    # Best solutions array
    # best_solutions_indices = np.zeros((number_of_objectives, number_of_objectives))
    best_solutions_indices = np.zeros((number_of_objectives))

    # Keep only the last objective columns
    data = data.iloc[:, -number_of_objectives:]

    # Initialize the start index
    start_index = 0
    # Loop through data_length
    for i in range(len(data_length)):
        end_index = int(data_length[i])
        print(start_index, end_index)
        # Get the best solutions
        if direction == "min":
            sliced_data = data.iloc[start_index:end_index, :]
            # Find the min value for first objective
            best_solutions_indices[i] = sliced_data.iloc[
                :, objective_of_interest
            ].idxmin()

            # Find the index for the min value for each column in sliced_data

            # best_solutions_indices[i, :] = sliced_data.idxmin()  # min(axis=0)

        elif direction == "max":
            pass
            # sliced_data = data.iloc[start_index:end_index]
            # # Find the index for the max value for each column in sliced_data
            # idx = sliced_data.idxmax()
            # best_solutions_indices[i] = data.iloc[idx]

        start_index = end_index + 1

    # Convert the best_solutions array to a dataframe with the same columns as data
    # best_solutions_indices = pd.DataFrame(best_solutions_indices, columns=data.columns)

    return data, best_solutions_indices


def visualize_tradeoffs(
    figsize=(15, 10),
    set_style="whitegrid",
    font_scale=1.8,
    number_of_objectives=4,
    colourmap="Set2",
    linewidth=0.4,
    alpha=0.05,
    path_to_data=None,
    path_to_output="./data/plots",
    output_file_name="",
    objective_of_interest=0,
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

    for index, file in enumerate(input_data):
        output_file_name = output_file_name + file.split(".")[0] + "_"

    # Preprocess the data
    # data, best_solutions_indices = preprocess_data(
    #     data, number_of_objectives, data_length, direction="min"
    # )
    # # Keep only the last objective columns
    data = data.iloc[:, -number_of_objectives:]

    limits = parcoords.get_limits(data)
    axes = parcoords.ParallelAxes(limits)

    color_palette = sns.color_palette(colourmap)

    # Plot the data and save the figure
    start_index = 0
    for i in range(len(data_length)):

        end_index = int(data_length[i])

        _sliced_data = data.iloc[start_index:end_index]
        best_solutions_indices = _sliced_data.iloc[:, objective_of_interest].idxmin()
        print(best_solutions_indices)

        axes.plot(
            _sliced_data,
            color=color_palette[i],
            linewidth=linewidth,
            alpha=alpha,
            label=input_data[i].split(".")[0],
        )

        axes.plot(
            _sliced_data.iloc[best_solutions_indices],
            color=color_palette[i],
            linewidth=2,
            alpha=1,
        )
        start_index = end_index + 1

        # for j in range(best_solutions_indices.shape[1]):
        #     idx = int(best_solutions_indices[i, j])
        #     axes.plot(
        #         data.iloc[idx],
        #         color=color_palette[i],
        #         linewidth=2,
        #         alpha=1,
        #     )
        # Plot the best solutions
        # axes.plot(best_solutions_indices.iloc[i], color=color_palette[i], linewidth=2, alpha=1)
        # print(i)

    # for j in range(best_solutions_indices.shape[0]):
    #     axes.plot(
    #         data.iloc[int(best_solutions_indices[j])],
    #         color=color_palette[j],
    #         linewidth=2,
    #         alpha=1,
    #     )

    # Add the legend
    axes.legend()

    # Save the figure
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    plt.savefig(path_to_output + "/" + output_file_name, dpi=300)
