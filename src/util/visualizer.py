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


def visualize_tradeoffs(
    figsize=(15, 10),
    set_style="whitegrid",
    font_scale=1.8,
    number_of_objectives=4,
    colourmap="Spectral",
    linewidth=0.8,
    alpha=0.2,
    path_to_data=None,
    path_to_output="./data/plots",
    output_file_name="",
    **kwargs
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
            data = pd.concat(
                [pd.read_csv(path_to_data + "/" + file) for file in input_data]
            )
            # output_file_name = "_".join(input_data)
            # output_file_name = output_file_name.split(".")[0]

        else:
            data = pd.read_csv(path_to_data + "/" + input_data)

            output_file_name = input_data

    for file in input_data:
        output_file_name = output_file_name + file.split(".")[0] + "_"

    print(output_file_name)

    # Keep only the last objective columns
    data = data.iloc[:, -number_of_objectives:]

    limits = parcoords.get_limits(data)
    axes = parcoords.ParallelAxes(limits)
    # Define the number of lines
    num_lines = len(data)

    # Define the color spectrum palette
    color_palette = cm.get_cmap(colourmap)(np.linspace(0, 1, num_lines))

    # Plot the data and save the figure
    for i in range(num_lines):
        axes.plot(
            data.iloc[i], color=color_palette[i], linewidth=linewidth, alpha=alpha
        )

    # Save the figure
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    plt.savefig(path_to_output + "/" + output_file_name, dpi=300)
