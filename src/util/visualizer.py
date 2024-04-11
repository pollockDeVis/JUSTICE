"""
This module contains methods to visualize different kinds of output data from the JUSTICE model.
"""

from ema_workbench.analysis import plotting, Density, parcoords
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from src.util.model_time import TimeHorizon
from src.util.enumerations import *
import pickle
from sklearn.preprocessing import MinMaxScaler


def process_input_data_for_tradeoff_plot(
    input_data,
    path_to_data,
    number_of_objectives,
    objective_of_interest,
    scaling=True,
    scaling_index=0,
    feature_adjustment_value=300,
):
    data = pd.DataFrame()
    output_file_name = ""
    sliced_data = {}
    filtered_idx = 0
    filtered_data = np.zeros((number_of_objectives, number_of_objectives))

    data_length = np.zeros(len(input_data))
    for index, file in enumerate(input_data):
        _read_data = pd.read_csv(path_to_data + "/" + file)
        # Keep only the last objective columns
        _read_data = _read_data.iloc[:, -number_of_objectives:]

        if scaling:
            # Scale the data, slice the data and filter the data
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Adjust the feature values if any values are over feature_adjustment_value, subtract feature_adjustment_value for the scaling_index
            if np.max(_read_data.iloc[:, scaling_index]) > feature_adjustment_value:
                _read_data.iloc[:, scaling_index] = (
                    _read_data.iloc[:, scaling_index] - feature_adjustment_value
                )

            _read_data = pd.DataFrame(
                scaler.fit_transform(_read_data), columns=_read_data.columns
            )  # .iloc[:, scaling_index]

        data = pd.concat([data, _read_data])
        output_file_name = output_file_name + file.split(".")[0] + "_"

        data_length[index] = int(data.shape[0])

        sliced_data[index] = _read_data

        filtered_idx = sliced_data[index].iloc[:, objective_of_interest].idxmin()
        filtered_data[index, :] = _read_data.iloc[filtered_idx]

    # Convert filtered data to a dataframe
    filtered_data = pd.DataFrame(filtered_data, columns=data.columns)

    return data, data_length, output_file_name, sliced_data, filtered_data


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
    show_best_solutions=True,
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
            data, data_length, output_file_name, sliced_data, filtered_data = (
                process_input_data_for_tradeoff_plot(
                    input_data,
                    path_to_data,
                    number_of_objectives,
                    objective_of_interest,
                )
            )

        else:
            data = pd.read_csv(path_to_data + "/" + input_data)

            output_file_name = input_data

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
    for i in range(len(data_length)):

        # end_index = int(data_length[i])

        _sliced_data = sliced_data[i]
        _sliced_data.columns = data.columns

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

    # Add the column labels
    filtered_data.columns = data.columns

    if show_best_solutions:
        # Loop through the best solutions dataframe and plot them
        for j in range(filtered_data.shape[0]):
            axes.plot(
                filtered_data.iloc[j],
                color=color_palette[j],
                linewidth=2.5,
                alpha=0.8,
            )

    # Add the legend
    axes.legend()

    # Save the figure
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    plt.savefig(path_to_output + "/" + output_file_name, dpi=300)


def plot_timeseries(
    figsize=(15, 10),
    set_style="white",
    colourmap="bright",
    path_to_data="data/reevaluation",
    path_to_output="./data/plots",
    fontsize=15,
    x_label="Years",
    y_label="Temperature Rise (°C)",
    variable_name="global_temperature",
    no_of_ensembles=1001,
    input_data=[],
    output_titles=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"],
    main_title="Global Temperature Rise - ",
    yaxis_lower_limit=0,
    yaxis_upper_limit=10,
    alpha=0.1,
    linewidth=2.5,
    lower_percentile=0,
    upper_percentile=100,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
):

    # Assert if input_data list is empty
    assert input_data, "No input data provided for visualization."

    # Set color palette
    color_palette = sns.color_palette(colourmap)

    # Set the time horizon
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    list_of_years = time_horizon.model_time_horizon
    columns = list_of_years
    data_scenario = np.zeros((len(Scenario), len(list_of_years), no_of_ensembles))
    ssp_rcp_string_list = Scenario.get_ssp_rcp_strings()

    # Loop through the input data and plot the timeseries
    for plotting_idx, file in enumerate(input_data):
        # Load the scenario data from the pickle file
        with open(path_to_data + "/" + file, "rb") as f:  # input_data[plotting_idx]
            scenario_data = pickle.load(f)

        for idx, scenarios in enumerate(list(Scenario.__members__.keys())):
            data_scenario[idx, :, :] = scenario_data[scenarios][variable_name]

        mean_data = np.zeros((len(Scenario), len(list_of_years)))
        sns.set_style(set_style)
        fig, ax = plt.subplots(figsize=figsize)

        # Set y-axis limits
        plt.ylim(yaxis_lower_limit, yaxis_upper_limit)

        for idx, scenarios in enumerate(list(Scenario.__members__.keys())):

            print(scenarios)
            temp_df = pd.DataFrame(data_scenario[idx, :, :].T, columns=list_of_years)
            print(temp_df.max().max())
            label = ssp_rcp_string_list[idx]
            color = color_palette[idx]  # colors[idx]

            # Calculate the percentiles
            p_l = np.percentile(temp_df, lower_percentile, axis=0)
            p_h = np.percentile(temp_df, upper_percentile, axis=0)

            # Calculate the mean
            mean_data[idx, :] = temp_df.mean()

            # Plot percentiles as bands
            ax.fill_between(list_of_years, p_l, p_h, color=color, alpha=alpha)

        # Convert the mean_data to a dataframe
        mean_data = pd.DataFrame(mean_data, columns=list_of_years)

        for i in range(mean_data.shape[0]):
            label = ssp_rcp_string_list[i]
            color = color_palette[i]
            sns.lineplot(
                data=mean_data.iloc[i],
                color=color,
                alpha=alpha * 8,
                linewidth=linewidth,
                label=label,
                ax=ax,
            )
        # Access the current Axes instance on the current figure:

        ax = plt.gca()

        # Remove top and right border
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Remove top and right ticks
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        # Set font size of axis labels
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Add labels, legend and title
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.legend(loc="upper left", fontsize=fontsize)
        plt.title(main_title + output_titles[plotting_idx], fontsize=fontsize)

        # Save the figure
        if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)

        # if output_file_name is None:
        output_file_name = file.split(".")[0]  # input_data[plotting_idx]

        plt.savefig(
            path_to_output + "/" + output_file_name + "_" + output_titles[plotting_idx],
            dpi=300,
        )


def plot_ssp_rcp_subplots():

    ssp_rcp_string_list = [
        "SSP1-RCP1.9",
        "SSP1-RCP2.6",
        "SSP2-RCP4.5",
        "SSP3-RCP7.0",
        "SSP4-RCP3.4",
        "SSP4-RCP6.0",
        "SSP5-RCP3.4-overshoot",
        "SSP5-RCP8.5",
    ]

    scenario = list(Scenario)
    # Color Mapping
    colors = ["red", "orange", "green", "blue", "indigo"]

    # Time Horizon Setup
    time_horizon = TimeHorizon(
        start_year=2015, end_year=2300, data_timestep=5, timestep=1
    )
    list_of_years = time_horizon.model_time_horizon

    # rice50_temp = pd.DataFrame(interpolated_TATM, columns=list_of_years)
    # damages_array_summed = np.sum(damages_array, axis=1)
    # damages_array_summed = pd.DataFrame(damages_array_summed, columns=list_of_years)
    # Sum the damages for all regions
    rice50_damages = np.sum(interpolated_damages, axis=1)
    # Use the list of years as x-axis and the interpolated damages for each scenario as y-axis
    rice50_damages = pd.DataFrame(rice50_damages, columns=list_of_years)

    # Create subplots in grid of 4 rows and 2 columns
    fig, axs = plt.subplots(2, 4, figsize=(25, 12))

    # Reshape axs to 1D for easy iteration
    axs = axs.ravel()

    # find overall min and max temperatures (5th and 95th percentile respectively) amongst all data
    global_min = np.min(
        [
            np.percentile(np.sum(damages_array_sorted[i], axis=1), 5, axis=0)
            for i in range(8)
        ]
    )
    global_max = np.max(
        [
            np.percentile(np.sum(damages_array_sorted[i], axis=1), 95, axis=0)
            for i in range(8)
        ]
    )

    for i in range(8):
        # calculate 5th and 95th percentiles
        p_5 = np.percentile(np.sum(damages_array_sorted[i], axis=1), 5, axis=0)
        p_95 = np.percentile(np.sum(damages_array_sorted[i], axis=1), 95, axis=0)

        # Get the economic scenario corresponding to the index
        idx = get_economic_scenario(i)

        # Plot percentiles as bands
        axs[i].fill_between(list_of_years, p_5, p_95, color=colors[idx], alpha=0.2)

        # Plot
        sns.lineplot(data=rice50_damages.iloc[idx, :], color=colors[idx], ax=axs[i])

        # Styling each subplot
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].xaxis.set_ticks_position("bottom")
        axs[i].yaxis.set_ticks_position("left")

        # axs[i].set_xlabel('Year')
        axs[i].set_ylabel("Economic Damages Trillion $")
        axs[i].legend([f"SSP {idx+1}"], loc="upper left")
        # Set title for each subplot
        axs[i].set_title(ssp_rcp_string_list[i])  # (scenario[i].value[2])
        # Set title font size
        axs[i].title.set_size(20)

        axs[i].set_ylim(global_min, global_max)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_timeseries(
        path_to_data="data/reevaluation",
        path_to_output="./data/plots",
        x_label="Years",
        y_label="Temperature Rise (°C)",
        variable_name="global_temperature",
        input_data=[
            "UTIL_100049.pkl",
            "EGAL_101948.pkl",
            "PRIOR_101765.pkl",
            "SUFF_102924.pkl",
        ],
        output_titles=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"],
        main_title="Global Temperature Rise - ",
        yaxis_lower_limit=0,
        yaxis_upper_limit=10,
        alpha=0.1,
        linewidth=2.5,
    )
