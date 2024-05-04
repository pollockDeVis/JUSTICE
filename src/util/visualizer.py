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
from src.util.data_loader import DataLoader

from sklearn.preprocessing import MinMaxScaler

import json
import pycountry
import plotly.express as px


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
    filtered_data = np.zeros((number_of_objectives, number_of_objectives))

    data_length = np.zeros(len(input_data))
    for index, file in enumerate(input_data):
        _read_data = pd.read_csv(path_to_data + "/" + file)
        # Keep only the last objective columns
        _read_data = _read_data.iloc[:, -number_of_objectives:]

        if scaling:
            # Adjust the feature values if any values are over feature_adjustment_value, subtract feature_adjustment_value for the scaling_index
            # TODO: This operation happens twice. Might be good to combine the groups and apply the scaling together
            if np.max(_read_data.iloc[:, scaling_index]) > feature_adjustment_value:
                feature_range = (0.51, 1)
                _read_data.iloc[:, scaling_index] = MinMaxScaler(
                    feature_range
                ).fit_transform(_read_data.iloc[:, scaling_index].values.reshape(-1, 1))
            else:
                feature_range = (0, 0.49)
                _read_data.iloc[:, scaling_index] = MinMaxScaler(
                    feature_range
                ).fit_transform(_read_data.iloc[:, scaling_index].values.reshape(-1, 1))

        data = pd.concat([data, _read_data])
        output_file_name = output_file_name + file.split(".")[0] + "_"

        data_length[index] = int(data.shape[0])

        sliced_data[index] = _read_data

    if scaling:

        # Scale the data, slice the data and filter the data
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Actual columns
        actual_columns = data.columns
        # Get the column name for the scaling index
        scaling_column = data.columns[scaling_index]

        # Get the list of columns without the scaling column
        filtered_columns = data.columns.tolist()
        filtered_columns.remove(scaling_column)

        # Transform data except for the scaling column
        data[filtered_columns] = pd.DataFrame(
            scaler.fit_transform(data[filtered_columns]),
            # columns=data.columns,
        )
        data.columns = actual_columns

        # Create a single dataframe with all the data from sliced_data
        combined_data = pd.concat(sliced_data.values(), ignore_index=True)

        # Apply minmax scaling to the combined data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = pd.DataFrame(
            scaler.fit_transform(combined_data), columns=combined_data.columns
        )

        # Redistribute the scaled data back to sliced_data
        start_index = 0
        for index, df in sliced_data.items():
            end_index = start_index + len(df)
            sliced_data[index] = scaled_data.iloc[start_index:end_index]
            start_index = end_index

        # Iterate through sliced data and select the row with the minimum value for the objective of interest
        for index, df in sliced_data.items():

            # Reset index for sliced data
            df.reset_index(drop=True, inplace=True)

            # Get the index of the row with the minimum value for the objective of interest
            filtered_idx = df.iloc[:, objective_of_interest].idxmin()
            filtered_data[index, :] = df.iloc[filtered_idx].values

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
    scaling=True,
    scaling_index=0,
    column_labels=None,
    legend_labels=None,
    axis_rotation=45,
    fontsize=15,
    feature_adjustment_value=300,
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
                    input_data=input_data,
                    path_to_data=path_to_data,
                    number_of_objectives=number_of_objectives,
                    objective_of_interest=objective_of_interest,
                    scaling=scaling,
                    scaling_index=scaling_index,
                    feature_adjustment_value=feature_adjustment_value,
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


# TODO: Add scenario list
# These are the same thing - Redundant
# ssp_rcp_string_list = Scenario.get_ssp_rcp_strings() # These are the pretty strings
# scenario_list = ['SSP245'] #list(Scenario.__members__.keys()) # ['SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585']


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
    show_title=False,
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

        if show_title:
            plt.title(main_title + output_titles[plotting_idx], fontsize=fontsize)

        # Save the figure
        if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)

        output_file_name = variable_name

        plt.savefig(
            path_to_output + "/" + output_file_name + "_" + output_titles[plotting_idx],
            dpi=300,
        )


def process_country_data_for_choropleth_plot(
    region_list=None,
    data=None,
    list_of_years=None,
    axis_to_average=2,
    year_to_visualize=2100,
    data_label="Emission Control Rate",
    ssp_scenario=0,
    region_to_country_mapping="data/input/rice50_regions_dict.json",
    scaler=True,
    feature_scale=(0, 1),
    data_correction=True,
):
    # Assert if region list, data, and list of years are None
    assert region_list is not None, "Region list is not provided."
    assert data is not None, "Data is not provided."
    assert list_of_years is not None, "List of years is not provided."

    # Convert the region list, a numpy array, to a list
    regions_str = region_list.tolist()

    # Selecting data for the specific scenario
    data_scenario = data[ssp_scenario, :, :, :]

    # Take the mean of ensembles on the last column
    data_scenario = np.mean(data_scenario, axis=axis_to_average)
    data_scenario = pd.DataFrame(
        data_scenario, index=region_list, columns=list_of_years
    )

    # Select the year to visualize
    data_scenario_year = data_scenario[year_to_visualize]

    # Convert the index of data_scenario_year from byte string to normal string
    data_scenario_year.index = regions_str

    # Convert the data_scenario_year to a dataframe
    data_scenario_year = pd.DataFrame(data_scenario_year)

    # Change the index name to 'Region'
    data_scenario_year.index.name = "Region"

    # load region to country mapping from JSON file
    with open(region_to_country_mapping) as json_file:
        region_to_country = json.load(json_file)

    # Create a new dataframe from the mapping
    mapping_df = pd.DataFrame(
        list(region_to_country.items()), columns=["Region", "CountryCode"]
    )

    # Merge the mapping dataframe with the dataframe
    data_scenario_year = pd.merge(
        mapping_df,
        data_scenario_year.reset_index().rename(
            columns={"index": "Region", 0: data_label}
        ),
        on="Region",
    )

    data_scenario_year.columns = [
        "Region",
        "CountryCode",
        data_label,
    ]  # redundant

    # Create a new dataframe from the mapping
    data_scenario_year_by_country = []

    for idx, row in data_scenario_year.iterrows():
        for country in row["CountryCode"]:
            data_scenario_year_by_country.append([country, row[data_label]])

    # Convert list of lists to DataFrame
    data_scenario_year_by_country = pd.DataFrame(
        data_scenario_year_by_country, columns=["CountryCode", data_label]
    )

    if scaler:
        # Scale the data
        data_scenario_year_by_country[data_label] = MinMaxScaler(
            feature_scale
        ).fit_transform(data_scenario_year_by_country[data_label].values.reshape(-1, 1))

    # Create a new column 'CountryName' in data_scenario_year_by_country
    data_scenario_year_by_country["CountryName"] = data_scenario_year_by_country[
        "CountryCode"
    ].apply(
        lambda x: (
            pycountry.countries.get(alpha_3=x).name
            if pycountry.countries.get(alpha_3=x)
            else None
        )
    )

    if data_correction:
        # Check for the country code 'ATA' in the dataframe and set it to 0
        data_scenario_year_by_country.loc[
            data_scenario_year_by_country["CountryCode"] == "ATA", data_label
        ] = 0

        # Check for the CountryCode 'KSV' and set the 'CountryName' to 'Kosovo'
        data_scenario_year_by_country.loc[
            data_scenario_year_by_country["CountryCode"] == "KSV", "CountryName"
        ] = "Kosovo"

    return data_scenario_year_by_country


# TODO: Add scenario list
# These are the same thing - Redundant
# ssp_rcp_string_list = Scenario.get_ssp_rcp_strings() # These are the pretty strings
# scenario_list = ['SSP245'] #list(Scenario.__members__.keys()) # ['SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585']


def plot_choropleth(
    variable_name="constrained_emission_control_rate",
    path_to_data="data/reevaluation/",
    path_to_output="./data/plots",
    year_to_visualize=2100,
    input_data=None,
    output_titles=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"],
    title="Mitigation Burden Distribution in ",
    data_label="Emission Control Rate",
    legend_label="% Mitigation\n",
    colourmap="matter",
    projection="natural earth",
    scope="world",
    height=700,
    width=1200,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
    no_of_ensembles=1001,
    saving=False,
    scenario_list=[],
):

    # Assert if input_data list and output_titles list is None
    assert input_data, "No input data provided for visualization."
    assert output_titles, "No output titles provided for visualization."
    assert scenario_list, "No scenario list provided for visualization."

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    list_of_years = time_horizon.model_time_horizon
    data_loader = DataLoader()

    region_list = data_loader.REGION_LIST
    columns = list_of_years

    # Loop through the input data and plot the choropleth
    for plotting_idx, file in enumerate(input_data):
        # Load the scenario data from the pickle file
        with open(path_to_data + file, "rb") as f:
            scenario_data = pickle.load(f)

        data_scenario = np.zeros(
            (len(scenario_list), len(region_list), len(list_of_years), no_of_ensembles)
        )

        # Loop through all the scenarios and store the data in a 4D numpy array
        for idx, scenarios in enumerate(
            scenario_list
        ):  # list(Scenario.__members__.keys())
            data_scenario[idx, :, :, :] = scenario_data[scenarios][variable_name]

            # Process the data for choropleth plot
            data_scenario_year_by_country = process_country_data_for_choropleth_plot(
                region_list=region_list,
                data=data_scenario,
                list_of_years=list_of_years,
                year_to_visualize=year_to_visualize,
                data_label=data_label,
                ssp_scenario=idx,
            )

            choropleth_title = (
                title
                + str(year_to_visualize)
                + "-"
                + Scenario[scenarios].value[-1]  # Scenario.get_ssp_rcp_strings()[idx]
            )

            fig = px.choropleth(
                data_scenario_year_by_country,
                locations="CountryCode",
                color=data_label,
                hover_name="CountryName",
                scope=scope,
                projection=projection,
                title=choropleth_title,
                height=height,
                width=width,
                color_continuous_scale=colourmap,
            )

            # Update the layout
            fig.update_layout(
                title={
                    "text": choropleth_title,
                    "xanchor": "center",
                    "yanchor": "top",
                    "x": 0.5,
                    "y": 0.95,
                },
                coloraxis_colorbar=dict(title=legend_label),
                # coloraxis_colorbar_x=-0.1,
            )

            output_file_name = (
                variable_name
                + "_"
                + output_titles[plotting_idx]
                + "_"
                + Scenario.get_ssp_rcp_strings()[idx]
            )
            if saving:
                # Save the plot as a png file
                fig.write_image(path_to_output + "/" + output_file_name + ".png")

    # plotting_idx = 2
    return fig, data_scenario_year_by_country


# TODO: Under Construction
def plot_ssp_rcp_subplots(
    path_to_data="data/reevaluation",
    path_to_output="data/reevaluation",
    variable_name="global_temperature",
    scenario_list=[],
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
    output_file_names=[],
    colourmap="bright",
    data_loader=None,
    sns_set_style="white",
    figsize=(50, 24),
    font_scale=3,
    subplot_rows=2,
    subplot_columns=4,
    alpha=0.07,
    lower_percentile_value=5,
    upper_percentile_value=95,
    title_font_size=20,
    main_title="",
    yaxis_lower_limit=0,
    yaxis_upper_limit=8,
):

    # Assert if scenario list, output file names, and data loader is None
    assert scenario_list, "No scenario list provided for visualization."
    assert output_file_names, "No output file names provided for visualization."
    assert data_loader, "Data loader is not provided."

    # region_list = data_loader.REGION_LIST
    # Set the time horizon
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )

    list_of_years = time_horizon.model_time_horizon
    # columns = list_of_years

    sns.set_style(sns_set_style)

    # Create subplots in grid of 4 rows and 2 columns
    fig, axs = plt.subplots(subplot_rows, subplot_columns, figsize=figsize)

    sns.set_theme(font_scale=font_scale)

    # Reshape axs to 1D for easy iteration
    axs = axs.ravel()

    lines = []

    # colors = ["red", "orange", "green", "blue"]
    color_palette = sns.color_palette(colourmap)
    # Iterate through each scenario list and iterate through each output file name to read the pickle file
    for scenario_idx, scenario in enumerate(scenario_list):
        for pf_idx, output_file_name in enumerate(output_file_names):
            with open(
                f"{path_to_data}/{output_file_name}_{scenario}_{variable_name}.pkl",
                "rb",
            ) as f:
                loaded_data = pickle.load(f)

                # Check if the data is a 3D array
                if len(loaded_data.shape) == 3:
                    # Sum up all the regions
                    loaded_data = np.sum(loaded_data, axis=0)
                elif len(loaded_data.shape) == 2:
                    loaded_data = loaded_data.T

                # Calculate the 5th and 95th percentile
                lower_percentile = np.percentile(
                    loaded_data, lower_percentile_value, axis=1
                )
                upper_percentile = np.percentile(
                    loaded_data, upper_percentile_value, axis=1
                )

                # Fill between the 5th and 95th percentile
                axs[scenario_idx].fill_between(
                    list_of_years,
                    lower_percentile,
                    upper_percentile,
                    alpha=alpha,
                    color=color_palette[pf_idx],
                )

                # Mean across ensemble members
                loaded_data = np.mean(loaded_data, axis=1)

                # Plot the data
                line = sns.lineplot(
                    x=list_of_years,
                    y=loaded_data,
                    ax=axs[scenario_idx],
                    color=color_palette[pf_idx],
                    label=output_file_name,
                )  #
                lines.append(line)
                # Set the title and axis labels
                axs[scenario_idx].title.set_fontsize(title_font_size)

                axs[scenario_idx].set_title(scenario)

                # axs[scenario_idx].set_xlabel('Year')
                axs[scenario_idx].set_ylabel("")
                # Styling each subplot
                axs[scenario_idx].spines["right"].set_visible(False)
                axs[scenario_idx].spines["top"].set_visible(False)
                axs[scenario_idx].xaxis.set_ticks_position("bottom")
                axs[scenario_idx].yaxis.set_ticks_position("left")
                axs[scenario_idx].set_ylim(yaxis_lower_limit, yaxis_upper_limit)

    # Remove the unused subplots
    for i in range(len(scenario_list), len(axs)):
        fig.delaxes(axs[i])

    # Add title to the figure
    fig.suptitle(main_title, fontsize=title_font_size * 2)
    # Adjust the layout and spacing
    fig.tight_layout()

    # Save the figure
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    plt.savefig(f"{path_to_output}/{variable_name}_subplots.png", dpi=300)

    return fig


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
