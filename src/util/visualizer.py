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
from src.util.regional_configuration import justice_region_aggregator
import pickle
from src.util.data_loader import DataLoader

from sklearn.preprocessing import MinMaxScaler

import json
import pycountry
import plotly.express as px
import plotly.graph_objects as go


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

    # TODO: Flip the axis here. Create a new argument for this
    # axes.invert_axis(data.columns[2])
    # axes.invert_axis(data.columns[3])

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
    show_title=True,
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
    visualization_start_year=2015,
    visualization_end_year=2300,
    saving=False,
    scenario_list=[],
):
    """
    @param figsize: Tuple, default=(15, 10)
    @param set_style: String, default="white"
    @param colourmap: String, default="bright"
    @param path_to_data: String, default="data/reevaluation"
    @param path_to_output: String, default="./data/plots"
    @param fontsize: Integer, default=15
    @param x_label: String, default="Years"
    @param y_label: String, default="Temperature Rise (°C)"
    @param variable_name: String, default="global_temperature"
    @param no_of_ensembles: Integer, default=1001
    @param input_data: List, default=[]
    @param output_titles: List, default=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"]
    @param main_title: String, default="Global Temperature Rise - "
    @param show_title: Boolean, default=True
    @param yaxis_lower_limit: Integer, default=0
    @param yaxis_upper_limit: Integer, default=10
    @param alpha: Float, default=0.1
    @param linewidth: Float, default=2.5
    @param lower_percentile: Integer, default=0
    @param upper_percentile: Integer, default=100
    @param start_year: Integer, default=2015
    @param end_year: Integer, default=2300
    @param data_timestep: Integer, default=5
    @param timestep: Integer, default=1
    @param visualization_start_year: Integer, default=2015
    @param visualization_end_year: Integer, default=2300
    @param saving: Boolean, default=False
    @param scenario_list: List, default=[], Accepted values: ['SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585']

    """

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

    data_scenario = np.zeros((len(scenario_list), len(list_of_years), no_of_ensembles))

    # Loop through the input data and plot the timeseries
    for plotting_idx, file in enumerate(input_data):
        # Load the scenario data from the pickle file
        with open(path_to_data + "/" + file, "rb") as f:
            scenario_data = pickle.load(f)

        # Loop through the scenarios
        for idx, scenarios in enumerate(scenario_list):

            data_scenario[idx, :, :] = scenario_data[scenarios][variable_name]

        # Select list of years for visualization
        list_of_years_sliced = list_of_years[
            time_horizon.year_to_timestep(
                visualization_start_year, timestep=timestep
            ) : (
                time_horizon.year_to_timestep(visualization_end_year, timestep=timestep)
                + timestep
            )
        ]

        median_data = np.zeros((len(scenario_list), len(list_of_years_sliced)))

        sns.set_style(set_style)
        fig, ax = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 0.5]}
        )

        # Set y-axis limits
        plt.ylim(yaxis_lower_limit, yaxis_upper_limit)

        label_list = []
        for idx, scenarios in enumerate(scenario_list):

            temp_df = pd.DataFrame(data_scenario[idx, :, :].T, columns=list_of_years)
            # Select temp_df for visualization
            temp_df = temp_df.loc[:, list_of_years_sliced]

            label = Scenario[scenarios].value[-1]
            color = color_palette[idx]

            # Calculate the percentiles
            p_l = np.percentile(temp_df, lower_percentile, axis=0)
            p_h = np.percentile(temp_df, upper_percentile, axis=0)

            # Calculate the median
            median_data[idx, :] = np.median(temp_df, axis=0)

            label = Scenario[scenarios].value[-1]  # ssp_rcp_string_list[idx]
            label_list.append(label)
            color = color_palette[idx]

            # Plot percentiles as bands
            ax[0].fill_between(list_of_years_sliced, p_l, p_h, color=color, alpha=alpha)

            # Select the last year of the data for the KDE plot
            temp_df_last_year = temp_df.iloc[:, -1]

            # Plot the boxplot on the second subplot
            sns.boxplot(
                x=idx,
                y=temp_df_last_year,
                color=color,
                ax=ax[1],
                width=0.5,
                showfliers=False,
            )
            # Plot the KDE plot
            # sns.kdeplot(
            #     y=temp_df_last_year, color=color, ax=ax[1], fill=True, alpha=alpha
            # )

        # Convert the mean_data to a dataframe
        median_data = pd.DataFrame(median_data, columns=list_of_years_sliced)

        for i in range(median_data.shape[0]):
            label = label_list[i]
            color = color_palette[i]
            sns.lineplot(
                data=median_data.iloc[i],
                color=color,
                alpha=alpha * 10,
                linewidth=linewidth,
                label=label,
                ax=ax[0],
            )
            ax[0].legend(loc="upper left", fontsize=fontsize)

        # Set the x-axis limit to end at the last year
        ax[0].set_xlim(visualization_start_year, visualization_end_year)

        # Set y-axis limits
        ax[0].set_ylim(yaxis_lower_limit, yaxis_upper_limit)

        # Adjust the space between subplots
        plt.subplots_adjust(wspace=0.1)

        # Remove top and right border for both plots
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)

        # Remove top and right ticks
        ax[0].xaxis.set_ticks_position("bottom")
        ax[0].yaxis.set_ticks_position("left")
        ax[1].xaxis.set_ticks_position("bottom")
        ax[1].yaxis.set_ticks_position("left")

        # Remove ticks for the second plot
        # ax[1].set_yticks([])

        # Set the labels
        ax[0].set_xlabel(x_label, fontsize=fontsize)
        ax[0].set_ylabel(y_label, fontsize=fontsize)
        ax[1].set_xlabel("Distribution", fontsize=fontsize)
        ax[1].set_ylabel(" ", fontsize=fontsize)

        # Set the font size of the tick labels
        ax[0].tick_params(axis="both", which="major", labelsize=fontsize)
        ax[1].tick_params(axis="both", which="major", labelsize=fontsize)

        if show_title:
            plt.suptitle(main_title, fontsize=fontsize)

        if saving:
            # Save the figure
            if not os.path.exists(path_to_output):
                os.makedirs(path_to_output)

            # Getting the input file's policy index number

            filename = file.split(".")[0]
            # Split filename based on the underscore and select the last element
            filename = filename.split("_")[-1]

            output_file_name = variable_name + "_" + filename

            plt.savefig(
                path_to_output
                + "/"
                + output_file_name
                + "_"
                + output_titles[plotting_idx],
                dpi=300,
                bbox_inches="tight",
            )

    return fig


def process_country_data_for_choropleth_plot(
    region_list=None,
    data=None,
    list_of_years=None,
    axis_to_average=2,
    year_to_visualize=2100,
    data_label="Emission Control Rate",
    ssp_scenario=0,
    region_to_country_mapping="data/input/rice50_regions_dict.json",
    scaler=False,
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
        ] = np.nan  # 0

        # Check for the CountryCode 'KSV' and set the 'CountryName' to 'Kosovo'
        data_scenario_year_by_country.loc[
            data_scenario_year_by_country["CountryCode"] == "KSV", "CountryName"
        ] = "Kosovo"

    return data_scenario_year_by_country


def process_2D_regional_data_for_choropleth_plot_v2_opt(  # OUTPUT dataset VALIDATED with the previous version
    region_list=None,
    data=None,
    list_of_years=None,
    year_to_visualize=2100,
    data_label="Emission Control Rate",
    region_to_country_mapping=None,
    # scaler=False,
    feature_scale=(0, 1),
    data_correction=True,
):
    # Assert if region list, data, and list of years are None
    assert region_list is not None, "Region list is not provided."
    assert data is not None, "Data is not provided."
    assert list_of_years is not None, "List of years is not provided."
    assert (
        region_to_country_mapping is not None
    ), "Region to country mapping is not provided."

    # Convert the region list, a numpy array, to a list
    regions_str = region_list.tolist()

    # Create DataFrame for the specific scenario
    data_scenario = pd.DataFrame(data, index=regions_str, columns=list_of_years)

    # Select the year to visualize
    data_scenario_year = data_scenario[year_to_visualize].reset_index()
    data_scenario_year.columns = ["Region", data_label]

    # Load region to country mapping from JSON file
    with open(region_to_country_mapping) as json_file:
        region_to_country = json.load(json_file)

    # Create a DataFrame from the mapping
    mapping_df = pd.DataFrame(
        list(region_to_country.items()), columns=["Region", "CountryCode"]
    )

    # Ensure 'Region' column exists in both DataFrames
    if "Region" not in data_scenario_year.columns or "Region" not in mapping_df.columns:
        raise KeyError("The 'Region' column is missing from one of the DataFrames.")

    # Merge the mapping DataFrame with the data
    data_scenario_year = pd.merge(
        mapping_df, data_scenario_year, on="Region", how="inner"
    )

    # Explode the CountryCode if it's a list
    data_scenario_year = data_scenario_year.explode("CountryCode")

    # Precompute country code to name mapping
    country_code_to_name = {
        country.alpha_3: country.name for country in pycountry.countries
    }

    # Map CountryCode to CountryName
    data_scenario_year["CountryName"] = data_scenario_year["CountryCode"].map(
        country_code_to_name
    )

    if data_correction:
        # Apply data corrections
        data_scenario_year.loc[
            data_scenario_year["CountryCode"] == "ATA", data_label
        ] = np.nan
        data_scenario_year.loc[
            data_scenario_year["CountryCode"] == "KSV", "CountryName"
        ] = "Kosovo"

    return data_scenario_year[["CountryCode", data_label, "CountryName"]]


def min_max_scaler(X, global_min, global_max):

    # Check if data is a numpy array
    if not isinstance(X, np.ndarray):
        print("Data is not a numpy array.")
        X = np.array(X)

    # print(np.nanmin(X), np.nanmax(X))

    X_std = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))

    X_scaled = X_std * (global_max - global_min) + global_min

    return X_scaled


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
    feature_scale=(0, 1),
    choropleth_data_length=3,
    data_normalization=True,
    show_colorbar=True,
    show_title=True,
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

    data_scenario_year_by_country_dict = {}

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
            # TODO: Make this an array
            data_scenario_year_by_country = process_country_data_for_choropleth_plot(
                region_list=region_list,
                data=data_scenario,
                list_of_years=list_of_years,
                year_to_visualize=year_to_visualize,
                data_label=data_label,
                ssp_scenario=idx,
                scaler=False,
            )

            data_scenario_year_by_country_dict[(plotting_idx, scenarios)] = (
                data_scenario_year_by_country
            )

    if data_normalization:
        # Initialize the scaler
        # scaler = MinMaxScaler(feature_scale)
        # Find the global minimum and maximum
        global_min = min(
            df[data_label].min() for df in data_scenario_year_by_country_dict.values()
        )

        global_max = max(
            df[data_label].max() for df in data_scenario_year_by_country_dict.values()
        )

        print("Global Min & Max", global_min, global_max)

        # Set the scaler's min and max
        # scaler.min_, scaler.scale_ = global_min, 1.0 / (global_max - global_min)

        # Loop over the keys in the dictionary
        for key in data_scenario_year_by_country_dict.keys():
            print(key)
            # Reshape the 'Emission' column to fit the scaler
            normalized_data = data_scenario_year_by_country_dict[key][
                data_label
            ].values.reshape(-1, 1)

            # Transform the 'Emission' column
            data_scenario_year_by_country_dict[key][data_label] = min_max_scaler(
                normalized_data, global_min, global_max
            )

            # print(data_scenario_year_by_country_dict[key][data_label])
            # data_scenario_year_by_country_dict[key][data_label] = scaler.transform(
            #     normalized_data
            # )

    # Loop through the input data and plot the choropleth
    for plotting_idx, file in enumerate(input_data):

        # Loop through all the scenarios and store the data in a 4D numpy array
        for idx, scenarios in enumerate(scenario_list):
            # TODO: Separate the the loops and carry normalization here
            choropleth_title = (
                title
                + str(year_to_visualize)
                + "-"
                + Scenario[scenarios].value[-1]  # Scenario.get_ssp_rcp_strings()[idx]
            )

            fig = px.choropleth(
                data_scenario_year_by_country_dict[(plotting_idx, scenarios)],
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

            if show_colorbar == False:
                fig.update_layout(coloraxis_showscale=False)

            if show_title:
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
            else:
                fig.update_layout(title_text="")

            # Policy index number
            filename = file.split(".")[0]
            # Split filename based on the underscore and select the last element
            filename = filename.split("_")[-1]

            output_file_name = (
                variable_name
                + "_"
                + filename
                + "_"
                + output_titles[plotting_idx]
                + "_"
                + Scenario[scenarios].value[-1]
            )
            if saving:
                # Save the plot as a png file
                # print("Saving plot for: ", scenarios, " - ", output_file_name)
                fig.write_image(path_to_output + "/" + output_file_name + ".png")

    # plotting_idx = 2
    return fig, data_scenario_year_by_country


def plot_choropleth_2D_data(
    variable_name="constrained_emission_control_rate",
    path_to_data="data/reevaluation/",
    path_to_output="./data/plots",
    year_to_visualize=2100,
    input_data=None,
    region_to_country_mapping="data/input/rice50_regions_dict.json",
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
    saving=False,
    scenario_list=[],
    data_normalization=True,
    show_colorbar=True,
    show_title=True,
):

    # Assert if input_data list and output_titles list is None
    assert input_data, "No input data provided for visualization."
    # assert output_titles, "No output titles provided for visualization."
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

    processed_data_dict = {}

    # Loop through the input data and plot the choropleth
    for plotting_idx, file in enumerate(input_data):
        # Load the scenario data from the pickle file
        data = np.load(path_to_data + file)

        processed_country_data = process_2D_regional_data_for_choropleth_plot_v2_opt(
            region_list=region_list,
            data=data,
            list_of_years=list_of_years,
            year_to_visualize=year_to_visualize,
            data_label=data_label,
            region_to_country_mapping=region_to_country_mapping,
        )

        processed_data_dict[(plotting_idx)] = processed_country_data

    if data_normalization:
        # Find the global minimum and maximum
        global_min = min(df[data_label].min() for df in processed_data_dict.values())

        global_max = max(df[data_label].max() for df in processed_data_dict.values())

        print("Global Min & Max", global_min, global_max)

        # Loop over the keys in the dictionary
        for idx in processed_data_dict.keys():
            print(idx)
            # Reshape the 'Emission' column to fit the scaler
            dataframe_to_normalize = processed_data_dict[(idx)]
            dataframe_to_normalize = dataframe_to_normalize[data_label].values.reshape(
                -1, 1
            )

            # Transform the 'Emission' column
            processed_data_dict[(idx)][data_label] = min_max_scaler(
                dataframe_to_normalize, global_min, global_max
            )

    # Loop through the input data and plot the choropleth
    for plotting_idx, file in enumerate(input_data):

        choropleth_title = " "  # output_titles[plotting_idx]

        fig = px.choropleth(
            processed_data_dict[(plotting_idx)],
            # processed_data_dict[plotting_idx],
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

        if show_colorbar == False:
            fig.update_layout(coloraxis_showscale=False)

        if show_title:
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
            )
        else:
            fig.update_layout(title_text="")

        # Policy index number
        filename = file.split(".")[0]
        # Split filename based on the underscore and select the last element
        filename = (
            filename.split("_")[0]
            + filename.split("_")[1]
            + "_"
            + filename.split("_")[-1]
        )

        output_file_name = filename
        print(output_file_name)
        if saving:
            # Save the plot as a png file
            fig.write_image(path_to_output + "/" + output_file_name + ".png")

    return fig, processed_data_dict


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

                # Set font size for tick labels
                axs[scenario_idx].tick_params(
                    axis="both", which="major", labelsize=title_font_size
                )
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


def plot_stacked_area_chart(
    variable_name=None,
    region_name_path="data/input/rice50_region_names.json",
    path_to_data="data/reevaluation",
    path_to_output="./data/plots",
    input_data=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"],
    output_titles=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"],
    scenario_list=[],
    title=None,
    title_x=0.5,
    xaxis_label=None,
    yaxis_label=None,
    legend_label=None,
    colour_palette=px.colors.qualitative.Light24,
    height=800,
    width=1200,
    visualization_start_year=2015,
    visualization_end_year=2300,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
    template="plotly_white",
    plot_title=None,
    groupnorm=None,
    region_aggegation=True,
    region_dict=None,
    saving=False,
    fontsize=15,
    yaxis_lower_limit=0,
    yaxis_upper_limit=25,
    show_legend=True,
    filetype=".pkl",
    regional_order=None,
):

    # Assert if input_data list, scenario_list and output_titles list is None
    assert input_data, "No input data provided for visualization."
    assert output_titles, "No output titles provided for visualization."
    assert scenario_list, "No scenario list provided for visualization."

    # Set the time horizon
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    list_of_years = time_horizon.model_time_horizon

    data_loader = DataLoader()

    region_list = data_loader.REGION_LIST

    if region_aggegation == True:
        assert region_dict, "Region dictionary is not provided."
        # region_list = list(region_dict.keys())
    else:
        with open(region_name_path, "r") as f:
            region_names = json.load(f)

        # Use region list to get the region names using the region names dictionary
        region_list = [region_names[region] for region in region_list]
        # Convert into a flat list
        region_list = [item for sublist in region_list for item in sublist]

    # print(region_list)

    # Load the data
    # Enumerate through the input data and load the data
    for idx, file in enumerate(input_data):
        for scenario in scenario_list:
            print("Loading data for: ", scenario, " - ", file)
            # Check if the file is pickle or numpy
            if ".npy" in filetype:
                data = np.load(
                    path_to_data
                    + "/"
                    + file
                    + "_"
                    + scenario
                    + "_"
                    + variable_name
                    + ".npy"
                )
            else:
                with open(
                    path_to_data
                    + "/"
                    + file
                    + "_"
                    + scenario
                    + "_"
                    + variable_name
                    + ".pkl",
                    "rb",
                ) as f:
                    data = pickle.load(f)

            # Check if region_aggegation is True
            if region_aggegation:
                # Aggregated Input Data
                region_list, data = justice_region_aggregator(
                    data_loader=data_loader, region_config=region_dict, data=data
                )

            # Check shape of data
            if len(data.shape) == 3:
                data = np.mean(data, axis=2)

            # Create a dataframe from the data
            data = pd.DataFrame(data, index=region_list, columns=list_of_years)

            # Create the slice according to visualization years
            data = data.loc[:, visualization_start_year:visualization_end_year]

            print("Region list: ", region_list)

            if regional_order is not None:
                region_list = regional_order

            # Create plotly figure
            fig = px.area(
                data.T,
                x=data.columns,
                y=data.index,
                title=plot_title,
                template=template,
                labels={"value": variable_name, "variable": "Region", "x": "Year"},
                height=height,
                width=width,
                color_discrete_sequence=colour_palette,
                groupnorm=groupnorm,
                category_orders={"variable": region_list},
            )
            if groupnorm is None:
                fig.update_layout(yaxis_range=[yaxis_lower_limit, yaxis_upper_limit])

            # Update layout
            fig.update_layout(
                legend_title_text=legend_label,
                # X-axis label
                xaxis_title=xaxis_label,
                # Y-axis label
                yaxis_title=yaxis_label,
                title_text=title,
                title_x=title_x,
                font=dict(size=fontsize),
                legend_traceorder="reversed",
                # legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )
            # Check if display legend is True
            if show_legend == False:
                fig.update_layout(showlegend=False)

            if saving:
                # Save the figure
                if not os.path.exists(path_to_output):
                    os.makedirs(path_to_output)

                output_file_name = (
                    variable_name + "_" + output_titles[idx] + "_" + scenario
                )
                print("Saving plot for: ", scenario, " - ", output_file_name)
                fig.write_image(path_to_output + "/" + output_file_name + ".png")

    return fig


def plot_hypervolume(
    path_to_data="data/convergence_metrics",
    path_to_output="./data/plots/convergence_plots",
    input_data={},  # Dictionary with input data types as keys and lists of seed files as values
    xaxis_title="Number of Function Evaluations",
    yaxis_title="Hypervolume",
    linewidth=3,
    colour_palette=px.colors.qualitative.Dark24,
    template="plotly_white",
    yaxis_upper_limit=0.7,
    title_x=0.5,
    width=1000,
    height=800,
    fontsize=15,
    saving=False,
):
    # Assert if input_data dictionary is empty
    assert input_data, "No input data provided for visualization."

    # Loop through the input data dictionary
    for idx, (data_type, seed_files) in enumerate(input_data.items()):
        fig = go.Figure()

        for seed_idx, file in enumerate(seed_files):
            data = pd.read_csv(path_to_data + "/" + file)
            # Keep only nfe and hypervolume columns
            data = data[["nfe", "hypervolume"]]
            data = data.sort_values(by="nfe")

            seed_number = file.split("_")[-2].split(".")[0]
            # Add the seed data to the plot
            fig.add_trace(
                go.Scatter(
                    x=data["nfe"],
                    y=data["hypervolume"],
                    fill="none",
                    mode="lines",
                    line=dict(
                        color=colour_palette[seed_idx % len(colour_palette)],
                        width=linewidth,
                    ),
                    name=f"Seed {seed_idx + 1} ({seed_number})",
                )
            )

        # Convert the data_type from all uppercase to title case
        titletext = data_type.title()

        # Set the chart title and axis labels
        fig.update_layout(
            title=dict(text=titletext),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            width=width,
            height=height,
            template=template,
            yaxis_range=[0, yaxis_upper_limit],
            title_x=title_x,
            font=dict(size=fontsize),
        )

        # Avoid zero tick in the y-axis - minor cosmetic change
        fig.update_yaxes(tickvals=(np.arange(0, yaxis_upper_limit, 0.1))[1:])

        # Save the figure
        if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)

        if saving:
            output_file_name = f"{titletext}_hypervolume_plot"
            fig.write_image(path_to_output + "/" + output_file_name + ".png")

    return fig


# def plot_hypervolume(
#     path_to_data="data/convergence_metrics",
#     path_to_output="./data/plots/convergence_plots",
#     input_data=[],  # Provide the list of input data files with extension
#     xaxis_title="Number of Function Evaluations",
#     yaxis_title="Hypervolume",
#     linewidth=3,
#     colour_palette=px.colors.qualitative.Dark24,
#     template="plotly_white",
#     yaxis_upper_limit=0.7,
#     title_x=0.5,
#     width=1000,
#     height=800,
#     fontsize=15,
#     saving=False,
# ):
#     # Assert if input_data list is empty
#     assert input_data, "No input data provided for visualization."

#     # Loop through the input data list and load the data
#     for idx, file in enumerate(input_data):
#         data = pd.read_csv(path_to_data + "/" + file)
#         # Keep only nfe and hypervolume columns
#         data = data[["nfe", "hypervolume"]]
#         data = data.sort_values(by="nfe")

#         # Find the max nfe value
#         nfe_max = data["nfe"].max()
#         # Get title text from filename
#         titletext = file.split("_")[0]
#         # Convert the titletext from all uppercase to title case
#         titletext = titletext.title()

#         fig = go.Figure(
#             data=[
#                 go.Scatter(
#                     x=data["nfe"],
#                     y=data["hypervolume"],
#                     fill="none",
#                     mode="lines",  #'none',
#                     line=dict(color=colour_palette[idx], width=linewidth),
#                     showlegend=False,
#                 )
#             ]
#         )

#         # Set the chart title and axis labels
#         fig.update_layout(
#             title=dict(text=titletext),
#             xaxis_title=xaxis_title,
#             yaxis_title=yaxis_title,
#             width=width,
#             height=height,
#             template=template,
#             yaxis_range=[0, yaxis_upper_limit],
#             title_x=title_x,
#             font=dict(size=fontsize),
#         )

#         # Avoid zero tick in the y-axis - minor cosmetic change
#         fig.update_yaxes(tickvals=(np.arange(0, yaxis_upper_limit, 0.1))[1:])

#         # Save the figure
#         if not os.path.exists(path_to_output):
#             os.makedirs(path_to_output)

#         if saving:
#             output_file_name = f"{titletext}_{nfe_max}_hypervolume_plot"
#             fig.write_image(path_to_output + "/" + output_file_name + ".png")

#     return fig


if __name__ == "__main__":

    fig, data = plot_choropleth(
        variable_name="constrained_emission_control_rate",
        path_to_data="data/reevaluation/only_welfare_temp/",  # "data/reevaluation/balanced/",  # "data/reevaluation",
        path_to_output="./data/plots/regional/only_welfare_temp",
        projection="natural earth",
        # scope='usa',
        year_to_visualize=2100,
        input_data=[
            "UTILITARIAN_reference_set_idx16.pkl",
            "PRIORITARIAN_reference_set_idx196.pkl",
            "SUFFICIENTARIAN_reference_set_idx57.pkl",
            "EGALITARIAN_reference_set_idx404.pkl",
        ],
        output_titles=[
            "Utilitarian",
            "Prioritarian",
            "Sufficientarian",
            "Egalitarian",
        ],
        title="Mitigation Burden Distribution in ",
        data_label="Emission Control Rate",
        colourmap="OrRd",
        legend_label="\n",
        # scenario_list= ['SSP245'],
        scenario_list=[
            "SSP119",
            "SSP126",
            "SSP245",
            "SSP370",
            "SSP434",
            "SSP460",
            "SSP534",
            "SSP585",
        ],  # ['SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585'],
        data_normalization=True,
        saving=True,
        show_colorbar=True,
        show_title=False,
    )

    ############################################################################################################
    # fig = plot_timeseries(
    #     path_to_data="data/reevaluation/only_welfare_temp/",  # "data/reevaluation",  # /balanced,  # "data/reevaluation",
    #     path_to_output="./data/plots/regional/only_welfare_temp",  # "./data/plots/regional",
    #     x_label="Years",
    #     y_label="Temperature Rise (°C)",
    #     variable_name="global_temperature",
    #     input_data=[
    #         "UTILITARIAN_reference_set_idx16.pkl",
    #         "PRIORITARIAN_reference_set_idx196.pkl",
    #         # "UTILITARIAN_reference_set_idx51.pkl",
    #         # "UTILITARIAN_reference_set_idx51_idx62.pkl",
    #         # "PRIORITARIAN_reference_set_idx817.pkl",
    #         # "PRIORITARIAN_reference_set_idx817_idx59.pkl",
    #         # "UTILITARIAN_reference_set_idx88.pkl",
    #         # "PRIORITARIAN_reference_set_idx748.pkl",
    #         # "SUFFICIENTARIAN_reference_set_idx99.pkl",
    #         # "EGALITARIAN_reference_set_idx147.pkl",
    #     ],
    #     output_titles=[
    #         "Utilitarian",
    #         "Prioritarian",
    #         # "Sufficientarian",
    #         # "Egalitarian",
    #     ],
    #     main_title="Global Temperature Rise - ",
    #     show_title=False,
    #     saving=True,
    #     yaxis_lower_limit=0,
    #     yaxis_upper_limit=6,
    #     alpha=0.1,
    #     linewidth=2.5,
    #     start_year=2015,
    #     end_year=2300,
    #     visualization_start_year=2025,
    #     visualization_end_year=2100,
    #     scenario_list=[
    #         "SSP119",
    #         "SSP245",
    #         "SSP370",
    #         "SSP434",
    #         "SSP585",
    #     ],  # ['SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585'], # #
    # )
