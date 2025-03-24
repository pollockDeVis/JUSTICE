"""
This module contains methods to visualize different kinds of output data from the JUSTICE model.
"""

from ema_workbench.analysis import plotting, Density, parcoords
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib.lines import Line2D
from src.util.model_time import TimeHorizon
from src.util.enumerations import *
from src.util.regional_configuration import (
    justice_region_aggregator,
    get_region_mapping,
)
import pickle
from src.util.data_loader import DataLoader
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

import json
import pycountry
import plotly.express as px
import plotly.graph_objects as go


def plot_emission_control_rate(
    data_files,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
    viz_end_year=2100,
    variable_name="constrained_emission_control",
    output_path=None,
    saving=False,
    show_max_line=False,
    show_top_10_regions=False,
    region_dict_path="data/input/9_regions.json",
    rice_50_region_dict_path="data/input/rice50_regions_dict.json",
    rice_50_names_path="data/input/rice50_region_names.json",
):
    """
    Plots the emission control rate for the given list of data files.

    Args:
        data_files (list): List of paths to the numpy files containing the data.
        start_year (int): The start year for the model time horizon.
        end_year (int): The end year for the model time horizon.
        data_timestep (int): The data time step, default is 5.
        timestep (int): The time step for year-to-timestep conversion, default is 1.
    """

    with open(region_dict_path, "r") as f:
        region_dict = json.load(f)

    with open(rice_50_region_dict_path, "r") as f:
        rice_50_region_dict = json.load(f)

    with open(rice_50_names_path, "r") as f:
        rice_50_names = json.load(f)

    for idx, data_file in enumerate(data_files):
        print("Loading data from: ", data_file)
        # Load the data
        data = np.load(data_file)

        # Check shape of data
        if len(data.shape) == 3:
            print("Average across the last dimension")
            data = np.mean(data, axis=2)

        # Initialize TimeHorizon and DataLoader
        time_horizon = TimeHorizon(
            start_year=start_year,
            end_year=end_year,
            data_timestep=data_timestep,
            timestep=timestep,
        )
        list_of_years = time_horizon.model_time_horizon

        data_loader = DataLoader()
        region_list = data_loader.REGION_LIST

        # Convert the data to a dataframe with list_of_years as columns and region_list as index
        data = pd.DataFrame(data, columns=list_of_years, index=region_list)

        data = process_data_for_stacked_area_plot(
            data,
            variable_name=variable_name,
            visualization_start_year=2015,
            visualization_end_year=2100,
            start_year=2015,
            end_year=2300,
            data_timestep=5,
            timestep=1,
            region_dict=region_dict,
            rice_50_names=rice_50_names,
            rice_50_region_dict=rice_50_region_dict,
        )

        fig = px.line(data, x="Year", y=variable_name, color="RICE50_Region_Names")

        # # Update layout for better visualization
        fig.update_layout(
            title=" ",
            xaxis_title="Year",
            yaxis_title="Emission Control Rate",
            template="plotly_white",
            height=600,
            width=1200,
        )

        # Hide the legend
        fig.update_layout(showlegend=False)

        if show_max_line:
            # Query the emission control
            query_string = f"Year == {viz_end_year}"
            sorted_data = data.query(query_string)
            # Sort regions based on the emission control rate in descending order
            sorted_data = sorted_data.sort_values(by=variable_name, ascending=False)

            # Get the maximum value of the emission control rate from sorted data
            max_value = sorted_data[variable_name].max()

        if show_top_10_regions:

            # extract the top 10 region from RICE50_Region_Names columns
            top_10_regions = sorted_data["RICE50_Region_Names"].head(10)

            # Convert the top 10 regions to a list
            top_10_regions = top_10_regions.tolist()

            # Add annotations for the top 10 regions
            for idx, region in enumerate(reversed(top_10_regions)):
                fig.add_annotation(
                    x=2110, y=(idx / 10 + 0.1), text=region, showarrow=False
                )

            # Show the max value as an annotation and draw a horizontal line
            fig.add_shape(
                type="line",
                x0=2015,
                y0=max_value,
                x1=2100,
                y1=max_value,
                line=dict(color="red", width=1, dash="dash"),
            )

            # Add annotation for the max value
            fig.add_annotation(
                x=2110,
                y=1.1,
                text="Top 10 Regions",
                # Update the font color
                font=dict(color="red"),
                showarrow=False,
            )

            fig.add_annotation(
                x=2020, y=1.03, text=f"Max EC: {max_value:.2f}", showarrow=False
            )

        if saving:
            # Save the plot as a png file
            fig.write_image(
                output_path + f"/{data_file.split('/')[-1].split('.')[0]}.svg"
            )
            # Save it as html file
            # fig.write_html(
            #     output_path + f"/{data_file.split('/')[-1].split('.')[0]}.html"
            # )

        fig.show()
    return fig, data


def plot_emissions_comparison_with_boxplots(
    data_paths,  # List of paths for the data
    start_year,
    end_year,
    data_timestep,
    timestep,
    visualization_start_year,
    visualization_end_year,
    yaxis_range,
    opacity,
    plot_title,
    xaxis_title,
    yaxis_title,
    template,
    width,
    height,
    baseline_path=None,
    colors=["coral", "lightgreen"],
    median_colors=["red", "green"],
    baseline_color="gray",
    fontsize=18,
    column_widths=[0.8, 0.2],
    output_path=None,
    saving=False,
):

    # Set the time horizon
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    list_of_years = time_horizon.model_time_horizon

    # Load baseline data if provided
    if baseline_path:
        baseline = np.load(baseline_path)
        if len(baseline.shape) == 3:
            baseline = np.sum(baseline, axis=0)
        baseline = pd.DataFrame(baseline.T, columns=list_of_years)
        baseline = baseline.loc[:, visualization_start_year:visualization_end_year]
        baseline = baseline.T
        baseline = baseline.mean(axis=1)

    # Load the data and create dataframes
    data_frames = []
    for path in data_paths:
        filetype = os.path.splitext(path)[1]
        if filetype == ".npy":
            data = np.load(path)
        elif filetype == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
        elif filetype == ".csv":
            data = pd.read_csv(path)

        # Only sum over regions if the data is 3D
        if len(data.shape) == 3:
            data = np.sum(data, axis=0)

        data = data.T
        df = pd.DataFrame(data, columns=list_of_years).loc[
            :, visualization_start_year:visualization_end_year
        ]
        data_frames.append(df.T)

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2, column_widths=column_widths, subplot_titles=[plot_title, " "]
    )

    # Loop through data frames and add subplots for each
    for idx, emissions in enumerate(data_frames):
        color = colors[idx]
        median_color = median_colors[idx]

        # Calculate the percentiles
        max_percentile = np.percentile(emissions, 100, axis=1)
        min_percentile = np.percentile(emissions, 0, axis=1)
        p75 = np.percentile(emissions, 75, axis=1)
        p25 = np.percentile(emissions, 25, axis=1)

        # Add traces for the envelopes
        fig.add_trace(
            go.Scatter(
                x=emissions.index,
                y=max_percentile,
                mode="lines",
                line=dict(color=color, width=0.5),
                fill=None,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=emissions.index,
                y=min_percentile,
                mode="lines",
                line=dict(color=color, width=0.5),
                fill="tonexty",
                opacity=opacity * 0.01,  # make it more transparent
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Add traces for the 25th to 75th percentile envelope
        fig.add_trace(
            go.Scatter(
                x=emissions.index,
                y=p75,
                mode="lines",
                line=dict(color=color, width=0.5),
                fill=None,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=emissions.index,
                y=p25,
                mode="lines",
                line=dict(color=color, width=0.5),
                fill="tonexty",
                opacity=opacity,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Add trace for the median emissions
        fig.add_trace(
            go.Scatter(
                x=emissions.index,
                y=emissions.median(axis=1),
                mode="lines",
                line=dict(color=median_color, width=2),
                name=f"Median {idx+1}",
            ),
            row=1,
            col=1,
        )

        # Add box plot for the last year's data
        last_year_data = emissions.iloc[-1]
        filename = data_paths[idx].split("/")[-1].split(".")[0]
        # Now split the "_" and take the first part
        filename = filename.split("_")[0]
        fig.add_trace(
            go.Box(
                y=last_year_data,
                name=filename,
                marker=dict(color=median_color),
                width=0.1,
            ),
            row=1,
            col=2,
        )

    # Add baseline if provided
    if baseline_path:
        fig.add_trace(
            go.Scatter(
                x=emissions.index,
                y=baseline,
                mode="lines",
                line=dict(color=baseline_color, width=2, dash="dash"),
                name="Baseline",
            ),
            row=1,
            col=1,
        )

    # Styling the box plots
    fig.update_traces(
        marker=dict(line=dict(width=0.3, color=baseline_color)), row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=plot_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template=template,
        height=height,
        width=width,
    )

    # Align the y-axes for both subplots
    fig.update_yaxes(
        title_text=yaxis_title, range=yaxis_range, showgrid=False, row=1, col=1
    )
    fig.update_yaxes(
        range=yaxis_range, showticklabels=False, showgrid=False, row=1, col=2
    )

    # Update x-axis for the line plot
    fig.update_xaxes(title_text=xaxis_title, showgrid=False, row=1, col=1)

    # Update font size
    fig.update_layout(font=dict(size=fontsize))

    # Adjust the width of the first subplot (column=1) to be more than the second subplot (column=2)
    fig.update_layout(
        xaxis=dict(domain=[0, 0.8]),  # First subplot takes 80% of the width
        xaxis2=dict(
            domain=[0.95, 1]
        ),  # Second subplot takes the remaining 10% of the width
    )

    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black", ticks="outside", row=1, col=1
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black", ticks="outside", row=1, col=1
    )

    # Show the figure
    fig.show()

    if saving:
        filename = "_".join(
            [os.path.splitext(os.path.basename(path))[0] for path in data_paths]
        )
        # Save the plot
        fig.write_image(f"{output_path}/{filename}.svg")


def plot_median_emission_comparison_with_baseline(
    data_paths=[],
    labels=[],
    fill=["none", "none", "tozeroy"],
    path_to_output="./data/plots",
    xaxis_title=None,
    yaxis_title=None,
    linewidth=3,
    colour_palette=px.colors.qualitative.Dark2,
    template="plotly_white",
    yaxis_upper_limit=0.7,
    visualization_start_year=2025,
    visualization_end_year=2100,
    title_x=0.5,
    width=1000,
    height=800,
    fontsize=15,
    saving=False,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
):

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    list_of_years = time_horizon.model_time_horizon

    data_frames = []
    for idx, path in enumerate(data_paths):
        filetype = os.path.splitext(path)[1]
        if filetype == ".npy":
            data = np.load(path)
        elif filetype == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
        elif filetype == ".csv":
            data = pd.read_csv(path)

        # Check if data is 3D. Then take the mean across the first dimension
        if len(data.shape) == 3:
            data = np.sum(data, axis=0)

        data = data.T
        df = pd.DataFrame(data, columns=list_of_years).loc[
            :, visualization_start_year:visualization_end_year
        ]

        # Calculate the median across the rows
        median = df.median()

        data_frames.append(median)

    fig = go.Figure()
    # Enumerate through the data_frames and plot the line plots in the same figure
    for idx, median in enumerate(data_frames):
        line_dash = "dash" if idx == 0 else "solid"
        fig.add_trace(
            go.Scatter(
                x=median.index,
                y=median.values,
                fill=fill[idx],
                mode="lines",
                line=dict(
                    color=colour_palette[idx % len(colour_palette)],
                    width=linewidth,
                    dash=line_dash,
                ),
                name=labels[idx],
            )
        )

    # Set the chart title and axis labels
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        template=template,
        yaxis_range=[
            0.01,
            yaxis_upper_limit,
        ],  # 0.01 is purely cosmetic to avoid the y-axis starting at 0
        title_x=title_x,
        font=dict(size=fontsize),
    )

    # get rid of gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Show x and y axis line with ticks outside
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", ticks="outside")

    fig.show()

    if saving:
        if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)

        # Loop through labels and append them to the output file name
        output_file_name = "emission_comparison_" + "_".join(labels)
        fig.write_image(path_to_output + "/" + output_file_name + "colorblind" + ".svg")

    return fig, data_frames


def plot_comparison_with_boxplots(
    data_paths,
    labels,
    start_year,
    end_year,
    data_timestep,
    timestep,
    visualization_start_year,
    visualization_end_year,
    yaxis_range,
    plot_title,
    output_path,
    xaxis_title,
    yaxis_title,
    template,
    width,
    height,
    linecolors=["red", "green", "blue", "orange"],
    colors=[
        "rgba(255, 0, 0, 0.2)",
        "rgba(0, 128, 0, 0.2)",
        "rgba(0, 0, 255, 0.2)",
        "rgba(255, 165, 0, 0.2)",
    ],
    show_red_dashed_line=False,
    saving=False,
    fontsize=18,
    show_interquartile_range=True,
    first_plot_proportion=[0, 0.8],
    second_plot_proportion=[0.95, 1],
    transpose_data=True,
):
    # Set the time horizon
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    list_of_years = time_horizon.model_time_horizon

    # Load the data and create dataframes
    data_frames = []
    for path in data_paths:
        filetype = os.path.splitext(path)[1]
        if filetype == ".npy":
            data = np.load(path)
        elif filetype == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
        elif filetype == ".csv":
            data = pd.read_csv(path)
        # Check if data is 3D. Then take the mean across the first dimension
        if len(data.shape) == 3:
            data = np.sum(data, axis=0)

        if transpose_data:
            data = data.T
        df = pd.DataFrame(data, columns=list_of_years).loc[
            :, visualization_start_year:visualization_end_year
        ]
        data_frames.append(df)

    # Function to calculate statistics
    def calc_stats(df):
        median = df.median(axis=0)
        min_vals = df.min(axis=0)
        max_vals = df.max(axis=0)
        percentile_25 = df.quantile(0.25, axis=0)
        percentile_75 = df.quantile(0.75, axis=0)
        return median, min_vals, max_vals, percentile_25, percentile_75

    # Calculate statistics for each dataframe
    stats = [calc_stats(df) for df in data_frames]

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.7, 0.3], subplot_titles=[plot_title, " "]
    )

    # Function to create traces for the line plot, including 25-75 envelope
    def add_traces(
        fig, median, min_vals, max_vals, p25, p75, name, color, linecolor, col
    ):
        fig.add_trace(
            go.Scatter(
                x=median.index,
                y=median.values,
                mode="lines",
                name=name,
                line=dict(color=linecolor),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=median.index,
                y=max_vals.values,
                mode="lines",
                line=dict(width=0),
                fillcolor=color,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=median.index,
                y=min_vals.values,
                mode="lines",
                fill="tonexty",
                fillcolor=color,
                line=dict(width=0),
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        if show_interquartile_range:
            # Add traces for the 25-75 percentile envelope
            fig.add_trace(
                go.Scatter(
                    x=median.index,
                    y=p75.values,
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=color.replace("0.2", "0.5"),  # Make 25-75 more visible
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=median.index,
                    y=p25.values,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=color.replace("0.2", "0.5"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

    for i, (median, min_vals, max_vals, p25, p75) in enumerate(stats):
        add_traces(
            fig,
            median,
            min_vals,
            max_vals,
            p25,
            p75,
            labels[i],
            colors[i % len(colors)],
            linecolors[i % len(linecolors)],
            col=1,
        )

    # Extract the last column data for each dataframe
    last_year_data = [df.iloc[:, -1] for df in data_frames]

    # Add box plots as a second subplot
    for i, data in enumerate(last_year_data):
        fig.add_trace(
            go.Box(
                y=data,
                name=labels[i],
                marker=dict(
                    color=linecolors[i % len(linecolors)]
                ),  # Use corresponding color from linecolors
                width=0.2,  # Adjust width
            ),
            row=1,
            col=2,
        )

    # Add marker borders
    fig.update_traces(marker=dict(line=dict(width=0.3, color="gray")), row=1, col=2)

    # Add styling and labels for the box plot
    fig.update_yaxes(title_text=" ", range=yaxis_range, row=1, col=2)

    fig.update_yaxes(title_text=yaxis_title, range=yaxis_range, row=1, col=1)

    # Remove y axis labels for the box plot
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # Remove vertical gridlines for first subplot
    fig.update_xaxes(showgrid=False, title_text=xaxis_title, row=1, col=1)

    # Make Y axis label for every 1 unit
    fig.update_yaxes(tick0=0, dtick=1, row=1, col=1)

    if show_red_dashed_line:
        # Add a red dashed line at 2°C for reference
        fig.add_shape(
            dict(
                type="line",
                x0=-1,
                y0=2,
                x1=3.1,
                y1=2,
                line=dict(color="red", width=1, dash="dash"),
            ),
            row=1,
            col=2,
        )

        # Add red dashed line annotation to the first subplot
        fig.add_shape(
            dict(
                type="line",
                x0=2025,
                y0=2,
                x1=2100,
                y1=2,
                line=dict(color="red", width=1, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Update layout
    fig.update_layout(template=template, width=width, height=height)

    # Adjust the width of the first subplot (column=1) to be more than the second subplot (column=2)
    fig.update_layout(
        xaxis=dict(
            domain=first_plot_proportion
        ),  # First subplot takes 70% of the width first_plot_proportion=[0, 0.8]second_plot_proportion=[0.95, 1]
        xaxis2=dict(
            domain=second_plot_proportion  #
        ),  # Second subplot takes the remaining 25% of the width
    )

    # Update font size
    fig.update_layout(font=dict(size=fontsize))

    # Remove gridlines
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=2)

    # Ticks
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black", ticks="outside", row=1, col=1
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black", ticks="outside", row=1, col=1
    )

    # Show the plot
    fig.show()

    if saving:
        filename = data_paths[0].split("/")[-1].split(".")[0]
        # Save the plot
        fig.write_image(f"{output_path}/{filename}.svg")


def plot_reevaluated_welfare(
    prioritarian_data_path,
    utilitarian_data_path,
    output_path,
    reevaluation="UTIL_PRIOR",
    scaling=False,
    saving=False,
):
    # Load the data
    prioritarian_data = pd.read_csv(prioritarian_data_path)
    utilitarian_data = pd.read_csv(utilitarian_data_path)

    prioritarian_pareto_set = prioritarian_data[
        ["welfare_utilitarian", "welfare_prioritarian"]
    ]
    utilitarian_pareto_set = utilitarian_data[
        ["welfare_utilitarian", "welfare_prioritarian"]
    ]

    # Combine the two dataframes for scaling. Essential to get the global min and max values
    combined_data = pd.concat([prioritarian_pareto_set, utilitarian_pareto_set])

    if scaling:
        # Normalize the columns based on the min and max values
        scaler = MinMaxScaler()
        # Fit the scaler on the data
        scaler.fit(combined_data)
        prioritarian_pareto_set = scaler.transform(prioritarian_pareto_set)
        utilitarian_pareto_set = scaler.transform(utilitarian_pareto_set)

        # Adjust the direction of the welfare columns
        prioritarian_pareto_set[:, 0] = 1 - prioritarian_pareto_set[:, 0]
        utilitarian_pareto_set[:, 0] = 1 - utilitarian_pareto_set[:, 0]

        prioritarian_pareto_set[:, 1] = 1 - prioritarian_pareto_set[:, 1]
        utilitarian_pareto_set[:, 1] = 1 - utilitarian_pareto_set[:, 1]

    else:
        prioritarian_pareto_set = prioritarian_pareto_set.values
        utilitarian_pareto_set = utilitarian_pareto_set.values

    fig = go.Figure()

    # Add traces for Prioritarian
    fig.add_trace(
        go.Scatter(
            x=prioritarian_pareto_set[:, 0],
            y=prioritarian_pareto_set[:, 1],
            mode="markers",
            name="Prioritarian",
            marker=dict(color="#8da0cb"),
        )
    )

    # Add traces for Utilitarian
    fig.add_trace(
        go.Scatter(
            x=utilitarian_pareto_set[:, 0],
            y=utilitarian_pareto_set[:, 1],
            mode="markers",
            name="Utilitarian",
            marker=dict(color="#fc8d62"),
        )
    )

    # Update layout
    fig.update_layout(
        title=" ",
        xaxis_title="Utilitarian Welfare",
        yaxis_title="Prioritarian Welfare",
        template="plotly_white",
        width=700,
        height=700,
    )
    # Remove the gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Add axis line with ticks
    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linewidth=1,
        linecolor="black",
        ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        showticklabels=True,
        linewidth=1,
        linecolor="black",
        ticks="outside",
    )

    if scaling:
        # Make x and y axis start from 0
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])

    if saving:
        filename = (
            "REEVAL_"
            + reevaluation
            + "_"
            + utilitarian_data_path.split("/")[-1].split(".")[0]
            + "_vs_"
            + prioritarian_data_path.split("/")[-1].split(".")[0]
        )
        # Save the plot
        fig.write_image(f"{output_path}/{filename}.svg")

    # Show the figure
    fig.show()

    return utilitarian_data, prioritarian_data


def plot_sunburst(
    variable_name=None,
    path_to_data="data/reevaluation",
    path_to_output="./data/plots",
    input_data=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"],
    output_titles=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"],
    scenario_list=[],
    path=["Selected Year", "Region", "RICE50_Region_Names"],
    year_to_visualize=2100,
    height=800,
    width=800,
    visualization_start_year=2015,
    visualization_end_year=2300,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
    region_dict_filepath="data/input/9_regions.json",
    saving=False,
    filetype=".pkl",
    color_discrete_map={
        "(?)": "rgb(255,255,255)",
        "Rest of the World": "rgb(153,153,153)",
        "Europe": "rgb(247,129,191)",
        "Gulf Countries": "rgb(166,86,40)",
        "South and Southeast Asia": "rgb(255,255,51)",
        "Other High Income": "rgb(255,127,0)",
        "Sub-Saharan Africa": "rgb(152,78,163)",
        "China": "rgb(77,175,74)",
        "India": "rgb(55,126,184)",
        "United States": "rgb(228,26,28)",
    },
):

    # Assert if input_data list, scenario_list and output_titles list is None
    assert input_data, "No input data provided for visualization."
    assert output_titles, "No output titles provided for visualization."
    assert scenario_list, "No scenario list provided for visualization."
    assert region_dict_filepath, "No region dictionary provided for visualization."

    pd.options.mode.copy_on_write = True

    # Load the dictionary from the json file
    with open(region_dict_filepath, "r") as f:
        region_dict = json.load(f)

    with open("data/input/rice50_regions_dict.json", "r") as f:
        rice_50_region_dict = json.load(f)

    with open("data/input/rice50_region_names.json", "r") as f:
        rice_50_names = json.load(f)

    # Enumerate through the input data and load the data
    for idx, file in enumerate(input_data):
        for scenario in scenario_list:
            print("Loading data for: ", scenario, " - ", file)

            # Search for file in the path
            filename = path_to_data + "/" + file + "_" + scenario + "_" + variable_name

            # Check if the file is pickle or numpy
            if filetype == ".npy":
                data = np.load(filename + ".npy")
            else:
                with open(
                    filename + ".pkl",
                    "rb",
                ) as f:
                    data = pickle.load(f)

            # Check shape of data
            if len(data.shape) == 3:
                data = np.mean(data, axis=2)

            data = process_data_for_stacked_area_plot(
                data,
                variable_name,
                visualization_start_year,
                visualization_end_year,
                start_year,
                end_year,
                data_timestep,
                timestep,
                region_dict,
                rice_50_names,
                rice_50_region_dict,
            )

            # Get all the keys of region_dict
            aggregated_region_dict_keys = list(region_dict.keys())

            # data query for the selected year
            query_string = "Year == " + str(year_to_visualize)
            data = data.query(query_string)

            # Add a column to the dataframe with the selected year
            data.loc[:, "Selected Year"] = str(year_to_visualize)

            fig = px.sunburst(
                data,
                path=path,
                values=variable_name,  # 'abatement_cost'
                color="Region",
                color_discrete_map=color_discrete_map,
            )

            # Update the size of the figure
            fig.update_layout(width=width, height=height)

            if saving:
                # Save the figure
                if not os.path.exists(path_to_output):
                    os.makedirs(path_to_output)

                output_file_name = (
                    variable_name
                    + "_"
                    + output_titles[idx]
                    + "_"
                    + scenario
                    + "_"
                    + str(year_to_visualize)
                )
                print("Saving plot for: ", scenario, " - ", output_file_name)
                fig.write_image(path_to_output + "/" + output_file_name + ".png")

    return fig, data


def process_data_for_stacked_area_plot(
    data,
    variable_name,
    visualization_start_year,
    visualization_end_year,
    start_year,
    end_year,
    data_timestep,
    timestep,
    region_dict,
    rice_50_names,
    rice_50_region_dict,
):
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

    # Create a DataFrame from the data
    data = pd.DataFrame(data, index=region_list, columns=list_of_years)

    # Slice the data according to visualization years
    data = data.loc[:, visualization_start_year:visualization_end_year]

    # Pivot the data so that all years are in one column
    data = data.reset_index().melt(
        id_vars="index", var_name="Year", value_name=variable_name
    )

    # Sort the data by regions and years
    data.sort_values(by=["index", "Year"], inplace=True)

    # Rename index to RICE50_Region
    data.rename(columns={"index": "RICE50_Region"}, inplace=True)

    # Create a new column with RICE50_Region names using vectorized operations
    data["RICE50_Region_Names"] = data["RICE50_Region"].map(
        lambda x: rice_50_names.get(x, [x])[0]
    )

    # Get the mapping dictionary
    mapping_dictionary = get_region_mapping(
        aggregated_region_dict=region_dict,
        disaggregated_region_dict=rice_50_region_dict,
        similarity_threshold=0.01,
    )

    # Create a reverse mapping for efficient lookup
    reverse_mapping = {v: k for k, values in mapping_dictionary.items() for v in values}

    # Map the RICE50_Region to Region using the reverse mapping
    data["Region"] = data["RICE50_Region"].map(reverse_mapping)

    return data


def plot_stacked_area_chart_v2(
    variable_name=None,
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
    colour_palette=px.colors.qualitative.Set1_r,
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
    region_dict_filepath="data/input/9_regions.json",
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
    assert region_dict_filepath, "No region dictionary provided for visualization."

    # Load the dictionary from the json file
    with open(region_dict_filepath, "r") as f:
        region_dict = json.load(f)

    with open("data/input/rice50_regions_dict.json", "r") as f:
        rice_50_region_dict = json.load(f)

    with open("data/input/rice50_region_names.json", "r") as f:
        rice_50_names = json.load(f)

    # Enumerate through the input data and load the data
    for idx, file in enumerate(input_data):
        for scenario in scenario_list:
            print("Loading data for: ", scenario, " - ", file)

            # Search for file in the path
            filename = path_to_data + "/" + file + "_" + scenario + "_" + variable_name

            # Check if the file is pickle or numpy
            if filetype == ".npy":
                data = np.load(filename + ".npy")
            else:
                with open(
                    filename + ".pkl",
                    "rb",
                ) as f:
                    data = pickle.load(f)

            # Check shape of data
            if len(data.shape) == 3:
                data = np.mean(data, axis=2)

            data = process_data_for_stacked_area_plot(
                data,
                variable_name,
                visualization_start_year,
                visualization_end_year,
                start_year,
                end_year,
                data_timestep,
                timestep,
                region_dict,
                rice_50_names,
                rice_50_region_dict,
            )

            # Get all the keys of region_dict
            aggregated_region_dict_keys = list(region_dict.keys())

            # Compare the regional_order with the aggregated_region_dict_keys
            if regional_order:
                assert set(regional_order) == set(
                    aggregated_region_dict_keys
                ), "Regional order does not match the aggregated region dictionary keys."

            # Create plotly figure
            fig = px.area(
                data,
                x="Year",
                y=variable_name,
                color="Region",
                line_group="RICE50_Region_Names",
                title=plot_title,
                template=template,
                height=height,
                width=width,
                color_discrete_sequence=colour_palette,
                groupnorm=groupnorm,
                category_orders={"Region": regional_order},
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
            )

            # Check if display legend is True
            if show_legend == False:
                fig.update_layout(showlegend=False)

            # Remove gridlines
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            if saving:
                # Save the figure
                if not os.path.exists(path_to_output):
                    os.makedirs(path_to_output)

                output_file_name = (
                    variable_name + "_" + output_titles[idx] + "_" + scenario
                )
                print("Saving plot for: ", scenario, " - ", output_file_name)
                fig.write_image(path_to_output + "/" + output_file_name + ".svg")

    return fig, data


def visualize_tradeoffs(
    input_data=[],
    figsize=(15, 10),
    set_style="whitegrid",
    font_scale=1.8,
    colourmap="bright",
    linewidth=0.4,
    alpha=0.1,
    path_to_data="data/reevaluation/",
    path_to_output="./data/plots/only_welfare_temp",
    scaling=True,
    feature_range=(0, 1),
    column_labels=None,
    legend_labels=None,
    show_legend=True,
    axis_rotation=30,
    fontsize=12,
    list_of_objectives=[
        "welfare_utilitarian",
        "years_above_temperature_threshold",
        "damage_cost_per_capita_utilitarian",
        "abatement_cost_per_capita_utilitarian",
    ],
    direction_of_optimization=[
        "min",
        "min",
        "max",
        "max",
    ],
    pretty_labels=[
        "Welfare",
        "Years Above Temp Threshold",
        "Welfare Loss Damage",
        "Welfare Loss Abatement",
    ],
    default_colors=["red", "blue"],
    top_percentage=0.1,
    objective_of_interest="welfare_utilitarian",
    show_best_solutions=False,
    temperature_filter=False,
    saving=False,
):

    sns.set_theme(font_scale=font_scale)
    sns.set_style(set_style)
    sns.set_theme(rc={"figure.figsize": figsize})

    # Assertions
    assert input_data, "Input data not provided"
    assert path_to_data, "Path to reference set is not provided"
    assert len(list_of_objectives) == len(
        direction_of_optimization
    ), "Length of objectives and direction of optimization not equal"

    color_mapping = {
        file: default_colors[i % len(default_colors)]
        for i, file in enumerate(input_data)
    }

    if column_labels:
        assert len(column_labels) == len(
            list_of_objectives
        ), "Length of column labels and objectives not equal"

    concatenated_df = pd.DataFrame()

    for file in input_data:
        data = pd.read_csv(path_to_data + "/" + file)
        data = data[list_of_objectives]
        data = np.abs(data)

        # Add a column to track the data type
        data["type"] = file

        concatenated_df = pd.concat([concatenated_df, data], axis=0)

    # Reset index for concatenated_df
    concatenated_df.reset_index(drop=True, inplace=True)

    # Determine top 10% indices for the objective of interest
    top_indices = {}
    if show_best_solutions:
        for file in input_data:
            df_type = concatenated_df[concatenated_df["type"] == file]

            top_indices[file] = (
                df_type[objective_of_interest]
                .nsmallest(int(df_type.shape[0] * top_percentage))
                .index
            )
            print(file, len(top_indices[file]))

            # Now within the top_indices, find the index with lowest years_above_temperature_threshold
            if temperature_filter:
                index = df_type.loc[top_indices[file]][
                    "years_above_temperature_threshold"
                ].idxmin()
                print(index)
                top_indices[file] = [index]

    if scaling:
        # Printing min max values of the objectives
        print("Min and Max values of the objectives", list_of_objectives)
        print(concatenated_df[list_of_objectives].min())
        print(concatenated_df[list_of_objectives].max())

        # Scale the data
        scaler = MinMaxScaler(feature_range=feature_range)
        concatenated_df[list_of_objectives] = scaler.fit_transform(
            concatenated_df[list_of_objectives]
        )

        for i, direction in enumerate(direction_of_optimization):
            if direction == "min":
                concatenated_df[list_of_objectives[i]] = (
                    1 - concatenated_df[list_of_objectives[i]]
                )

    limits = parcoords.get_limits(concatenated_df[list_of_objectives])
    limits.columns = pretty_labels
    axes = parcoords.ParallelAxes(limits, rot=axis_rotation, fontsize=fontsize)

    adjusted_linewidth = linewidth
    # Plot each row with its corresponding color
    for idx, row in concatenated_df.iterrows():
        if show_best_solutions:
            # Default to gray for all lines, except top indices get color
            file_color = "gray"
            adjusted_linewidth = linewidth
            for _type, indices in top_indices.items():
                if idx in indices:
                    file_color = color_mapping[_type]
                    if temperature_filter:
                        adjusted_linewidth = linewidth * 5
                    break
        else:
            # Color differentiation based on file type
            file_color = color_mapping.get(
                row["type"], "green"
            )  # Default to 'green' if no specific color is set

        _sliced_data = pd.DataFrame(row[list_of_objectives].values).T
        _sliced_data.columns = pretty_labels
        if show_best_solutions or temperature_filter:
            axes.plot(
                _sliced_data,
                color=file_color,
                linewidth=adjusted_linewidth,
                alpha=alpha if (show_best_solutions and file_color == "gray") else 1.0,
            )
        else:
            axes.plot(
                _sliced_data,
                color=file_color,
                linewidth=adjusted_linewidth,
                alpha=alpha,
            )

    # Creating a legend
    if show_legend:  # Only show legend when not highlighting best solutions
        if legend_labels is None:
            legend_labels = list(color_mapping.keys())
            # Split the legend labels by '_' and keep the first part
            legend_labels = [label.split("_")[0] for label in legend_labels]

            unique_colors = list(color_mapping.values())
            legend_elements = [
                Line2D([0], [0], color=unique_colors[i], lw=2, label=legend_labels[i])
                for i in range(len(unique_colors))
            ]

            plt.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize="small",
                bbox_to_anchor=(1.1, 1.1),
            )

    if saving:
        output_file_name = (
            "tradeoffs_"
            + "_".join([file.split("_")[0] for file in input_data])
            + "_"
            + ".svg"
        )
        # Save the figure
        if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)
        # Save the plot as svg
        plt.savefig(path_to_output + "/" + output_file_name, dpi=300)
        # plt.savefig(path_to_output + "/" + output_file_name, dpi=300)

    # Show the plot
    plt.show()
    return concatenated_df


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
        # Find the global minimum and maximum
        global_min = min(
            df[data_label].min() for df in data_scenario_year_by_country_dict.values()
        )

        global_max = max(
            df[data_label].max() for df in data_scenario_year_by_country_dict.values()
        )

        print("Global Min & Max", global_min, global_max)

        # Loop over the keys in the dictionary
        for key in data_scenario_year_by_country_dict.keys():
            print(key)
            # Reshape the 'data' column to fit the scaler
            normalized_data = data_scenario_year_by_country_dict[key][
                data_label
            ].values.reshape(-1, 1)

            # Transform the 'data' column
            data_scenario_year_by_country_dict[key][data_label] = min_max_scaler(
                normalized_data, global_min, global_max
            )

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
                fig.write_image(path_to_output + "/" + output_file_name + ".svg")

    return fig, data_scenario_year_by_country


def plot_choropleth_2D_data(
    path_to_data="data/reevaluation/",
    path_to_output="./data/plots",
    year_to_visualize=2050,
    input_data_path_list=None,
    region_to_country_mapping="data/input/rice50_regions_dict.json",
    title=None,
    output_titles=None,
    data_label="Emission Control Rate",
    legend_label=" ",
    colourmap=px.colors.sequential.Reds,
    projection="natural earth",
    scope="world",
    height=700,
    width=1200,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
    saving=False,
    data_normalization=True,
    show_colorbar=True,
    normalized_colorbar=False,
    tickvals=[0, 0.25, 0.5, 0.75, 1],
    ticktext=["0%", "25%", "50%", "75%", "100%"],
):

    # Assert if input_data list and output_titles list is None
    assert input_data_path_list, "No input data provided for visualization."

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    list_of_years = time_horizon.model_time_horizon
    data_loader = DataLoader()

    region_list = data_loader.REGION_LIST

    processed_data_dict = {}

    # Loop through the input data and plot the choropleth
    for plotting_idx, file in enumerate(input_data_path_list):

        # Load the data
        data = np.load(path_to_data + file)

        # Check if data is 3D
        if len(data.shape) == 3:
            print("Taking average over the last dimension.")
            data = np.mean(data, axis=2)

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

        # print("Global Min & Max", global_min, global_max)

        # Loop over the keys in the dictionary
        for idx in processed_data_dict.keys():
            print(idx)
            # Reshape the 'data' column to fit the scaler
            dataframe_to_normalize = processed_data_dict[(idx)]
            dataframe_to_normalize = dataframe_to_normalize[data_label].values.reshape(
                -1, 1
            )

            # Transform the 'data' column
            processed_data_dict[(idx)][data_label] = min_max_scaler(
                dataframe_to_normalize, global_min, global_max
            )

    # Loop through the input data and plot the choropleth
    for plotting_idx, file in enumerate(input_data_path_list):

        fig = px.choropleth(
            processed_data_dict[(plotting_idx)],
            locations="CountryCode",
            color=data_label,
            hover_name="CountryName",
            scope=scope,
            projection=projection,
            height=height,
            width=width,
            color_continuous_scale=colourmap,
        )

        # Set the discrete color scale and discrete legend
        if normalized_colorbar:
            n_colors = len(colourmap)
            discrete_values = [(i / (n_colors - 1)) for i in range(n_colors)]
            color_values = [
                (value, colourmap[i]) for i, value in enumerate(discrete_values)
            ]
            fig.update_layout(
                coloraxis=dict(
                    colorscale=color_values,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title=legend_label,
                        tickvals=tickvals,
                        ticktext=ticktext,
                        lenmode="pixels",  # Fixed length for discrete ticks
                        ticks="inside",
                        tickmode="array",
                    ),
                )
            )

        if not show_colorbar:
            fig.update_layout(coloraxis_showscale=False)

        if title:
            # Assert if len of input_data_path_list is not equal to len of output_titles
            assert len(input_data_path_list) == len(
                output_titles
            ), "Input data path list and output titles list are not equal."

            choropleth_title = title + output_titles[plotting_idx]

            fig.update_layout(
                title={
                    "text": choropleth_title,
                    "xanchor": "center",
                    "yanchor": "top",
                    "x": 0.5,
                    "y": 0.95,
                },
            )
        else:
            fig.update_layout(title_text="")

        # Policy index number
        filename = file.split(".")[0]
        # filename = (
        #     filename.split("_")[0]
        #     + filename.split("_")[1]
        #     + "_"
        #     + filename.split("_")[-1]
        # )

        output_file_name = filename
        if saving:
            fig.write_image(
                path_to_output
                + "/"
                + output_file_name
                + str(year_to_visualize)
                + ".svg"
            )

        fig.show()

    return fig, processed_data_dict


def process_economic_data_for_barchart(
    input_data_paths,
    region_mapping_path,
    rice_region_dict_path,
    start_year,
    end_year,
    splice_start_year,
    splice_end_year,
    data_timestep=5,
    timestep=1,
):
    """
    Process economic data for barchart by aggregating regional economic damages over a specified time horizon.

    Args:
        input_data_paths (list): List of file paths for different economic data to be loaded as numpy arrays.
        region_mapping_path (str): File path for the region mapping JSON.
        rice_region_dict_path (str): File path for the rice region dictionary JSON.
        start_year (int): The start year for the model time horizon.
        end_year (int): The end year for the model time horizon.
        splice_start_year (int): The start year for splicing the data.
        splice_end_year (int): The end year for splicing the data.
        data_timestep (int): The data time step, default is 5.
        timestep (int): The time step for year-to-timestep conversion, default is 1.

    Returns:
        list: A list of pandas DataFrames containing the aggregated economic data for each input data path.
    """

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    data_loader = DataLoader()

    # Load the region mapping from the JSON file
    with open(region_mapping_path, "r") as f:
        region_mapping_json = json.load(f)

    # Load the rice region dictionary from the JSON file
    with open(rice_region_dict_path, "r") as f:
        rice_50_dict_ISO3 = json.load(f)

    # Prepare the data container
    aggregated_dataframes = []

    # Iterate over each input data path to load and process data
    for data_path in input_data_paths:
        economic_data = np.load(data_path)

        # Aggregate the regions
        region_list, economic_data_aggregated = justice_region_aggregator(
            data_loader, region_mapping_json, economic_data
        )

        # Get year to time step
        start_time_step = time_horizon.year_to_timestep(
            splice_start_year, timestep=timestep
        )
        end_time_step = time_horizon.year_to_timestep(
            splice_end_year, timestep=timestep
        )

        # Slice the data
        economic_data_aggregated = economic_data_aggregated[
            :, start_time_step:end_time_step
        ]

        # Sum up over the years
        economic_data_aggregated = np.sum(economic_data_aggregated, axis=1)

        # Convert to DataFrame with region_list as columns
        economic_data_df = pd.DataFrame(economic_data_aggregated.T, columns=region_list)

        # Add to result list
        aggregated_dataframes.append(economic_data_df)

    return aggregated_dataframes


def plot_comparison_bar_chart_sorted(
    input_data_paths,
    path_to_output,
    output_file_name,
    region_mapping_path,
    rice_region_dict_path,
    start_year,
    end_year,
    splice_start_year,
    splice_end_year,
    data_timestep=5,
    timestep=1,
    bar_width=0.35,
    plot_height=600,
    plot_width=1200,
    color_palette=["salmon", "lightblue", "lightgreen", "orange", "purple"],
    datanames=["Utilitarian", "Prioritarian", "Egalitarian", "Sufficientarian"],
    plot_title=None,
    x_axis_title=None,
    y_axis_title=None,
    saving=False,
):
    """
    Plot a comparison bar chart for economic data between two different scenarios.

    Args:
        economic_dataframes (list): List of pandas DataFrames containing the economic data for each scenario.
        region_mapping_path (str): File path for the region mapping JSON.
        rice_region_dict_path (str): File path for the rice region dictionary JSON.
        start_year (int): The start year for the model time horizon.
        end_year (int): The end year for the model time horizon.
        splice_start_year (int): The start year for splicing the data.
        splice_end_year (int): The end year for splicing the data.
        data_timestep (int): The data time step, default is 5.
        timestep (int): The time step for year-to-timestep conversion, default is 1.
    """

    economic_dataframes = process_economic_data_for_barchart(
        input_data_paths=input_data_paths,
        region_mapping_path=region_mapping_path,
        rice_region_dict_path=rice_region_dict_path,
        start_year=start_year,
        end_year=end_year,
        splice_start_year=splice_start_year,
        splice_end_year=splice_end_year,
    )

    # Loop through the economic dataframes to calculate means and standard deviations
    means = []
    stds = []

    for idx, economic_data in enumerate(economic_dataframes):
        # Calculate mean and standard deviation for each region
        mean = economic_data.mean(axis=0)
        std = economic_data.std(axis=0)

        # Append to the lists
        means.append(mean)
        stds.append(std)

    # Define the regions
    regions = means[0].index

    # Calculate the differences for sorting
    max_min_diffs = np.max(means, axis=0) - np.min(means, axis=0)
    sorted_indices = np.argsort(-max_min_diffs)
    sorted_regions = regions[sorted_indices]

    # Sort means and stds according to the sorted region indices
    sorted_means = [mean.iloc[sorted_indices] for mean in means]
    sorted_stds = [std.iloc[sorted_indices] for std in stds]

    # Create the figure by looping through the dataframes and add_trace for each formulation
    fig = go.Figure()

    for idx, sorted_mean in enumerate(sorted_means):
        # Add bars for the mean
        fig.add_trace(
            go.Bar(
                x=sorted_regions,
                y=sorted_mean,
                error_y=dict(type="data", array=sorted_stds[idx]),
                name=datanames[idx],
                marker_color=color_palette[idx],
            )
        )

    # Update layout
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        barmode="group",
        xaxis_tickangle=-45,
        template="plotly_white",
    )
    # Remove gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Show y axis with out ticks
    fig.update_yaxes(
        showticklabels=True,
        showline=True,
        linewidth=1,
        linecolor="black",
        ticks="outside",
    )
    # Set the figure size
    fig.update_layout(width=plot_width, height=plot_height)

    # Save the plot
    if saving:
        # Assert if path_to_output and output_file_name is None
        assert path_to_output, "No path to output provided for saving."
        assert output_file_name, "No output file name provided for saving."

        fig.write_image(path_to_output + "/" + output_file_name + ".svg")

    return fig


def plot_comparison_bar_chart(
    input_data_paths,
    region_mapping_path,
    rice_region_dict_path,
    start_year,
    end_year,
    splice_start_year,
    splice_end_year,
    data_timestep=5,
    timestep=1,
    bar_width=0.35,
    plot_height=600,
    plot_width=1200,
    color_palette=["salmon", "lightblue", "lightgreen", "orange", "purple"],
    datanames=["Utilitarian", "Prioritarian", "Egalitarian", "Sufficientarian"],
    plot_title=None,
    x_axis_title=None,
    y_axis_title=None,
):
    """
    Plot a comparison bar chart for economic data between two different scenarios.

    Args:
        economic_dataframes (list): List of pandas DataFrames containing the economic data for each scenario.
        region_mapping_path (str): File path for the region mapping JSON.
        rice_region_dict_path (str): File path for the rice region dictionary JSON.
        start_year (int): The start year for the model time horizon.
        end_year (int): The end year for the model time horizon.
        splice_start_year (int): The start year for splicing the data.
        splice_end_year (int): The end year for splicing the data.
        data_timestep (int): The data time step, default is 5.
        timestep (int): The time step for year-to-timestep conversion, default is 1.
    """

    economic_dataframes = process_economic_data_for_barchart(
        input_data_paths=input_data_paths,
        region_mapping_path=region_mapping_path,
        rice_region_dict_path=rice_region_dict_path,
        start_year=start_year,
        end_year=end_year,
        splice_start_year=splice_start_year,
        splice_end_year=splice_end_year,
    )

    # Loop through the economic dataframes to calculate means and standard deviations
    means = []
    stds = []

    for idx, economic_data in enumerate(economic_dataframes):
        # Calculate mean and standard deviation for each region
        mean = economic_data.mean(axis=0)
        std = economic_data.std(axis=0)

        # Append to the lists
        means.append(mean)
        stds.append(std)

    # Define the regions
    regions = means[0].index

    # Create the figure by looping through the dataframes and add_trace for each formulation
    fig = go.Figure()

    for idx, mean in enumerate(means):
        # Add bars for the mean
        fig.add_trace(
            go.Bar(
                x=regions,
                y=mean,
                error_y=dict(type="data", array=stds[idx]),
                name=datanames[idx],
                marker_color=color_palette[idx],
            )
        )

    # Update layout
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        barmode="group",
        xaxis_tickangle=-45,
        template="plotly_white",
    )
    # Remove gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # Set the figure size
    fig.update_layout(width=plot_width, height=plot_height)

    return fig


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


def plot_stacked_area_chart_with_baseline_emissions(
    variable_name=None,
    region_name_path="data/input/rice50_region_names.json",
    path_to_data="data/temporary",
    path_to_output="./data/plots",
    baseline_emissions_path="data/temporary/baseline_emissions_16.npy",
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
    fontsize=16,
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

            if baseline_emissions_path:
                baseline_emissions = np.load(baseline_emissions_path)

                print("Baseline emissions shape: ", baseline_emissions.shape)
                print("Data shape: ", data.shape)

            # Check if region_aggegation is True
            if region_aggegation:
                # Aggregated Input Data
                region_list, data = justice_region_aggregator(
                    data_loader=data_loader, region_config=region_dict, data=data
                )

                if baseline_emissions_path:

                    region_list, baseline_emissions = justice_region_aggregator(
                        data_loader=data_loader,
                        region_config=region_dict,
                        data=baseline_emissions,
                    )
                    # Take mean over the ensemble dimension
                    baseline_emissions = np.mean(baseline_emissions, axis=2)

                    # Convert to dataframe
                    baseline_emissions = pd.DataFrame(
                        baseline_emissions, index=region_list, columns=list_of_years
                    )

                    # Create the slice according to visualization years
                    baseline_emissions = baseline_emissions.loc[
                        :, visualization_start_year:visualization_end_year
                    ]

            # Check shape of data
            if len(data.shape) == 3:
                data = np.mean(data, axis=2)

            # Create a dataframe from the data
            data = pd.DataFrame(data, index=region_list, columns=list_of_years)

            # Create the slice according to visualization years
            data = data.loc[:, visualization_start_year:visualization_end_year]

            if baseline_emissions_path:
                abated_emissions = baseline_emissions - data

                # Update the name of regions in the abated_emissions dataframe by adding _abated
                abated_emissions.index = [
                    region + "_abated" for region in abated_emissions.index
                ]

                # Concatenate the dataframes abaated_emissions and data but keep the similar region names together
                data = pd.concat([data, abated_emissions])

                # # Use string similarity to sort the regions
                # data = data.reindex(sorted(data.index, key=lambda x: x.split("_")[0]))

                # Shape of the data

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
                # pattern_shape=data.index,
                # Pattern Shape sequence for only the abated emissions
                # pattern_shape_sequence=["x", None, "x", None, "x", None, "x", None, "x", None, "x", None, "x", None, "x", None, "x", None],
                # pattern_shape_sequence=["x"],
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

            # Remove gridlines
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Show ticks on x and y axes with axis lines
            fig.update_xaxes(
                showline=True, linewidth=1, linecolor="black", ticks="outside"
            )
            fig.update_yaxes(
                showline=True, linewidth=1, linecolor="black", ticks="outside"
            )

            # Set fontsize for tick labels
            fig.update_xaxes(tickfont=dict(size=fontsize))

            if saving:
                # Save the figure
                if not os.path.exists(path_to_output):
                    os.makedirs(path_to_output)

                output_file_name = (
                    variable_name + "_" + output_titles[idx] + "_" + scenario
                )
                print("Saving plot for: ", scenario, " - ", output_file_name)
                fig.write_image(
                    path_to_output + "/" + output_file_name + "_v2_with_abated" + ".svg"
                )

    return fig, data, abated_emissions


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
    fontsize=16,
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

            # Remove gridlines
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Show ticks on x and y axes with axis lines
            fig.update_xaxes(
                showline=True, linewidth=1, linecolor="black", ticks="outside"
            )
            fig.update_yaxes(
                showline=True, linewidth=1, linecolor="black", ticks="outside"
            )

            # Set fontsize for tick labels
            fig.update_xaxes(tickfont=dict(size=fontsize))

            if saving:
                # Save the figure
                if not os.path.exists(path_to_output):
                    os.makedirs(path_to_output)

                output_file_name = (
                    variable_name + "_" + output_titles[idx] + "_" + scenario
                )
                print("Saving plot for: ", scenario, " - ", output_file_name)
                fig.write_image(
                    path_to_output + "/" + output_file_name + "_v1" + ".svg"
                )

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
            fig.write_image(path_to_output + "/" + output_file_name + ".svg")

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
