import sys

sys.path.append("justice/")

import os
import pickle
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import pandas as pd
import numpy as np
import json
import pycountry
from justice.objectives.objective_functions import calculate_gini_index_c1

from constants import (
    policy_color_dict,
    policy_name_dict,
    variable_to_label_dict,
)


CLUSTER_SIZES = [
    6,
    4,
    2,
    6,
    5,
    6,
    1,
    1,
    1,
    1,
    1,
    23,
]  # Number of regions in each cluster
CLIP_IDX = -1


def moving_average(arr: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(arr, np.ones(n) / n, mode="valid")


def load_rice_data(current_working_dir=None):

    RICE_DATA_DIR = os.path.join(current_working_dir, "rice-data")

    rice_abated_emissions_data = pd.read_csv(
        os.path.join(RICE_DATA_DIR, "rice_50_abated_emissions.csv"), delimiter=";"
    )

    rice_abated_emissions_data = rice_abated_emissions_data.iloc[:, 1:-1]

    rice_abated_emissions_data = rice_abated_emissions_data.to_numpy()

    rice_emissions_data = pd.read_csv(
        os.path.join(RICE_DATA_DIR, "rice_50_emissions.csv"), delimiter=";"
    )

    rice_emissions_data = rice_emissions_data.iloc[:, 1:-1]

    rice_emissions_data = rice_emissions_data.to_numpy()

    rice_global_temperature_data = pd.read_csv(
        os.path.join(RICE_DATA_DIR, "rice_50_global_temperature.csv"), delimiter=";"
    )

    rice_global_temperature_data = rice_global_temperature_data.iloc[:, 1:-1]

    rice_global_temperature_data = rice_global_temperature_data.to_numpy()

    rice_net_economic_output_data = pd.read_csv(
        os.path.join(RICE_DATA_DIR, "rice_50_net_economic_output.csv"), delimiter=";"
    )

    rice_net_economic_output_data = rice_net_economic_output_data.iloc[:, 1:-1]

    rice_net_economic_output_data = rice_net_economic_output_data.to_numpy()

    return (
        rice_abated_emissions_data,
        rice_emissions_data,
        rice_global_temperature_data,
        rice_net_economic_output_data,
    )


def plot_choropleth_2D_data(
    data,
    save_path,
    region_list,
    region_to_country_mapping,
    title=None,
    data_label="Emission Control Rate",
    legend_label=" ",
    colourmap=px.colors.sequential.Reds,
    projection="natural earth",
    scope="world",
    height=700,
    width=1200,
    show_colorbar=True,
    tickvals=[0, 0.25, 0.5, 0.75, 1],
    ticktext=["0%", "25%", "50%", "75%", "100%"],
    range_color=None,
    mode="save",
):
    # Process the provided data
    processed_country_data = process_2D_regional_data_for_choropleth_plot(
        region_list=region_list,
        data=data,
        data_label=data_label,
        region_to_country_mapping=region_to_country_mapping,
    )

    # Create choropleth plot
    fig = px.choropleth(
        processed_country_data,
        locations="CountryCode",
        color=data_label,
        hover_name="CountryName",
        scope=scope,
        projection=projection,
        height=height,
        width=width,
        color_continuous_scale=colourmap,
        range_color=range_color,  # Set min and max for the color bar
    )

    if not show_colorbar:
        fig.update_layout(coloraxis_showscale=False)

    fig.update_layout(
        title={
            "text": title,
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
            "y": 0.95,
        },
    )

    fig.write_image(save_path)

    if mode == "show":
        fig.show()


def process_2D_regional_data_for_choropleth_plot(
    region_list,
    data,
    data_label="Emission Control Rate",
    region_to_country_mapping=None,
    data_correction=True,
):
    # Validate input arguments
    assert region_list is not None, "Region list is not provided."
    assert data is not None, "Data is not provided."
    assert (
        region_to_country_mapping is not None
    ), "Region to country mapping is not provided."

    # Create DataFrame
    data_scenario_year = pd.DataFrame(
        {data_label: data}, index=region_list
    ).reset_index()
    data_scenario_year.columns = ["Region", data_label]

    # Load region-to-country mapping
    with open(region_to_country_mapping) as json_file:
        region_to_country = json.load(json_file)

    # Convert mapping into a DataFrame
    mapping_df = pd.DataFrame(
        list(region_to_country.items()), columns=["Region", "CountryCode"]
    )

    # Merge data with country mapping
    data_scenario_year = pd.merge(
        mapping_df, data_scenario_year, on="Region", how="inner"
    )

    # Handle multiple country codes per region
    data_scenario_year = data_scenario_year.explode("CountryCode")

    # Precompute country names
    country_code_to_name = {
        country.alpha_3: country.name for country in pycountry.countries
    }
    data_scenario_year["CountryName"] = data_scenario_year["CountryCode"].map(
        country_code_to_name
    )

    # Apply data corrections
    if data_correction:
        data_scenario_year.loc[
            data_scenario_year["CountryCode"] == "ATA", data_label
        ] = np.nan
        data_scenario_year.loc[
            data_scenario_year["CountryCode"] == "KSV", "CountryName"
        ] = "Kosovo"

    return data_scenario_year[["CountryCode", data_label, "CountryName"]]


def aggregate_data_across_seeds(evaluation_data):
    aggregated_data = {}

    for policy in evaluation_data.keys():
        aggregated_policy_data = {}
        std_policy_data = {}
        policy_data = evaluation_data[policy]

        # Extract the list of keys (seeds) and the sub-keys (variables)
        seeds = list(policy_data.keys())
        variables = list(policy_data[seeds[0]].keys())

        # Iterate over each variable and compute the mean and standard deviation across seeds
        for var in variables:
            arrays = np.array(
                [policy_data[seed][var] for seed in seeds]
            )  # Shape: (num_seeds, ...)

            # Compute mean and standard deviation along the first axis (seeds)
            aggregated_policy_data[var] = np.mean(arrays, axis=0)
            std_policy_data[var] = np.std(arrays, axis=0)

            if var in ["emissions", "net_economic_output"]:
                for i, cluster_size in enumerate(CLUSTER_SIZES):
                    aggregated_policy_data[var][i, :] = (
                        aggregated_policy_data[var][i, :] * cluster_size
                    )
                    std_policy_data[var][i, :] = (
                        std_policy_data[var][i, :] * cluster_size
                    )

        # Store both mean and std in the final dictionary
        aggregated_data[policy] = {
            "mean": aggregated_policy_data,
            "std": std_policy_data,
        }

    return aggregated_data


def plot_economic_output_over_time(
    aggregated_data, rice_data, current_working_dir=None, mode="save"
):

    for policy in aggregated_data.keys():
        policy_data = aggregated_data[policy]["mean"]
        std_policy_data = aggregated_data[policy]["std"]

        # we want to obtain a sum over the economic output taking the cluster size into account for each region

        economic_output = policy_data["net_economic_output"][:, :CLIP_IDX].sum(axis=0)
        std_economic_output = std_policy_data["net_economic_output"][:, :CLIP_IDX].sum(
            axis=0
        )

        # Plot the economic output over time
        plt.plot(
            economic_output,
            label=policy_name_dict[policy],
            color=policy_color_dict[policy],
        )
        plt.fill_between(
            np.arange(len(economic_output)),
            economic_output - std_economic_output,
            economic_output + std_economic_output,
            alpha=0.075,
            color=policy_color_dict[policy],
        )

    rice_data = rice_data.sum(axis=0)[1 : CLIP_IDX + 1 if CLIP_IDX != -1 else None]

    plt.plot(rice_data, label="RICE-50 Policy", color="red")

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Economic Output", fontsize=12)
    # plt.title("Total economic output")
    plt.legend()
    plt.xticks(ticks=[0, 135, 285], labels=[2015, 2150, 2300])

    # Save the plot
    if mode == "save":
        save_dir = os.path.join(current_working_dir, "eval_results/time-series/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(
            os.path.join(
                save_dir,
                "total_economic_output_over_time.svg",
            ),
            format="svg",
        )
    else:
        plt.show()

    plt.close()


def plot_total_emissions_over_time(
    aggregated_data, rice_data, current_working_dir=None, mode="save"
):

    for policy in aggregated_data.keys():
        policy_data = aggregated_data[policy]["mean"]
        std_policy_data = aggregated_data[policy]["std"]

        emissions = policy_data["emissions"][:, :CLIP_IDX].sum(axis=0)
        std_emissions = std_policy_data["emissions"][:, :CLIP_IDX].sum(axis=0)

        # Plot the emissions over time
        plt.plot(
            emissions, label=policy_name_dict[policy], color=policy_color_dict[policy]
        )
        plt.fill_between(
            np.arange(len(emissions)),
            emissions - std_emissions,
            emissions + std_emissions,
            alpha=0.075,
            color=policy_color_dict[policy],
        )

    rice_data = rice_data.sum(axis=0)[1 : CLIP_IDX + 1 if CLIP_IDX != -1 else None]

    plt.plot(rice_data, label="RICE-50 Policy", color="red")

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Emissions", fontsize=12)
    # plt.title("Total Emissions")
    plt.legend()
    plt.xticks(ticks=[0, 135, 285], labels=[2015, 2150, 2300])

    # Save the plot
    if mode == "save":
        save_dir = os.path.join(current_working_dir, "eval_results/time-series/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(
            os.path.join(
                save_dir,
                "total_emissions_over_time.svg",
            ),
            format="svg",
        )
    else:
        plt.show()
    plt.close()


def plot_average_global_temperature_over_time(
    aggregated_data, rice_data, current_working_dir=None, mode="save"
):

    for policy in aggregated_data.keys():
        policy_data = aggregated_data[policy]["mean"]
        std_policy_data = aggregated_data[policy]["std"]

        global_temperature = policy_data["global_temperature"][:CLIP_IDX]
        std_global_temperature = std_policy_data["global_temperature"][:CLIP_IDX]

        # Plot the global temperature over time
        plt.plot(
            global_temperature,
            label=policy_name_dict[policy],
            color=policy_color_dict[policy],
        )
        plt.fill_between(
            np.arange(len(global_temperature)),
            global_temperature - std_global_temperature,
            global_temperature + std_global_temperature,
            alpha=0.3,
            color=policy_color_dict[policy],
        )

    rice_data = rice_data.sum(axis=0)[1 : CLIP_IDX + 1 if CLIP_IDX != -1 else None]

    plt.plot(rice_data, label="RICE-50 Policy", color="red")

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Global Average Annual Temperature Raise", fontsize=12)
    # plt.title("Average Global Temperature")
    plt.legend()
    plt.xticks(ticks=[0, 135, 285], labels=[2015, 2150, 2300])

    # Save the plot
    if mode == "save":
        save_dir = os.path.join(current_working_dir, "eval_results/time-series/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(
            os.path.join(
                save_dir,
                "global_temperature_over_time.svg",
            ),
            format="svg",
        )
    else:
        plt.show()

    plt.close()


def plot_total_abated_emissions_over_time(
    aggregated_data, rice_data, current_working_dir=None, mode="save"
):

    for policy in aggregated_data.keys():
        policy_data = aggregated_data[policy]["mean"]
        std_policy_data = aggregated_data[policy]["std"]

        total_abated_emissions = policy_data["abated_emissions"][:, :CLIP_IDX].sum(
            axis=0
        )
        std_total_abated_emissions = std_policy_data["abated_emissions"][
            :, :CLIP_IDX
        ].sum(axis=0)

        # Plot the total abated emissions over time
        plt.plot(
            total_abated_emissions,
            label=policy_name_dict[policy],
            color=policy_color_dict[policy],
        )
        plt.fill_between(
            np.arange(len(total_abated_emissions)),
            total_abated_emissions - std_total_abated_emissions,
            total_abated_emissions + std_total_abated_emissions,
            alpha=0.075,
            color=policy_color_dict[policy],
        )

    rice_data = rice_data.sum(axis=0)[1 : CLIP_IDX + 1 if CLIP_IDX != -1 else None]

    plt.plot(rice_data, label="RICE-50 Policy", color="red")

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Abated Emissions", fontsize=12)
    # plt.title("Total Abated Emissions")
    plt.legend()
    plt.xticks(ticks=[0, 135, 285], labels=[2015, 2150, 2300])

    # Save the plot
    if mode == "save":
        save_dir = os.path.join(current_working_dir, "eval_results/time-series/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(
            os.path.join(
                save_dir,
                "total_abated_emissions_over_time.svg",
            ),
            format="svg",
        )
    else:
        plt.show()

    plt.close()


def create_csv_files(aggregated_data, current_working_dir=None):

    with open("justice/data/rice_12_regions_dict.json") as f:
        region_data = json.load(f)

    for policy_name in aggregated_data.keys():
        policy_data = aggregated_data[policy_name]["mean"]
        data_save_dir = os.path.join(
            current_working_dir, "eval_results", "csv-files", policy_name
        )

        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)

        for var in policy_data.keys():
            if var not in ["global_temperature"]:

                var_data = policy_data[var]  # [:, :86]  # Keeping data till 2100

                df = pd.DataFrame(
                    var_data,
                    index=list(region_data.keys()),
                    columns=list(range(2015, 2300)),
                )

                df.to_csv(os.path.join(data_save_dir, f"{var}.csv"))


def plot_regional_maps(abatement_policy_data, current_working_dir=None, mode="save"):
    evaluation_years = [35, 85]
    with open("justice/data/rice_12_regions_dict.json") as f:
        region_data = json.load(f)

    ranges_dict = {}

    for year in evaluation_years:
        range_min = (
            np.min(
                [
                    np.min(np.sum(policy_data[:, :year], axis=1))
                    for _, policy_data in abatement_policy_data.items()
                ]
            )
            - 1
        )
        range_max = (
            np.max(
                [
                    np.max(np.sum(policy_data[:, :year], axis=1))
                    for _, policy_data in abatement_policy_data.items()
                ]
            )
            + 1
        )

        ranges_dict[year] = {"min": range_min, "max": range_max}

    for policy_name in abatement_policy_data.keys():

        policy_data = abatement_policy_data[policy_name]
        data_save_dir = os.path.join(
            current_working_dir, "eval_results", "regional-data", policy_name
        )

        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)

        for evaluation_year in evaluation_years:  # This represents 2050 and 2100
            accumulated_abated_emissions = np.sum(
                policy_data[:, :evaluation_year], axis=1
            )

            plot_choropleth_2D_data(
                data=accumulated_abated_emissions,
                save_path=os.path.join(
                    data_save_dir, f"abated_emissions_{2015 + evaluation_year}.svg"
                ),
                region_list=list(region_data.keys()),
                region_to_country_mapping="justice/data/rice_12_regions_dict.json",
                title=f"Abated Emissions in {2015 + evaluation_year}",
                range_color=(
                    ranges_dict[evaluation_year]["min"],
                    ranges_dict[evaluation_year]["max"],
                ),
                mode=mode,
            )


def plot_gini_over_time(
    aggregated_data,
    rice_data,
    current_working_dir=None,
    mode="save",
    moving_average_n=None,
):
    gini_coefficient_keys = ["net_economic_output", "emissions", "abated_emissions"]

    if mode == "save":
        data_save_dir = os.path.join(current_working_dir, "eval_results", "gini")
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)

    for gini_key in gini_coefficient_keys:
        for policy in aggregated_data.keys():

            gini_data = calculate_gini_index_c1(
                aggregated_data[policy]["mean"][gini_key][:CLIP_IDX]
            )

            if moving_average_n:
                gini_data = moving_average(gini_data, n=moving_average_n)

            plt.plot(
                gini_data,
                label=policy_name_dict[policy],
                color=policy_color_dict[policy],
            )

        rice_gini_data = calculate_gini_index_c1(
            rice_data[gini_key][1 : CLIP_IDX + 1 if CLIP_IDX != -1 else None]
        )

        plt.plot(rice_gini_data, label="RICE-50", color="red")

        plt.xlabel("Year")
        plt.ylabel(f"GINI Index")
        plt.legend()
        plt.xticks(ticks=[0, 135, 285], labels=[2015, 2150, 2300])
        plt.title(f"GINI {variable_to_label_dict[gini_key]}")

        # Save the plot
        if mode == "save":
            plt.savefig(
                os.path.join(current_working_dir, data_save_dir, f"{gini_key}.svg"),
                format="svg",
            )
        else:
            plt.show()
        plt.close()