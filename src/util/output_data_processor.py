import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
from JUSTICE_example import JUSTICE_stepwise_run
from src.util.enumerations import *
import pickle
from src.util.enumerations import Scenario
from src.util.model_time import TimeHorizon
from src.util.data_loader import DataLoader

from ema_workbench import load_results, ema_logging

ema_logging.log_to_stderr(level=ema_logging.DEFAULT_LEVEL)


def reevaluated_optimal_policy_variable_extractor(
    scenario_list=None,
    region_list=None,
    list_of_years=None,
    path_to_data="data/reevaluation",
    path_to_output="data/reevaluation",
    variable_name=None,
    data_shape=None,
    no_of_ensembles=None,
    input_data=None,
    output_file_names=["Utilitarian", "Egalitarian", "Prioritarian", "Sufficientarian"],
):
    # Assert if any arguments are None
    assert scenario_list is not None, "Scenario list not provided"
    assert region_list is not None, "Region list not provided"
    assert list_of_years is not None, "List of years not provided"
    assert variable_name is not None, "Variable name not provided"
    assert data_shape is not None, "Data shape not provided"
    assert no_of_ensembles is not None, "Number of ensembles not provided"
    assert input_data is not None, "Input data not provided"

    # Create a if condition to check if the input variable is 2D or 3D
    if data_shape == 2:
        data_scenario = np.zeros(
            (len(scenario_list), len(list_of_years), no_of_ensembles)
        )
    elif data_shape == 3:
        data_scenario = np.zeros(
            (len(scenario_list), len(region_list), len(list_of_years), no_of_ensembles)
        )

    # Loop through the input data and plot the timeseries
    for plotting_idx, file in enumerate(input_data):

        # Get the string out of the input_data list
        file_name = input_data[plotting_idx]
        # Load the scenario data from the pickle file
        with open(
            path_to_data + "/" + file_name, "rb"
        ) as f:  # input_data[plotting_idx]
            scenario_data = pickle.load(f)

        for idx, scenarios in enumerate(scenario_list):
            print(scenarios)

            if data_shape == 2:
                data_scenario[idx, :, :] = scenario_data[scenarios][variable_name]
                processed_data = pd.DataFrame(
                    data_scenario[idx, :, :].T, columns=list_of_years
                )
            elif data_shape == 3:
                data_scenario[idx, :, :, :] = scenario_data[scenarios][variable_name]
                processed_data = data_scenario[idx, :, :, :]

            if output_file_names is None:
                output_file_name = (
                    input_data[plotting_idx].split(".")[0]
                    + "_"
                    + scenarios
                    + "_"
                    + variable_name
                    + ".pkl"
                )
            else:
                output_file_name = (
                    output_file_names[plotting_idx]
                    + "_"
                    + scenarios
                    + "_"
                    + variable_name
                    + ".pkl"
                )

            # Save the processed data as a pickle file
            with open(path_to_output + "/" + output_file_name, "wb") as f:
                pickle.dump(processed_data, f)

            # Print file saved as filename at location path
            print(f"File saved as {output_file_name} at location {path_to_output}")


def reevaluate_optimal_policy(
    input_data=[],
    output_data_name=None,
    path_to_rbf_weights=None,
    path_to_output=None,
    min=True,
    elasticity_of_marginal_utility_of_consumption=1.45,
    pure_rate_of_social_time_preference=0.015,
    inequality_aversion=0.5,
    n_inputs_rbf=2,
    max_annual_growth_rate=0.04,
    emission_control_start_timestep=10,
    min_emission_control_rate=0.01,
    max_temperature=16.0,
    min_temperature=0.0,
    max_difference=2.0,
    min_difference=0.0,
):
    # Assert if any arguments are None
    assert input_data is not None, "Input data not provided"
    assert path_to_rbf_weights is not None, "Path to RBF weights not provided"
    assert path_to_output is not None, "Path to output not provided"

    path_to_output = path_to_output  # "data/reevaluation/"
    # Loop through the elements in input_data
    for index, file in enumerate(input_data):

        rival_framing = file  # input_data[index]
        output_file_name = file.split(".")[0]  # input_data[index]

        path = path_to_rbf_weights + rival_framing  #

        df = pd.read_csv(path)
        rbf_policy_index = df["welfare_utilitarian"].idxmin()
        print(rbf_policy_index)
        # Create a dictionary to store the data for each scenario
        scenario_data = {}

        for idx, scenarios in enumerate(list(Scenario.__members__.keys())):
            print(index, idx, scenarios)

            scenario_data[scenarios] = JUSTICE_stepwise_run(
                scenarios=idx,
                elasticity_of_marginal_utility_of_consumption=elasticity_of_marginal_utility_of_consumption,
                pure_rate_of_social_time_preference=pure_rate_of_social_time_preference,
                inequality_aversion=inequality_aversion,
                path_to_rbf_weights=path,
                saving=False,
                output_file_name=None,
                rbf_policy_index=rbf_policy_index,
                n_inputs_rbf=n_inputs_rbf,
                max_annual_growth_rate=max_annual_growth_rate,
                emission_control_start_timestep=emission_control_start_timestep,
                min_emission_control_rate=min_emission_control_rate,
                allow_emission_fallback=False,  # Default is False
                endogenous_savings_rate=True,
                max_temperature=max_temperature,
                min_temperature=min_temperature,
                max_difference=max_difference,
                min_difference=min_difference,
            )

        # Save the scenario data as a dictionary
        with open(path_to_output + output_file_name + ".pkl", "wb") as f:
            pickle.dump(scenario_data, f)


def interpolator(data_array, data_time_horizon, model_time_horizon):
    # Check if data array is 3D
    if len(data_array.shape) == 3:
        interp_data = np.zeros(
            (
                data_array.shape[0],
                data_array.shape[1],
                len(model_time_horizon),
            )
        )

        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                f = interp1d(data_time_horizon, data_array[i, j, :], kind="linear")
                interp_data[i, j, :] = f(model_time_horizon)

        data_array = interp_data
    elif len(data_array.shape) == 2:
        interp_data = np.zeros(
            (
                data_array.shape[0],
                len(model_time_horizon),
            )
        )

        for i in range(data_array.shape[0]):
            f = interp1d(data_time_horizon, data_array[i, :], kind="linear")
            interp_data[i, :] = f(model_time_horizon)

        data_array = interp_data

    return data_array


def concatenate_data(number_of_runs, filename="optimal_open_exploration"):
    # create a list of filenames to load from
    filenames = [
        filename + "_" + str(number_of_runs) + "_" + metric
        for metric in ["5th", "median", "mean", "95th"]
    ]

    # Create arrays to store the data
    temperature_array = []
    damages_array = []
    disentangled_utility_array = []
    experiment_array = []

    # loop through filenames and load each file
    for filename in filenames:
        filepath = "./data/output/" + filename
        print(filepath)
        # load the data
        experiments, outcomes = load_results(filepath)
        experiments = experiments[
            [
                "elasticity_of_marginal_utility_of_consumption",
                "inequality_aversion",
                "pure_rate_of_social_time_preference",
                "ssp_rcp_scenario",
                "scenario",
                "policy",
            ]
        ]
        # experiments = experiments.iloc[:, [0, 1, 2, -4, -3, -2, -1]]
        temp = outcomes["global_temperature"]
        damages = outcomes["economic_damage"]
        dis_util = outcomes["disentangled_utility"]

        # Append the data to the arrays
        # Append the data to the lists
        temperature_array.append(temp)
        damages_array.append(damages)
        disentangled_utility_array.append(dis_util)
        experiment_array.append(experiments)

    # Convert the lists back to numpy arrays using concatenate
    temperature_array = np.concatenate(temperature_array, axis=0)
    damages_array = np.concatenate(damages_array, axis=0)
    disentangled_utility_array = np.concatenate(disentangled_utility_array, axis=0)
    experiment_array = np.concatenate(experiment_array, axis=0)

    return (
        temperature_array,
        damages_array,
        disentangled_utility_array,
        experiment_array,
    )


def calculate_welfare(
    experiments,
    disentangled_utility,
    time_horizon,
):
    # This is temporary.
    timestep_list = np.arange(
        0, len(time_horizon.model_time_horizon), time_horizon.timestep
    )
    pure_rate_of_social_time_preference = experiments[2]
    inequality_aversion = experiments[1]
    elasticity_of_marginal_utility_of_consumption = experiments[0]

    # print(pure_rate_of_social_time_preference, inequality_aversion, elasticity_of_marginal_utility_of_consumption)
    discount_rate = 1 / (
        np.power(
            (1 + pure_rate_of_social_time_preference),
            (time_horizon.timestep * (timestep_list)),
        )
    )

    # print(discount_rate)
    disentangled_utility_summed = np.sum(disentangled_utility, axis=0)
    # print(disentangled_utility_summed.shape)
    disentangled_utility_powered = np.power(
        disentangled_utility_summed,
        (
            (1 - elasticity_of_marginal_utility_of_consumption)
            / (1 - inequality_aversion)
        ),
    )

    disentangled_utility_regional_powered = np.power(
        disentangled_utility,
        (
            (1 - elasticity_of_marginal_utility_of_consumption)
            / (1 - inequality_aversion)
        ),
    )

    welfare_utilitarian = np.sum(
        (
            np.divide(
                disentangled_utility_powered,
                (1 - elasticity_of_marginal_utility_of_consumption),
            )
            - 1
        )
        * discount_rate,
        axis=0,
    )

    welfare_utilitarian_regional = np.sum(
        (
            np.divide(
                disentangled_utility_regional_powered,
                (1 - elasticity_of_marginal_utility_of_consumption),
            )
            - 1
        )
        * discount_rate,
        axis=1,
    )

    return welfare_utilitarian_regional, welfare_utilitarian  #


if __name__ == "__main__":
    # reevaluate_optimal_policy(
    #     input_data=[
    #         "PRIOR_101765.csv",
    #         "SUFF_102924.csv",
    #     ],  # "UTIL_100049.csv", "EGAL_101948.csv",
    #     output_data_name=None,
    #     path_to_rbf_weights="data/optimized_rbf_weights/tradeoffs/",
    #     path_to_output="data/reevaluation/",
    # )
    scenario_list = list(
        Scenario.__members__.keys()
    )  # ['SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585']
    start_year = 2015
    end_year = 2300
    data_timestep = 5
    timestep = 1

    data_loader = DataLoader()
    region_list = data_loader.REGION_LIST

    # Set the time horizon
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )

    list_of_years = time_horizon.model_time_horizon
    columns = list_of_years
    # net_economic_output consumption, emissions, economic_damage, global_temperature
    reevaluated_optimal_policy_variable_extractor(
        scenario_list=scenario_list,  # ['SSP245'],
        region_list=region_list,
        list_of_years=list_of_years,
        path_to_data="data/reevaluation",
        path_to_output="data/reevaluation",
        variable_name="emissions",  # "global_temperature",
        data_shape=3,
        no_of_ensembles=1001,
        input_data=[
            "UTIL_100049.pkl",
            "EGAL_101948.pkl",
            "PRIOR_101765.pkl",
            "SUFF_102924.pkl",
        ],
        output_file_names=[
            "Utilitarian",
            "Egalitarian",
            "Prioritarian",
            "Sufficientarian",
        ],
    )
