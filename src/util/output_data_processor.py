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
import os
import h5py
from ema_workbench import load_results, ema_logging
import pandas as pd

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

    # Print the working directory with os
    print("Directory: ", os.getcwd())

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

            # TODO: Change from pickle to hdf5
            # Save the processed data as a pickle file
            with open(path_to_output + "/" + output_file_name, "wb") as f:
                pickle.dump(processed_data, f)

            # Print file saved as filename at location path
            print(f"File saved as {output_file_name} at location {path_to_output}")


def reevaluate_optimal_policy(
    input_data=[],
    scenario_list=[],
    list_of_objectives=[],
    direction_of_optimization=[],
    objective_of_interest=None,
    lowest_n_percent=None,
    path_to_rbf_weights=None,
    path_to_output=None,
    n_inputs_rbf=2,
    max_annual_growth_rate=0.04,
    emission_control_start_timestep=10,
    min_emission_control_rate=0.01,
    max_temperature=16.0,
    min_temperature=0.0,
    max_difference=2.0,
    min_difference=0.0,
):
    """
    Function to generate data for the optimal policy. It runs JUSTICE on the optimal policy and saves the data as a pickle file.

    @param input_data: List of input data files
    @param scenario_list: List of SSP scenarios e.g. ['SSP534', 'SSP585']
    @param list_of_objectives: List of objectives to optimize. This is only for finding the Pareto optimal policies for ALL objectives [Use this if not using objective_of_interest]
    @param direction_of_optimization: List of directions of optimization for the objectives. Needed to filter the Pareto optimal policies
    @param objective_of_interest: Objective of interest to optimize. This is only for finding the optimal policy for a single objective [Use this if not using list_of_objectives]
    @param lowest_n_percent: Percentage of the lowest n percent of the data to consider. It takes the lowest or highest proportion of the data based on the direction of optimization
    @param path_to_rbf_weights: Path to the RBF weights
    @param path_to_output: Path to save the output data
    @param n_inputs_rbf: Number of inputs for the RBF
    @param max_annual_growth_rate: Maximum annual growth rate
    @param emission_control_start_timestep: Emission control start timestep
    @param min_emission_control_rate: Minimum emission control rate
    @param max_temperature: Maximum temperature
    @param min_temperature: Minimum temperature
    @param max_difference: Maximum difference
    @param min_difference: Minimum difference

    """
    # Assert if any arguments are None
    assert input_data is not None, "Input data not provided"
    assert path_to_rbf_weights is not None, "Path to RBF weights not provided"
    assert path_to_output is not None, "Path to output not provided"

    path_to_output = path_to_output  # "data/reevaluation/"
    # Initialize the rbf_policy_index
    rbf_policy_index = None
    # Loop through the elements in input_data
    for input_data_index, file in enumerate(input_data):

        rival_framing = file  # input_data[index]
        output_file_name = file.split(".")[0]  # input_data[index]

        path = path_to_rbf_weights + rival_framing  #

        df = pd.read_csv(path)

        # Select the column of interest
        if list_of_objectives != []:
            # Assert if direction of optimization is not provided
            assert (
                direction_of_optimization != []
            ), "Direction of optimization not provided"

            list_of_pareto_optimal_policies = get_best_performing_policies(
                input_data=[file],
                lowest_n_percent=lowest_n_percent,
                data_path="data/optimized_rbf_weights/tradeoffs",
                list_of_objectives=[
                    "welfare",
                    "years_above_temperature_threshold",
                    "welfare_loss_damage",
                    "welfare_loss_abatement",
                ],
                direction_of_optimization=direction_of_optimization,
            )
            for _, pareto_optimal_policy_index in enumerate(
                list_of_pareto_optimal_policies[0]
            ):
                rbf_policy_index = pareto_optimal_policy_index
                print(
                    "list of index for Pareto Optimal Policies: ",
                    rbf_policy_index,
                )

                scenario_datasets = run_model_with_optimal_policy(
                    scenario_list=scenario_list,
                    path_to_rbf_weights=path_to_rbf_weights + file,
                    saving=False,
                    output_file_name=None,
                    rbf_policy_index=rbf_policy_index,
                    n_inputs_rbf=n_inputs_rbf,
                    max_annual_growth_rate=max_annual_growth_rate,
                    emission_control_start_timestep=emission_control_start_timestep,
                    min_emission_control_rate=min_emission_control_rate,
                    max_temperature=max_temperature,
                    min_temperature=min_temperature,
                    max_difference=max_difference,
                    min_difference=min_difference,
                )

                output_file_name = output_file_name + "_idx" + str(rbf_policy_index)
                with open(path_to_output + output_file_name + ".pkl", "wb") as f:
                    pickle.dump(scenario_datasets, f)

                    # for key in scenario_datasets.keys():
                    #     # Save the processed data as a pickle file
                    #     with open(
                    #         path_to_output + output_file_name + "_" + key + ".pkl", "wb"
                    #     ) as f:
                    #         pickle.dump(scenario_datasets[key], f)

                    # # Now save in hdf5 format
                    # with h5py.File(path_to_output + output_file_name + ".h5", "w") as f:
                    #     for key in scenario_datasets.keys():
                    #         f.create_dataset(key, data=scenario_datasets[key])

                print(f"File saved as {output_file_name} at location {path_to_output}")

        elif objective_of_interest is not None and list_of_objectives == []:
            # Choose column in df by index
            rbf_policy_index = df[objective_of_interest].idxmin()
            print("index for obj of interest: ", rbf_policy_index)

            scenario_datasets = run_model_with_optimal_policy(
                scenario_list=scenario_list,
                path_to_rbf_weights=path_to_rbf_weights + file,
                saving=False,
                output_file_name=None,
                rbf_policy_index=rbf_policy_index,
                n_inputs_rbf=n_inputs_rbf,
                max_annual_growth_rate=max_annual_growth_rate,
                emission_control_start_timestep=emission_control_start_timestep,
                min_emission_control_rate=min_emission_control_rate,
                max_temperature=max_temperature,
                min_temperature=min_temperature,
                max_difference=max_difference,
                min_difference=min_difference,
            )
            output_file_name = output_file_name + "_idx" + str(rbf_policy_index)

            with open(path_to_output + output_file_name + ".pkl", "wb") as f:
                pickle.dump(scenario_datasets, f)

            # for key in scenario_datasets.keys():
            #     # Save the processed data as a pickle file
            #     with open(
            #         path_to_output + output_file_name + "_" + key + ".pkl", "wb"
            #     ) as f:
            #         pickle.dump(scenario_datasets[key], f)
            #         # Print file saved as filename at location path

            print(f"File saved as {output_file_name} at location {path_to_output}")


def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as f:
        data = {}
        for key in f.keys():
            data[key] = f[key][:]
    return data


def run_model_with_optimal_policy(
    scenario_list=[],
    path_to_rbf_weights=None,
    saving=False,
    output_file_name=None,
    rbf_policy_index=None,
    n_inputs_rbf=2,
    max_annual_growth_rate=0.04,
    emission_control_start_timestep=10,
    min_emission_control_rate=0.01,
    max_temperature=16.0,
    min_temperature=0.0,
    max_difference=2.0,
    min_difference=0.0,
):
    # Create a dictionary to store the data for each scenario
    scenario_data = {}

    for _, scenarios in enumerate(scenario_list):
        scneario_idx = Scenario[scenarios].value[0]
        print(scneario_idx, scenarios)

        scenario_data[scenarios] = JUSTICE_stepwise_run(
            scenarios=scneario_idx,
            path_to_rbf_weights=path_to_rbf_weights,
            saving=saving,
            output_file_name=output_file_name,
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

        print("Keys of the scenario data: ", scenario_data.keys())
    return scenario_data


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


def get_best_performing_policies(
    input_data=[],
    direction_of_optimization=[],  # ["min", "min", "min", "min"],
    lowest_n_percent=0.51,
    data_path="data/optimized_rbf_weights/tradeoffs",
    list_of_objectives=[
        "welfare",
        "years_above_temperature_threshold",
        "welfare_loss_damage",
        "welfare_loss_abatement",
    ],
):
    # Assert if number of objectives is not equal to the number of directions of optimization
    assert len(list_of_objectives) == len(
        direction_of_optimization
    ), "Number of objectives not equal to number of directions of optimization"

    indices_list = []
    for file in input_data:
        df = pd.read_csv(data_path + "/" + file)
        df = df.iloc[:, -len(list_of_objectives) :]
        indices_per_objective = []
        indices_per_problem_formulation = []

        for idx, objective in enumerate(list_of_objectives):

            # print("objective: ", objective)
            # print("idx: ", idx)
            # print("direction_of_optimization: ", direction_of_optimization[idx])

            if direction_of_optimization[list_of_objectives.index(objective)] == "min":
                indices_per_objective = (
                    df[objective]
                    .nsmallest(int(lowest_n_percent * len(df)))
                    .index.tolist()
                )

            elif (
                direction_of_optimization[list_of_objectives.index(objective)] == "max"
            ):
                indices_per_objective = (
                    df[objective]
                    .nlargest(int(lowest_n_percent * len(df)))
                    .index.tolist()
                )

            indices_per_problem_formulation.append(indices_per_objective)

        # Use intersection to get the common indices in indices_list
        indices_list.append(
            list(set.intersection(*map(set, indices_per_problem_formulation)))
        )
    return indices_list


if __name__ == "__main__":
    # reevaluate_optimal_policy(
    #     input_data=[
    #         # "UTIL_100024_s1644652.csv",
    #         # "PRIOR_101400_s1644652.csv",
    #         "SUFF_100090_s9845531.csv",
    #         "EGAL_100417_s1644652.csv",
    #     ],  # "UTIL_100049.csv", "EGAL_101948.csv",
    #     path_to_rbf_weights="data/optimized_rbf_weights/tradeoffs/",
    #     path_to_output="data/reevaluation/",
    #     objective_of_interest="welfare",
    #     # direction_of_optimization=[
    #     #     "min",
    #     #     "min",
    #     #     "max",
    #     #     "max",
    #     # ],
    #     lowest_n_percent=0.51,
    #     # list_of_objectives=[
    #     #     "welfare",
    #     #     "years_above_temperature_threshold",
    #     #     "welfare_loss_damage",
    #     #     "welfare_loss_abatement",
    #     # ],
    #     scenario_list=list(Scenario.__members__.keys()),
    # )

    ########################################################################
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
        variable_name="net_economic_output",  # "economic_damage",  # "emissions", #abatement_cost, # "global_temperature",
        data_shape=3,
        no_of_ensembles=1001,
        input_data=[
            "UTIL_100024_s1644652_idx75.pkl",
            "PRIOR_101400_s1644652_idx236.pkl",
            "SUFF_100090_s9845531_idx86.pkl",
            "EGAL_100417_s1644652_idx31.pkl",
        ],
        output_file_names=[
            "Utilitarian",
            "Prioritarian",
            "Sufficientarian",
            "Egalitarian",
        ],
    )
