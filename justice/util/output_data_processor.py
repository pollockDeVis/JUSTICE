import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
from JUSTICE_example import JUSTICE_stepwise_run, JUSTICE_run_policy_index
from justice.util.enumerations import *
import pickle
from justice.util.enumerations import Scenario
from justice.util.model_time import TimeHorizon
from justice.util.data_loader import DataLoader
import os
import h5py
from ema_workbench import load_results, ema_logging
import pandas as pd
from justice.welfare.social_welfare_function import SocialWelfareFunction
from config.default_parameters import SocialWelfareDefaults
from justice.util.enumerations import get_economic_scenario
from justice.objectives.objective_functions import fraction_of_ensemble_above_threshold

ema_logging.log_to_stderr(level=ema_logging.DEFAULT_LEVEL)

# To run the code, you need to run the following command in the terminal to add the path to the PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/Users/palokbiswas/Desktop/pollockdevis_git/JUSTICE


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

    # Create empty dataframe to store the data named processed_data
    processed_data = pd.DataFrame()

    # Loop through the input data and plot the timeseries
    for plotting_idx, file in enumerate(input_data):

        # Get the string out of the input_data list
        file_name = input_data[plotting_idx]
        # Load the scenario data from the pickle file
        # with open(
        #     path_to_data + "/" + file_name, "rb"
        # ) as f:  # input_data[plotting_idx]
        #     scenario_data = pickle.load(f)
        scenario_data = {}
        # HDF5 file
        with h5py.File(path_to_data + "/" + file_name.split(".")[0] + ".h5", "r") as f:
            for scenario in f.keys():
                scenario_data[scenario] = {}
                scenario_group = f[scenario]
                for dataset in scenario_group.keys():
                    scenario_data[scenario][dataset] = np.array(scenario_group[dataset])

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
                # Construct the output file name using the base name, its last underscore part, the scenario, and the variable name.
                output_file_name = (
                    input_data[plotting_idx].split(".")[0]
                    + "_"
                    + input_data[plotting_idx].split(".")[0].split("_")[-1]
                    + "_"
                    + scenarios
                    + "_"
                    + variable_name
                )
            else:
                output_file_name = (
                    output_file_names[plotting_idx]
                    + "_"
                    + input_data[plotting_idx].split(".")[0].split("_")[-1]
                    + "_"
                    + scenarios
                    + "_"
                    + variable_name
                )

            # TODO: Change from pickle to hdf5
            # Save the processed data as a pickle file
            # with open(path_to_output + "/" + output_file_name, "wb") as f:
            #     pickle.dump(processed_data, f)

            # Save it as npy file
            np.save(
                path_to_output + "/" + output_file_name + ".npy",
                processed_data,
            )

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
    # Assert if direction of optimization is not provided
    assert direction_of_optimization != [], "Direction of optimization not provided"

    path_to_output = path_to_output  # "data/reevaluation/"

    # Loop through the elements in input_data
    for input_data_index, file in enumerate(input_data):

        rival_framing = file  # input_data[index]
        output_file_name = file.split(".")[0]  # input_data[index]

        path = path_to_rbf_weights + rival_framing  #

        df = pd.read_csv(path)

        # Select the column of interest #TODO: This should be moved to the top
        if objective_of_interest is None and rbf_policy_index is None:

            list_of_pareto_optimal_policies = get_best_performing_policies(
                input_data=[file],
                lowest_n_percent=lowest_n_percent,
                data_path=path_to_rbf_weights,  # "data/optimized_rbf_weights/tradeoffs",
                list_of_objectives=list_of_objectives,
                direction_of_optimization=direction_of_optimization,
            )

            # Print the length of the list of pareto optimal policies #TODO Remove this
            print(
                "Length of list of Pareto Optimal Policies: ",
                len(list_of_pareto_optimal_policies[0]),
            )
            for _, pareto_optimal_policy_index in enumerate(
                list_of_pareto_optimal_policies[0]
            ):
                rbf_policy_index = pareto_optimal_policy_index
                print(
                    "list of index for Pareto Optimal Policies: ",
                    rbf_policy_index,
                )

                scenario_datasets, _ = run_model_with_optimal_policy(
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
                # with open(path_to_output + output_file_name + ".pkl", "wb") as f:
                #     pickle.dump(scenario_datasets, f)

                # for key in scenario_datasets.keys():
                #     # Save the processed data as a pickle file
                #     with open(
                #         path_to_output + output_file_name + "_" + key + ".pkl", "wb"
                #     ) as f:
                #         pickle.dump(scenario_datasets[key], f)

                # Now save in hdf5 format
                with h5py.File(path_to_output + output_file_name + ".h5", "w") as f:
                    for scenario, arrays in scenario_datasets.items():
                        scenario_group = f.create_group(
                            scenario
                        )  # Create a group for each scenario
                        for key, array in arrays.items():
                            scenario_group.create_dataset(
                                key, data=array
                            )  # Save each array in its respective group

                print(f"File saved as {output_file_name} at location {path_to_output}")

        elif objective_of_interest is None and rbf_policy_index is not None:

            print("index for policy: ", rbf_policy_index)

            scenario_datasets, _ = run_model_with_optimal_policy(
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

            # with open(path_to_output + output_file_name + ".pkl", "wb") as f:
            #     pickle.dump(scenario_datasets, f)
            # Save as HDF5 file
            with h5py.File(path_to_output + output_file_name + ".h5", "w") as f:
                for scenario, arrays in scenario_datasets.items():
                    scenario_group = f.create_group(
                        scenario
                    )  # Create a group for each scenario
                    for key, array in arrays.items():
                        scenario_group.create_dataset(
                            key, data=array
                        )  # Save each array in its respective group
            print(f"File saved as {output_file_name} at location {path_to_output}")

        elif objective_of_interest is not None and rbf_policy_index is None:
            # Choose column in df by index
            rbf_policy_index = df[objective_of_interest].idxmin()
            print("index for obj of interest: ", rbf_policy_index)

            scenario_datasets, _ = run_model_with_optimal_policy(
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

            # with open(path_to_output + output_file_name + ".pkl", "wb") as f:
            #     pickle.dump(scenario_datasets, f)
            # Save as HDF5 file
            with h5py.File(path_to_output + output_file_name + ".h5", "w") as f:
                for scenario, arrays in scenario_datasets.items():
                    scenario_group = f.create_group(
                        scenario
                    )  # Create a group for each scenario
                    for key, array in arrays.items():
                        scenario_group.create_dataset(
                            key, data=array
                        )  # Save each array in its respective group
            print(f"File saved as {output_file_name} at location {path_to_output}")
            # for key in scenario_datasets.keys():
            #     # Save the processed data as a pickle file
            #     with open(
            #         path_to_output + output_file_name + "_" + key + ".pkl", "wb"
            #     ) as f:
            #         pickle.dump(scenario_datasets[key], f)
            #         # Print file saved as filename at location path

            print(f"File saved as {output_file_name} at location {path_to_output}")


def reevaluate_optimal_policy_with_temperature_threshold_filtering(
    input_data=[],
    scenario_list=[],
    list_of_objectives=[],
    direction_of_optimization=[],
    path_to_rbf_weights=None,
    path_to_output=None,
    rbf_policy_index=None,
    n_inputs_rbf=2,
    max_annual_growth_rate=0.04,
    emission_control_start_timestep=10,
    min_emission_control_rate=0.01,
    max_temperature=16.0,
    min_temperature=0.0,
    max_difference=2.0,
    min_difference=0.0,
    temperature_year_of_interest=2100,
    temperature_threshold=2.0,
    saving=False,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
):

    # Assert if any arguments are None
    assert input_data is not None, "Input data not provided"
    assert path_to_rbf_weights is not None, "Path to RBF weights not provided"
    assert path_to_output is not None, "Path to output not provided"
    # Assert if direction of optimization is not provided
    assert direction_of_optimization != [], "Direction of optimization not provided"

    path_to_output = path_to_output  # "data/reevaluation/"

    # Set the time horizon
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )

    temperature_year_of_interest_index = time_horizon.year_to_timestep(
        year=temperature_year_of_interest, timestep=timestep
    )

    # Loop through the elements in input_data
    for input_data_index, file in enumerate(input_data):

        output_file_name = file.split(".")[0]  # input_data[index]

        # TODO: Temporarily Commented Out
        # print("index for policy: ", rbf_policy_index)

        # Create a dictionary to store the data for each scenario
        scenario_data = {}

        for _, scenarios in enumerate(scenario_list):
            scenario_idx = Scenario[scenarios].value[0]
            # TODO: Temporarily Commented Out
            print("Scenario", scenario_idx, scenarios)

            data_dictionary, model = JUSTICE_stepwise_run(
                scenarios=scenario_idx,
                path_to_rbf_weights=path_to_rbf_weights + file,
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

            fraction_above_threshold = fraction_of_ensemble_above_threshold(
                temperature=data_dictionary["global_temperature"],
                temperature_year_index=temperature_year_of_interest_index,
                threshold=temperature_threshold,
            )

            # print(
            #     f"Fraction of ensemble members above threshold: {fraction_above_threshold}"
            # )

            # if fraction_above_threshold <= 0.5:
            # print(scenario_idx, scenarios)
            print("index for policy: ", rbf_policy_index)
            print(
                f"Fraction of ensemble members above threshold: {fraction_above_threshold}"
            )

            scenario_data[scenarios] = data_dictionary

        output_file_name = output_file_name + "_idx" + str(rbf_policy_index)

        if saving:
            # Save as HDF5 file
            with h5py.File(path_to_output + output_file_name + ".h5", "w") as f:
                for scenario, arrays in scenario_data.items():
                    scenario_group = f.create_group(
                        scenario
                    )  # Create a group for each scenario
                    for key, array in arrays.items():
                        scenario_group.create_dataset(
                            key, data=array
                        )  # Save each array in its respective group
            print(f"File saved as {output_file_name} at location {path_to_output}")
        # Reset the model
        model.reset()


#######################################################################################
#
#
##########################################################################################


def reevaluate_optimal_policy_for_robustness(
    model=None,
    filename=None,
    path_to_rbf_weights=None,
    path_to_output=None,
    rbf_policy_index=None,
    n_inputs_rbf=2,
    max_annual_growth_rate=0.04,
    emission_control_start_timestep=10,
    min_emission_control_rate=0.01,
    max_temperature=16.0,
    min_temperature=0.0,
    max_difference=2.0,
    min_difference=0.0,
    temperature_year_of_interest=2100,
    temperature_threshold=2.0,
):

    # Assert if any arguments are None
    assert filename is not None, "Input data not provided"
    assert path_to_rbf_weights is not None, "Path to RBF weights not provided"
    assert path_to_output is not None, "Path to output not provided"

    time_horizon = model.__getattribute__("time_horizon")
    data_loader = model.__getattribute__("data_loader")

    temperature_year_of_interest_index = time_horizon.year_to_timestep(
        year=temperature_year_of_interest, timestep=time_horizon.timestep
    )

    datasets = JUSTICE_run_policy_index(
        model=model,
        path_to_rbf_weights=path_to_rbf_weights + filename,
        rbf_policy_index=rbf_policy_index,
        time_horizon=time_horizon,
        data_loader=data_loader,
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

    fraction_above_threshold = fraction_of_ensemble_above_threshold(
        temperature=datasets["global_temperature"],
        temperature_year_index=temperature_year_of_interest_index,
        threshold=temperature_threshold,
    )

    global_temperature = datasets["global_temperature"][
        temperature_year_of_interest_index, :
    ]
    utilitarian_welfare = datasets["welfare_utilitarian"]
    prioritarian_welfare = datasets["welfare_prioritarian"]

    print("index for policy: ", rbf_policy_index)
    print(f"Fraction above threshold Reeval: {fraction_above_threshold}")
    # Print
    print("Utilitarian Welfare Reeval: ", utilitarian_welfare)
    print("Prioritarian Welfare Reeval: ", prioritarian_welfare)

    return (
        global_temperature,
        utilitarian_welfare,
        prioritarian_welfare,
    )


# def reevaluate_optimal_policy_for_robustness(
#     filename=None,
#     scenario=None,
#     list_of_objectives=[],
#     direction_of_optimization=[],
#     path_to_rbf_weights=None,
#     path_to_output=None,
#     rbf_policy_index=None,
#     n_inputs_rbf=2,
#     max_annual_growth_rate=0.04,
#     emission_control_start_timestep=10,
#     min_emission_control_rate=0.01,
#     max_temperature=16.0,
#     min_temperature=0.0,
#     max_difference=2.0,
#     min_difference=0.0,
#     temperature_year_of_interest=2100,
#     temperature_threshold=2.0,
#     saving=False,
#     start_year=2015,
#     end_year=2300,
#     data_timestep=5,
#     timestep=1,
# ):

#     # Assert if any arguments are None
#     assert filename is not None, "Input data not provided"
#     assert path_to_rbf_weights is not None, "Path to RBF weights not provided"
#     assert path_to_output is not None, "Path to output not provided"
#     # Assert if direction of optimization is not provided
#     assert direction_of_optimization != [], "Direction of optimization not provided"

#     path_to_output = path_to_output  # "data/reevaluation/"

#     # Set the time horizon
#     time_horizon = TimeHorizon(
#         start_year=start_year,
#         end_year=end_year,
#         data_timestep=data_timestep,
#         timestep=timestep,
#     )

#     temperature_year_of_interest_index = time_horizon.year_to_timestep(
#         year=temperature_year_of_interest, timestep=timestep
#     )

#     scenario_idx = Scenario[scenario].value[0]
#     # TODO: Temporarily Commented Out
#     print("Scenario", scenario_idx, scenario)

#     data_dictionary, model = JUSTICE_stepwise_run(
#         scenarios=scenario_idx,
#         path_to_rbf_weights=path_to_rbf_weights + filename,
#         saving=False,
#         output_file_name=None,
#         rbf_policy_index=rbf_policy_index,
#         n_inputs_rbf=n_inputs_rbf,
#         max_annual_growth_rate=max_annual_growth_rate,
#         emission_control_start_timestep=emission_control_start_timestep,
#         min_emission_control_rate=min_emission_control_rate,
#         allow_emission_fallback=False,  # Default is False
#         endogenous_savings_rate=True,
#         max_temperature=max_temperature,
#         min_temperature=min_temperature,
#         max_difference=max_difference,
#         min_difference=min_difference,
#     )

#     fraction_above_threshold = fraction_of_ensemble_above_threshold(
#         temperature=data_dictionary["global_temperature"],
#         temperature_year_index=temperature_year_of_interest_index,
#         threshold=temperature_threshold,
#     )

#     temperature_values_per_policy = data_dictionary["global_temperature"][
#         temperature_year_of_interest_index, :
#     ]
#     utilitarian_welfare = data_dictionary["welfare_utilitarian"]
#     prioritarian_welfare = data_dictionary["welfare_prioritarian"]

#     print("index for policy: ", rbf_policy_index)
#     print(f"Fraction above threshold Reeval: {fraction_above_threshold}")
#     # Print
#     print("Utilitarian Welfare Reeval: ", utilitarian_welfare)
#     print("Prioritarian Welfare Reeval: ", prioritarian_welfare)

#     return (
#         model,
#         temperature_values_per_policy,
#         utilitarian_welfare,
#         prioritarian_welfare,
#     )


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
    model_object = {}

    for _, scenarios in enumerate(scenario_list):
        scneario_idx = Scenario[scenarios].value[0]
        print(scneario_idx, scenarios)

        scenario_data[scenarios], model_object[scenarios] = JUSTICE_stepwise_run(
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
    return scenario_data, model_object


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


# Compares reevaluated Utilitarian and Prioritarian Pareto Front. Filters the best performing policies based on the top 10% of the Policies that are also coinciding with each other


def find_closest_pairs_of_pareto_solutions(
    utilitarian_data_path,
    prioritarian_data_path,
    column_of_interest=None,
    temperature_objective="years_above_temperature_threshold",
    columns_to_keep=[
        "welfare_utilitarian",
        "welfare_prioritarian",
        "years_above_temperature_threshold",
    ],
    direction="lowest",  # "highest" or "lowest",
):

    utilitarian_data = pd.read_csv(utilitarian_data_path)
    prioritarian_data = pd.read_csv(prioritarian_data_path)

    # Only keep the following columns "welfare_utilitarian", "welfare_prioritarian", "years_above_temperature_threshold"
    utilitarian_data = utilitarian_data[columns_to_keep]
    prioritarian_data = prioritarian_data[columns_to_keep]

    if direction == "lowest":
        # Filter the lowest 10% in prioritarian_data
        utilitarian_data_filtered = utilitarian_data[
            utilitarian_data[column_of_interest]
            <= utilitarian_data[column_of_interest].quantile(0.1)
        ]

        prioritarian_data_filtered = prioritarian_data[
            prioritarian_data[column_of_interest]
            <= prioritarian_data[column_of_interest].quantile(0.1)
        ]
    elif direction == "highest":
        # Filter the highest 10% in prioritarian_data
        utilitarian_data_filtered = utilitarian_data[
            utilitarian_data[column_of_interest]
            >= utilitarian_data[column_of_interest].quantile(0.9)
        ]

        prioritarian_data_filtered = prioritarian_data[
            prioritarian_data[column_of_interest]
            >= prioritarian_data[column_of_interest].quantile(0.9)
        ]

    min_diff = np.inf
    index_pairs = []

    # Round values to two decimal places
    df1_rounded = utilitarian_data_filtered[column_of_interest].round(9)
    df2_rounded = prioritarian_data_filtered[column_of_interest].round(9)

    # Collect pairs with the smallest difference
    for i, value1 in enumerate(df1_rounded):
        for j, value2 in enumerate(df2_rounded):
            diff = abs(value1 - value2)
            if diff < min_diff:
                min_diff = diff
                index_pairs = [
                    (
                        utilitarian_data_filtered.index[i],
                        prioritarian_data_filtered.index[j],
                    )
                ]  # Reset list with the new closest pair
            elif diff == min_diff:
                index_pairs.append(
                    (
                        utilitarian_data_filtered.index[i],
                        prioritarian_data_filtered.index[j],
                    )
                )

    # Print the welfare_utilitarian and welfare_prioritarian values for the indices
    print(
        f"Closest pairs of Pareto solutions with the same {column_of_interest} value:"
    )
    print(f"index pairs: {index_pairs}")
    for i, j in index_pairs:
        print(
            f"Utilitarian welfare_utilitarian: {utilitarian_data_filtered.loc[i, column_of_interest]}, "
            f"Prioritarian welfare_utilitarian: {prioritarian_data_filtered.loc[j, column_of_interest]}"
        )

    # Find the index for each dataframe on temperature_objective and get the index of the minimum value for the entire set
    utilitarian_temp_filtered = utilitarian_data[
        utilitarian_data[temperature_objective]
        <= utilitarian_data[temperature_objective].quantile(0.1)
    ]
    prioritarian_temp_filtered = prioritarian_data[
        prioritarian_data[temperature_objective]
        <= prioritarian_data[temperature_objective].quantile(0.1)
    ]

    temp1_rounded = utilitarian_temp_filtered[temperature_objective].round(9)
    temp2_rounded = prioritarian_temp_filtered[temperature_objective].round(9)

    min_diff_temp = np.inf
    temperature_index_pairs = []

    for i, value1 in enumerate(temp1_rounded):
        for j, value2 in enumerate(temp2_rounded):
            diff = abs(value1 - value2)
            if diff < min_diff_temp:
                min_diff_temp = diff
                temperature_index_pairs = [
                    (
                        utilitarian_temp_filtered.index[i],
                        prioritarian_temp_filtered.index[j],
                    )
                ]
            elif diff == min_diff_temp:
                temperature_index_pairs.append(
                    (
                        utilitarian_temp_filtered.index[i],
                        prioritarian_temp_filtered.index[j],
                    )
                )

    # Print the closest temperature objective pairs
    print(
        f"\nClosest pairs of Pareto solutions with respect to {temperature_objective} (lowest 10%):"
    )
    print(f"Index pairs: {temperature_index_pairs}")
    for i, j in temperature_index_pairs:
        print(
            f"Utilitarian {temperature_objective}: {utilitarian_temp_filtered.loc[i, temperature_objective]}, "
            f"Prioritarian {temperature_objective}: {prioritarian_temp_filtered.loc[j, temperature_objective]}"
        )

    # Print a newline
    print("\n")
    print("\n")

    return index_pairs, temperature_index_pairs


# Read the file 'UTILITARIAN_reference_set.csv' from the 'data/convergence_metrics' folder
def get_selected_policy_indices_based_on_welfare_temperature(
    rival_framings,
    data_dir,
    n_percent=0.1,
    number_of_objectives=4,
    suffix="_reference_set.csv",
    second_objective_of_interest="years_above_temperature_threshold",
):
    selected_indices = []
    for rival in rival_framings:
        file_path = f"{data_dir}/{rival}{suffix}"
        data = pd.read_csv(file_path)

        # Keep the last 4 columns of the data
        data = data.iloc[:, -number_of_objectives:]

        # Find the index of the lowest 10% of values in the 'welfare' column
        lowest_n_percent_indices = (
            data["welfare"].nsmallest(int(data.shape[0] * n_percent)).index
        )

        # Find the index of the lowest 'years_above_temperature_threshold' within the selected indices
        selected_policy_index = data.loc[
            lowest_n_percent_indices, second_objective_of_interest
        ].idxmin()
        print(f"Index of interest for {rival}: ", selected_policy_index)
        print(data.loc[selected_policy_index])

        selected_indices.append(selected_policy_index)
    return selected_indices


def select_policy_index_for_fraction_below_threshold(
    file_path, num_last_columns, fraction_column, threshold, objective_column
):
    """
    Reads the CSV file, selects a subset of columns, filters the rows
    based on the fraction_column being <= threshold, and returns the index
    of the policy with the minimum value in the objective_column.

    Parameters:
        file_path (str): Path to the CSV file.
        num_last_columns (int): Number of columns from the end to select.
        fraction_column (str): Column name for the fraction to check.
        threshold (float): Maximum allowed value for the fraction_column.
        objective_column (str): Column name whose minimum value determines the selection.

    Returns:
        numpy.int64: Index of the selected policy.
    """

    # Load the data
    data = pd.read_csv(file_path)
    # Select the last num_last_columns columns
    data_subset = data.iloc[:, -num_last_columns:]
    # Filter policies where fraction_column is less than or equal to threshold
    filtered = data_subset[data_subset[fraction_column] <= threshold]

    return list(filtered.index)


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


def reevaluate_all_for_utilitarian_prioritarian(
    input_data=[],
    path_to_rbf_weights="data/convergence_metrics/",
    path_to_output="data/reevaluation/",
    scenario_list=["SSP245"],  # NOTE: Only works with 1 scenario for now
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

    # Loop through the elements in input_data
    for input_data_index, file in enumerate(input_data):

        rival_framing = file  # input_data[index]
        output_file_name = file.split(".")[0]  # input_data[index]

        path = path_to_rbf_weights + rival_framing  #

        reference_set_df = pd.read_csv(path)  # The refset

        # Loop through the rows in input_data and set the rbf_policy_index
        for index, row in reference_set_df.iterrows():
            rbf_policy_index = index
            print("File: ", file)
            print("index for policy: ", rbf_policy_index)

            scenario_datasets, model_objects = run_model_with_optimal_policy(
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

            # Print scenario datasets keys
            print("Keys of the scenario data: ", scenario_datasets.keys())
            print(
                "Shape of CPC",
                (scenario_datasets[scenario_list[0]]["consumption_per_capita"]).shape,
            )
            model = model_objects[scenario_list[0]]

            # Accessing attributes using dot notation or getattr
            time_horizon = getattr(model, "time_horizon", None)  # or model.time_horizon
            data_loader = getattr(model, "data_loader", None)  # or model.data_loader
            no_of_ensembles = getattr(
                model, "no_of_ensembles", None
            )  # or model.no_of_ensembles

            # Using the attributes to get further information
            n_regions = len(data_loader.REGION_LIST) if data_loader else 0
            n_timesteps = len(time_horizon.model_time_horizon) if time_horizon else 0
            population = model.economy.get_population() if model.economy else None
            social_welfare_defaults = SocialWelfareDefaults()

            # Fetch the defaults for Social Welfare Function
            welfare_defaults_utilitarian = social_welfare_defaults.get_defaults(
                WelfareFunction.UTILITARIAN.name
            )

            # Fetch the defaults for Social Welfare Function
            welfare_defaults_prioritarian = social_welfare_defaults.get_defaults(
                WelfareFunction.PRIORITARIAN.name
            )

            welfare_function_utilitarian = SocialWelfareFunction(
                input_dataset=data_loader,
                time_horizon=time_horizon,
                climate_ensembles=no_of_ensembles,
                population=population,
                risk_aversion=welfare_defaults_utilitarian["risk_aversion"],
                elasticity_of_marginal_utility_of_consumption=welfare_defaults_utilitarian[
                    "elasticity_of_marginal_utility_of_consumption"
                ],
                pure_rate_of_social_time_preference=welfare_defaults_utilitarian[
                    "pure_rate_of_social_time_preference"
                ],
                inequality_aversion=welfare_defaults_utilitarian["inequality_aversion"],
                sufficiency_threshold=welfare_defaults_utilitarian[
                    "sufficiency_threshold"
                ],
                egality_strictness=welfare_defaults_utilitarian["egality_strictness"],
            )

            welfare_function_prioritarian = SocialWelfareFunction(
                input_dataset=data_loader,
                time_horizon=time_horizon,
                climate_ensembles=no_of_ensembles,
                population=population,
                risk_aversion=welfare_defaults_prioritarian["risk_aversion"],
                elasticity_of_marginal_utility_of_consumption=welfare_defaults_prioritarian[
                    "elasticity_of_marginal_utility_of_consumption"
                ],
                pure_rate_of_social_time_preference=welfare_defaults_prioritarian[
                    "pure_rate_of_social_time_preference"
                ],
                inequality_aversion=welfare_defaults_prioritarian[
                    "inequality_aversion"
                ],
                sufficiency_threshold=welfare_defaults_prioritarian[
                    "sufficiency_threshold"
                ],
                egality_strictness=welfare_defaults_prioritarian["egality_strictness"],
            )

            _, _, _, reference_set_df.loc[index, "welfare_utilitarian"] = (
                welfare_function_utilitarian.calculate_welfare(
                    consumption_per_capita=scenario_datasets[scenario_list[0]][
                        "consumption_per_capita"
                    ]
                )
            )
            (
                _,
                _,
                _,
                reference_set_df.loc[index, "damage_cost_per_capita_utilitarian"],
            ) = welfare_function_utilitarian.calculate_welfare(
                consumption_per_capita=scenario_datasets[scenario_list[0]][
                    "damage_cost_per_capita"
                ],
                welfare_loss=True,
            )

            (
                _,
                _,
                _,
                reference_set_df.loc[index, "abatement_cost_per_capita_utilitarian"],
            ) = welfare_function_utilitarian.calculate_welfare(
                consumption_per_capita=scenario_datasets[scenario_list[0]][
                    "abatement_cost_per_capita"
                ],
                welfare_loss=True,
            )

            _, _, _, reference_set_df.loc[index, "welfare_prioritarian"] = (
                welfare_function_prioritarian.calculate_welfare(
                    consumption_per_capita=scenario_datasets[scenario_list[0]][
                        "consumption_per_capita"
                    ]
                )
            )

            (
                _,
                _,
                _,
                reference_set_df.loc[index, "damage_cost_per_capita_prioritarian"],
            ) = welfare_function_prioritarian.calculate_welfare(
                consumption_per_capita=scenario_datasets[scenario_list[0]][
                    "damage_cost_per_capita"
                ],
                welfare_loss=True,
            )

            (
                _,
                _,
                _,
                reference_set_df.loc[index, "abatement_cost_per_capita_prioritarian"],
            ) = welfare_function_prioritarian.calculate_welfare(
                consumption_per_capita=scenario_datasets[scenario_list[0]][
                    "abatement_cost_per_capita"
                ],
                welfare_loss=True,
            )

        # # Transform the damage cost per capita to welfare loss value
        # _, _, welfare_loss_damage = model.welfare_function.calculate_welfare(
        #     datasets["damage_cost_per_capita"], welfare_loss=True
        # )
        # welfare_loss_damage = np.abs(welfare_loss_damage)
        # # Transform the abatement cost to welfare loss value
        # _, _, welfare_loss_abatement = model.welfare_function.calculate_welfare(
        #     datasets["abatement_cost_per_capita"], welfare_loss=True
        # )
        # welfare_loss_abatement = np.abs(welfare_loss_abatement)

        # Save the reference_set_df as a csv file
        reference_set_df.to_csv(
            path_to_output + output_file_name + "_reevaluated" + ".csv"
        )


if __name__ == "__main__":

    # reevaluate_all_for_utilitarian_prioritarian(
    #     input_data=[
    #         "UTILITARIAN_reference_set.csv",
    #         # "PRIORITARIAN_reference_set.csv",
    #         # "SUFFICIENTARIAN_reference_set.csv",
    #         # "EGALITARIAN_reference_set.csv",
    #     ],
    # )

    #######################################################################
    # reevaluate_optimal_policy(
    #     input_data=[
    #         "UTILITARIAN_reference_set.csv",
    #         # "PRIORITARIAN_reference_set.csv",
    #         # "SUFFICIENTARIAN_reference_set.csv",
    #         # "EGALITARIAN_reference_set.csv",
    #         # "UTILITARIAN_reference_set_reevaluated.csv",
    #         # "PRIORITARIAN_reference_set_reevaluated.csv",
    #     ],
    #     path_to_rbf_weights="data/convergence_metrics/",  #  reevaluation
    #     path_to_output="data/reevaluation/util_90_welfare_temp/",
    #     objective_of_interest=None,  # "years_above_temperature_threshold",  # "welfare", None
    #     direction_of_optimization=[
    #         "min",
    #         "min",
    #         "max",
    #         "max",
    #     ],
    #     lowest_n_percent=0.52,  # 0.51, 0.52 is needed for Prioritarian formulation
    #     rbf_policy_index=66,
    #     list_of_objectives=[
    #         "welfare",
    #         "years_above_temperature_threshold",
    #         "welfare_loss_damage",
    #         "welfare_loss_abatement",
    #     ],
    #     scenario_list=["SSP245"],  # list(Scenario.__members__.keys()),  #
    # )

    ########################################################################
    # scenario_list = list(
    #     Scenario.__members__.keys()
    # )  # ['SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585']
    scenario_list = ["SSP245"]
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
        path_to_data="data/reevaluation/util_90_welfare_temp",  # only_welfare_temp",  # "data/reevaluation/comparison_experiments",
        path_to_output="data/reevaluation/util_90_welfare_temp",  # only_welfare_temp/extracted_variable",  # "data/reevaluation/comparison_experiments",
        variable_name="emissions",  # "net_economic_output",  # "economic_damage",  # "emissions", #abatement_cost, # "global_temperature", gross_economic_output, consumption_per_capita
        data_shape=3,
        no_of_ensembles=1001,
        input_data=[
            "UTILITARIAN_reference_set_idx66.pkl",
            # "UTILITARIAN_reference_set_reevaluated_idx85.pkl",
            # "UTILITARIAN_reference_set_reevaluated_idx73.pkl",
            # "UTILITARIAN_reference_set_reevaluated_idx19.pkl",
            # "PRIORITARIAN_reference_set_reevaluated_idx639.pkl",
            # "PRIORITARIAN_reference_set_reevaluated_idx605.pkl",
            # "PRIORITARIAN_reference_set_reevaluated_idx529.pkl",
            # "SUFFICIENTARIAN_reference_set_idx57.pkl",
            # "EGALITARIAN_reference_set_idx404.pkl",
        ],
        output_file_names=[
            "Utilitarian_66",
            # "Utilitarian_85",
            # "Utilitarian_73",
            # "Utilitarian_19",
            # "Prioritarian_639",
            # "Prioritarian_605",
            # "Prioritarian_529",
            # "Sufficientarian",
            # "Egalitarian",
        ],
    )
