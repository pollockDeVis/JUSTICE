# Check this method out too: #NOTE ROUGH

from justice.util.output_data_processor import process_scenario

import os
import filecmp
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
from justice.util.enumerations import WelfareFunction, SSP
from justice.util.output_data_processor import process_scenario

from justice.util.output_data_processor import (
    reevaluate_optimal_policy,
    reevaluated_optimal_policy_variable_extractor,
)
from justice.util.model_time import TimeHorizon
from justice.util.data_loader import DataLoader


if __name__ == "__main__":
    ########################################
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

    ######################################
    scenario_list = ["SSP126", "SSP245", "SSP370", "SSP460", "SSP534"]

    social_welfare_function = WelfareFunction.PRIORITARIAN
    nfe = 50_000
    ssp = SSP.SSP3

    sw_name = social_welfare_function.value[1]
    path = f"data/temporary/NU_DATA/combined/{str(ssp).split('.')[1]}/"
    filename = f"{sw_name}_reference_set.csv"

    # Print the selected policy indices values of last 4 columns
    # Find the min and max welfare values for utilitarian and prioritarian
    loaded_df = pd.read_csv(path + filename)
    policy_indices = list(range(len(loaded_df)))

    sw_name = social_welfare_function.value[1]

    ###############################################
    # Process Scenario in Parallel Block
    # NOTE: Temporarily commented out as it is not needed for the current run
    # filename = f"{sw_name}_reference_set.csv"

    # loaded_df = pd.read_csv(path + filename)
    # print(f"Loading data for {sw_name} from {path+filename}")
    # print("Selected policy‑indices last 2 columns:")
    # print(loaded_df.iloc[policy_indices, -2:])

    # # spawn‐based pool so that each worker is a fresh interpreter
    # mp.set_start_method("spawn")
    # # Bind the fixed arguments swf and policy_indices to process_scenario
    # bound_process_scenario = partial(
    #     process_scenario, social_welfare_function, path, policy_indices
    # )
    # with mp.Pool(processes=len(scenario_list)) as pool:
    #     pool.map(bound_process_scenario, scenario_list)

    ####################################
    # Reevaluate Optimal Policy Block
    policy_index = 2  # Put the policy index here.

    base_dir = path
    input_data_name = (
        f"{social_welfare_function.value[1]}_reference_set_idx{policy_index}.h5"
    )

    scenario = scenario_list
    # [
    #     4
    # ]  # Iterate manually through the scenarios and reset kernel every scenario

    print(
        f"Processing scenario: {scenario} with policy index: {policy_index} for {social_welfare_function.value[1]}"
    )

    # NOTE: The following code generates large dataframes and saved them in the data/temporary folder. Size is ~ 1.5 GB each run

    reevaluate_optimal_policy(
        input_data=[
            f"{social_welfare_function.value[1]}_reference_set.csv",
        ],
        path_to_rbf_weights=path,  #  reevaluation
        path_to_output=path,  #  reevaluation
        direction_of_optimization=[
            "min",
            "min",
        ],
        rbf_policy_index=policy_index,  # selected_indices[0], # This chooses policy for a particular rival framing. Can also set to the index directly
        list_of_objectives=[
            "welfare",
            "fraction_above_threshold",
        ],
        scenario_list=scenario,  # [scenario], # This is only for a single scenario
    )

    ############################################################################################################

    variable_names_and_shapes = {
        "global_temperature": 2,
        "constrained_emission_control_rate": 3,
        "emissions": 3,
    }
    for variable_name, data_shape in variable_names_and_shapes.items():
        reevaluated_optimal_policy_variable_extractor(
            scenario_list=scenario,  # [scenario], # This is only for a single scenario
            region_list=region_list,
            list_of_years=list_of_years,
            path_to_data=path,
            path_to_output=path,
            variable_name=variable_name,
            data_shape=data_shape,  # 2 for temperature, 3 for rest
            no_of_ensembles=1001,
            input_data=[
                input_data_name,
            ],
            output_file_names=[
                f"{social_welfare_function.value[1]}_{variable_name}",
            ],
        )
