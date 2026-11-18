import pandas as pd
from justice.util.output_data_processor import (
    reevaluate_optimal_policy,
    reevaluated_optimal_policy_variable_extractor,
)
from justice.util.model_time import TimeHorizon
from justice.util.data_loader import DataLoader
from justice.util.enumerations import WelfareFunction, SSP
import json
import os


def reevaluate_policies_for_all_scenarios(
    base_path="data/temporary/NU_DATA/mmBorg/",
    scenario_list=["SSP126", "SSP245", "SSP370", "SSP460", "SSP534"],
    policy_indices_dict_name="min_regret_policy_indices.json",
    output_path=None,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
):

    # Throw error if output_path is None
    if output_path is None:
        raise ValueError(
            "output_path cannot be None. Please provide a valid output path."
        )

    start_year = 2015
    end_year = 2300
    data_timestep = 5
    timestep = 1

    data_loader = DataLoader()
    region_list = data_loader.REGION_LIST

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )

    list_of_years = time_horizon.model_time_horizon

    with open(base_path + policy_indices_dict_name, "r") as f:
        loaded_min_regret_policy_indices = json.load(f)

    for scenario, ethical_data in loaded_min_regret_policy_indices.items():

        for ethical_framing, regret_data in ethical_data.items():

            for regret_type, policy_index in regret_data.items():

                # ssp = SSP[scenario]
                social_welfare_function = WelfareFunction[ethical_framing]
                path = f"{base_path}/{ethical_framing}_{scenario}/"
                # Print
                print(
                    f"Reevaluating for: {ethical_framing}, ref scenario: {scenario}, Regret {regret_type} Policy Index: {policy_index}"
                )
                # TODO: Temporarily commenting out reevaluation to just extract variables
                reevaluate_optimal_policy(
                    input_data=[
                        f"{social_welfare_function.value[1]}_reference_set.csv",
                    ],
                    path_to_rbf_weights=path,  #  reevaluation
                    path_to_output=output_path,  #  reevaluation
                    direction_of_optimization=[
                        "min",
                        "min",
                    ],
                    rbf_policy_index=policy_index,  # selected_indices[0], # This chooses policy for a particular rival framing. Can also set to the index directly
                    list_of_objectives=[
                        "welfare",
                        "fraction_above_threshold",
                    ],
                    scenario_list=scenario_list,  # [scenario], # This is only for a single scenario
                    reference_scenario=scenario,
                    regret_type=regret_type,
                )

                # ############################################################################################################
                variable_names_and_shapes = {
                    "global_temperature": 2,
                    "constrained_emission_control_rate": 3,
                    "emissions": 3,
                }
                input_data_name = f"{ethical_framing}_reference_set_ref_{scenario}_{regret_type}_idx{policy_index}.h5"
                for variable_name, data_shape in variable_names_and_shapes.items():
                    reevaluated_optimal_policy_variable_extractor(
                        scenario_list=scenario_list,  # [scenario], # This is only for a single scenario
                        region_list=region_list,
                        list_of_years=list_of_years,
                        path_to_data=base_path,
                        path_to_output=base_path + f"/{ethical_framing}_{scenario}"
                        f"/ref_{scenario}_{regret_type}_idx{policy_index}",  # TODO Change this later
                        variable_name=variable_name,
                        data_shape=data_shape,  # 2 for temperature, 3 for rest
                        no_of_ensembles=1001,
                        input_data=[
                            input_data_name,
                        ],
                        output_file_names=[
                            f"{ethical_framing}_ref_{scenario}_{regret_type}_idx{policy_index}_{variable_name}",
                        ],
                    )

                # Delete the hdf5 file
                os.remove(base_path + "/" + input_data_name)
                print(f"Deleted HDF5 file at location {input_data_name}")


if __name__ == "__main__":

    reevaluate_policies_for_all_scenarios(output_path="data/temporary/NU_DATA/mmBorg/")
