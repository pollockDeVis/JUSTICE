"""

#############################
## PAPER EVALUATION SCRIPT ##
#############################

This script can be used to reproduce the evaluation results of the paper.

The script will save multiple metrics:

- emissions: The emissions produced by each agent at each timestep, for each seed
- abated_emissions: The emissions abated by each agent at each timestep, for each seed
- net_economic_output: The net economic output produced by each agent at each timestep, for each seed
- global_temperature: The global temperature at each timestep, for each seed
- regional_temperature: The regional temperature for each agent at each timestep, for each seed
- savings_rates: The savings rates for each agent at each timestep, for each seed
- emission_control_rates: The emission control rates for each agent at each timestep, for each seed

Moreover, the following plots will be generated and saved:

TIME SERIES PLOTS:
- Economic output over time
- Total emissions over time
- Average global temperature rise over time
- Total abated emissions over time

REGIONAL MAPS:
- Abated emissions over time

GINI COEFFICIENT PLOTS:
- Gini coefficient over time for economic output
- Gini coefficient over time for emissions
- Gini coefficient over time for abated emissions


"""

import os, glob, pickle, pandas as pd

from rl.args import Args
from paper.eval_utils import *
from eval import *

if __name__ == "__main__":

    CURRENT_WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
    PLOTS_FOLDER = os.path.join(
        CURRENT_WORKING_DIR,
    )  # Folder containing the plots
    POLICY_FOLDER = os.path.join(
        CURRENT_WORKING_DIR, "paper_policies"
    )  # Folder containing the checkpoints and pickles
    ALL_POLICIES = glob.glob(os.path.join(POLICY_FOLDER, "*"))

    baselines_emissions_data_path = os.path.join(
        CURRENT_WORKING_DIR, "baseline_emissions.csv"
    )  # Used to calculate abated emissions

    # Load the baseline emissions data
    base_emissions = pd.read_csv(baselines_emissions_data_path, delimiter=";")
    base_emissions = base_emissions.iloc[:, 1:]
    base_emissions = base_emissions.to_numpy()

    evaluation_seeds = range(1, 11)  # Seeds 1 to 10
    global_data_dict = {}

    for policy_base_path in ALL_POLICIES:

        policy_folder_name = policy_base_path.split("/")[-1]

        print(f"Running evaluations for policy {policy_folder_name}")

        checkpoint_path = glob.glob(os.path.join(policy_base_path, "checkpoints", "*"))[
            0
        ]

        loaded_args = Args()

        loaded_args.network_model_config = glob.glob("rl/nn/params/mlp.json")[0]

        for eval_seed in evaluation_seeds:

            print(f"Running evaluation for seed={str(eval_seed)}")

            # Evaluate the policy
            seed_evaluation_data = evaluate_seed(
                base_emissions=base_emissions,
                loaded_args=loaded_args,
                checkpoint_path=checkpoint_path,
                eval_seed=eval_seed,
            )

            seed_evaluation_dict = {str(eval_seed): seed_evaluation_data}

            if policy_folder_name not in global_data_dict.keys():
                global_data_dict[policy_folder_name] = seed_evaluation_dict = {
                    str(eval_seed): seed_evaluation_data
                }
            else:
                global_data_dict[policy_folder_name].update(seed_evaluation_dict)

    with open(
        os.path.join(
            CURRENT_WORKING_DIR, "eval_results", "evaluation_raw_data.pkl"
        ),
        "wb",
    ) as f:

        pickle.dump(global_data_dict, f)

    evaluation_data = global_data_dict

    aggregated_data = aggregate_data_across_seeds(evaluation_data=evaluation_data)

    create_csv_files(aggregated_data=aggregated_data, current_working_dir=PLOTS_FOLDER)

    (
        rice_abated_emissions_data,
        rice_emissions_data,
        rice_global_temperature_data,
        rice_net_economic_output_data,
    ) = load_rice_data(current_working_dir=PLOTS_FOLDER)

    plot_economic_output_over_time(
        aggregated_data=aggregated_data,
        rice_data=rice_net_economic_output_data,
        current_working_dir=PLOTS_FOLDER,
    )
    plot_total_emissions_over_time(
        aggregated_data=aggregated_data,
        rice_data=rice_emissions_data,
        current_working_dir=PLOTS_FOLDER,
    )
    plot_average_global_temperature_over_time(
        aggregated_data=aggregated_data,
        rice_data=rice_global_temperature_data,
        current_working_dir=PLOTS_FOLDER,
    )
    plot_total_abated_emissions_over_time(
        aggregated_data=aggregated_data,
        rice_data=rice_abated_emissions_data,
        current_working_dir=PLOTS_FOLDER,
    )

    # Plotting regional maps for abated emissions

    abated_emissions_data = {
        policy_name: policy_data["mean"]["abated_emissions"]
        for policy_name, policy_data in aggregated_data.items()
    }

    abated_emissions_data["rice"] = rice_abated_emissions_data

    plot_regional_maps(abated_emissions_data, current_working_dir=PLOTS_FOLDER)

    # Plotting Gini coefficients

    plot_gini_over_time(
        aggregated_data=aggregated_data,
        rice_data={
            "net_economic_output": rice_net_economic_output_data,
            "emissions": rice_emissions_data,
            "abated_emissions": rice_abated_emissions_data,
        },
        current_working_dir=PLOTS_FOLDER,
        moving_average_n=5,
    )
