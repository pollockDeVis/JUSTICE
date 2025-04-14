
import numpy as np
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
from policy_loader import PolicyLoader
import os

def JUSTICE_stepwise_run(
    path_to_output=None,
    saving=False,
):
    """
    Run the JUSTICE model for all the scenarios
    """
    # Ensure the output directory exists
    os.makedirs(path_to_output, exist_ok=True)

    # Load the data
    policy_loader = PolicyLoader(emission_control_rate_file_path="data/input/emissions_control_rate/Utilitarian_SSP245_constrained_emission_control_rate.npy")
    
    #Select the policy 
    recycling_rate = policy_loader.RECYCLING_RATE_2050_TARGET
    constrained_emission_control_rate = policy_loader.constrained_emission_control_rate

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(start_year=2015, end_year=2300, data_timestep=5, timestep=1)
    results_by_scenario = {}

    for scen_label, emission_control_rate_for_scenario in constrained_emission_control_rate.items():
        idx = Scenario[scen_label].value[0]
        print(f"Starting scenario {idx}: {scen_label}")
        climate_ensembles = None
        climate_ensembles_idx = slice(None)
        #climate_ensembles = climate_ensembles_idx = 570
        model = JUSTICE(
            scenario=idx,
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,
            matter=EconomySubModules.MATTER,
            climate_ensembles=climate_ensembles
        )

        for timestep in range(len(time_horizon.model_time_horizon)):
            # Extract the emission control rate for the current timestep
            current_emission_control_rate = emission_control_rate_for_scenario[:, timestep, climate_ensembles_idx]

            model.stepwise_run(
                emission_control_rate=current_emission_control_rate,
                timestep=timestep,
                endogenous_savings_rate=True,
                recycling_rate=recycling_rate
            )

            # Evaluate the model
            scenario_results = model.stepwise_evaluate(timestep=timestep)

        # Save the scenario results to a separate file
        if saving:
            scenario_output_path = os.path.join(path_to_output, f"{scen_label}.npz")
            np.savez_compressed(scenario_output_path, **scenario_results)
            print(f"Saved scenario {scen_label} to {scenario_output_path}")

        print(f"Completed scenario {idx}: {scen_label}")
        results_by_scenario[scen_label] = scenario_results
    
    return results_by_scenario

if __name__ == "__main__":
    datasets = JUSTICE_stepwise_run(path_to_output="data/output/ce_newimput",saving=False)
    # Print the keys of the datasets
    print(datasets.keys())
