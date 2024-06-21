
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
    policy_loader = PolicyLoader()
    
    #Select the policy 
    recycling_rate = policy_loader.RECYCLING_RATE_LINEAR_PROYECTION
    constrained_emission_control_rate = policy_loader.constrained_emission_control_rate

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(start_year=2015, end_year=2300, data_timestep=5, timestep=1)

    for idx, scenarios in enumerate(list(Scenario.__members__.keys())):
        print(f"Starting scenario {idx}: {scenarios}")
        
        #Initialize the model
        model = JUSTICE(
            scenario=idx,
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,
            matter=EconomySubModules.MATTER,
        )


        # Get the emission control rate for the current scenario
        emission_control_rate_for_scenario = constrained_emission_control_rate[scenarios]

        scenario_results = {}

        for timestep in range(len(time_horizon.model_time_horizon)):
            # Extract the emission control rate for the current timestep
            current_emission_control_rate = emission_control_rate_for_scenario[:, timestep, :]

            model.stepwise_run(
                emission_control_rate=current_emission_control_rate,
                timestep=timestep,
                endogenous_savings_rate=True,
                recycling_rate=recycling_rate
            )

            # Evaluate the model
            datasets = model.stepwise_evaluate(timestep=timestep)

        scenario_results = datasets

        # Save the scenario results to a separate file
        if saving:
            scenario_output_path = os.path.join(path_to_output, f"{scenarios}.npz")
            np.savez_compressed(scenario_output_path, **scenario_results)
            print(f"Saved scenario {scenarios} to {scenario_output_path}")

        print(f"Completed scenario {idx}: {scenarios}")
    
    return scenario_results

if __name__ == "__main__":
    datasets = JUSTICE_stepwise_run(path_to_output="data/output/bau",saving=True)
    # Print the keys of the datasets
    print(datasets.keys())
