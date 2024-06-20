
import numpy as np
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
from policy_loader import PolicyLoader

def JUSTICE_stepwise_run(
    path_to_output="data/output/",
    saving=True,
    output_file_name='bau_run',
):
    """
    Run the JUSTICE model for all the scenarios
    """
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

        # Initialize datasets to store the results
        datasets = {}
        # Get the emission control rate for the current scenario
        emission_control_rate_for_scenario = constrained_emission_control_rate[scenarios]

        for timestep in range(len(time_horizon.model_time_horizon)):
            # Extract the emission control rate for the current timestep
            current_emission_control_rate = emission_control_rate_for_scenario[:, timestep, :]

            model.stepwise_run(
                emission_control_rate=current_emission_control_rate,
                timestep=timestep,
                endogenous_savings_rate=True,
                recycling_rate=recycling_rate
            )

            #Evaluate the model
            datasets = model.stepwise_evaluate(timestep=timestep)

        print(f"Completed scenario {idx}: {scenarios}")

    # Save the datasets
    if saving:
        if output_file_name is not None:
            np.save(path_to_output + output_file_name, datasets)

    return datasets

if __name__ == "__main__":
    datasets = JUSTICE_stepwise_run()
    # Print the keys of the datasets
    print(datasets.keys())
