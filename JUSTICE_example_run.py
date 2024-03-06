import pandas as pd
import numpy as np

from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE


def get_linear_emission_control():
    """
    Linear emission control problem
    """
    data_loader = DataLoader()

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(
        start_year=2015, end_year=2300, data_timestep=5, timestep=1
    )

    # emissions control rate borrowed from emissions module

    # Variables to be changed/deleted later
    miu_initial = 0.0
    min_miu = 0.0  # 0.2 # 0.0 #1.0
    min_miu_year = 2055  # 9-original #8 in this model  # 2060
    max_miu = 0.0  # 1.0  # 1.2
    max_miu_year = 2200  # 38-original #37 in this model #2205

    t_min_miu = time_horizon.year_to_timestep(min_miu_year, timestep=1)
    t_max_miu = time_horizon.year_to_timestep(max_miu_year, timestep=1)

    # Initialize emissions control rate
    emissions_control_rate = np.zeros(
        (len(data_loader.REGION_LIST), len(time_horizon.model_time_horizon))
    )

    for t in range(len(time_horizon.model_time_horizon)):
        if t < t_min_miu:  # Before time of transition
            emissions_control_rate[:, t] = min_miu
        elif t <= t_max_miu:  # Transition
            # During the transition
            emissions_control_rate[:, t] = min_miu + (max_miu - min_miu) * (
                t - t_min_miu
            ) / (t_max_miu - t_min_miu)
        else:  # After the transition
            emissions_control_rate[:, t] = max_miu
    return emissions_control_rate


def JUSTICE_example_run(scenarios=7):
    """
    Run the JUSTICE model for a given scenario
    """
    model = JUSTICE(
        scenario=scenarios,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=WelfareFunction.UTILITARIAN,
        # climate_ensembles=570,
        # Declaring for endogenous fixed savings rate
        elasticity_of_marginal_utility_of_consumption=1.45,
        pure_rate_of_social_time_preference=0.015,
    )

    # Get example emissions control rate
    emissions_control_rate = get_linear_emission_control()

    # Run the model
    model.run(
        emission_control_rate=emissions_control_rate, endogenous_savings_rate=True
    )

    # Get the results
    datasets = model.evaluate()

    return datasets


if __name__ == "__main__":
    datasets = JUSTICE_example_run(scenarios=2)

    # Print the keys of the datasets
    print(datasets.keys())
