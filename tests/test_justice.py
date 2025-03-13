import numpy as np
import pytest
from src.climate.coupled_fair import CoupledFAIR
from src.util.model_time import TimeHorizon
from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.model import JUSTICE


def test_justice_fair_coupling():

    data_loader = DataLoader()

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(
        start_year=2015, end_year=2300, data_timestep=5, timestep=1
    )

    # emissions control rate borrowed from emissions module

    # Variables to be changed/deleted later
    miu_initial = 0.0
    min_miu = 0.0  # 0.2  # 0.0 #1.0
    min_miu_year = 2060  # 9-original #8 in this model  # 2060
    max_miu = 1.0  # 1.0  # 1.2
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

    # Instantiate the model
    scenarios = 2
    model = JUSTICE(
        scenario=scenarios,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=WelfareFunction.UTILITARIAN,
    )

    model.run(
        emission_control_rate=emissions_control_rate, endogenous_savings_rate=True
    )
    datasets = model.evaluate()  # Get the results of the simulation run
    emissions = datasets["emissions"]
    # Sum up the emissions for all regions
    global_emissions = np.sum(emissions, axis=0)
    # Get the global temperature
    global_temperature = datasets["global_temperature"]

    net_economy = datasets["net_economic_output"]
    global_net_economic_output = np.sum(net_economy, axis=0)

    emission_verification_data = np.load(
        "tests/verification_data/emissions_justice_linear_ecr.npy"
    )
    global_temp_verification_data = np.load(
        "tests/verification_data/global_temperature_justice_linear_ecr.npy"
    )

    global_net_economic_output_verification_data = np.load(
        "tests/verification_data/global_net_output_justice_linear_ecr.npy"
    )

    np.testing.assert_allclose(
        global_emissions, emission_verification_data, rtol=1e-5, atol=1e-5
    )

    np.testing.assert_allclose(
        global_temperature, global_temp_verification_data, rtol=1e-5, atol=1e-5
    )

    np.testing.assert_allclose(
        global_net_economic_output,
        global_net_economic_output_verification_data,
        rtol=1e-5,
        atol=1e-5,
    )
