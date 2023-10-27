"""
This module wraps the model for uncertainty analysis experiments using EMA.
"""
import numpy as np

from src.model import JUSTICE
from src.enumerations import *


def model_wrapper(**kwargs):

    scenario = kwargs.pop("ssp_rcp_scenario")
    elasticity_of_marginal_utility_of_consumption = kwargs.pop("elasticity_of_marginal_utility_of_consumption")
    pure_rate_of_social_time_preference = kwargs.pop("pure_rate_of_social_time_preference")
    inequality_aversion = kwargs.pop("inequality_aversion")
    economy_type = Economy.NEOCLASSICAL,
    damage_function_type = DamageFunction.KALKUHL,
    abatement_type = Abatement.ENERDATA,
    welfare_function = WelfareFunction.UTILITARIAN,

    n_regions = kwargs.pop("n_regions")
    n_timesteps = kwargs.pop("n_timesteps")

    savings_rate = np.zeros((n_regions, n_timesteps))
    emissions_control_rate = np.zeros((n_regions, n_timesteps))
    for i in range(n_regions):
        for j in range(n_timesteps):
            savings_rate[i, j] = kwargs.pop(f"savings_rate {i} {j}")
            emissions_control_rate[i, j] = kwargs.pop(f"emissions_control_rate {i} {j}")

    model = JUSTICE(
        start_year=2015,
        end_year=2300,
        timestep=1,
        scenario=scenario,
        economy_type=economy_type,
        damage_function_type=damage_function_type,
        abatement_type=abatement_type,
    )

    model.run(savings_rate=savings_rate, emissions_control_rate=emissions_control_rate)
    datasets = model.evaluate(
        welfare_function=welfare_function,
        elasticity_of_marginal_utility_of_consumption=elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference=pure_rate_of_social_time_preference,
        inequality_aversion=inequality_aversion,
    )
    net_economic_output = datasets["net_economic_output"]
    consumption = datasets["consumption"]
    consumption_per_capita = datasets["consumption_per_capita"]
    emissions = datasets["emissions"]
    global_temperature = datasets["global_temperature"]
    economic_damage = datasets["economic_damage"]
    abatement_cost = datasets["abatement_cost"]
    disentangled_utility = datasets["disentangled_utility"]

    return (
        net_economic_output,
        consumption,
        consumption_per_capita,
        emissions,
        global_temperature,
        economic_damage,
        abatement_cost,
        disentangled_utility,
    )
