"""
This module wraps the model for uncertainty analysis experiments using EMA.
"""

from src.model import JUSTICE
from src.enumerations import *


def model_wrapper(
    scenario,
    savings_rate,
    emissions_control_rate,
    elasticity_of_marginal_utility_of_consumption,
    pure_rate_of_social_time_preference,
    inequality_aversion,
    economy_type=Economy.NEOCLASSICAL,
    damage_function_type=DamageFunction.KALKUHL,
    abatement_type=Abatement.ENERDATA,
    welfare_function=WelfareFunction.UTILITARIAN,
):
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
