"""
This module calculate the utility based on the utilitarian principle.
Derived from RICE50 model which is based on Berger et al. (2020).
* REFERENCES
* Berger, Loic, and Johannes Emmerling (2020): Welfare as Equity Equivalents, Journal of Economic Surveys 34, no. 4 (26 August 2020): 727-752. https://doi.org/10.1111/joes.12368.
"""
import numpy as np
import pandas as pd
from src.enumerations import get_economic_scenario


def calculate_utilitarian_welfare(
    time_horizon,
    region_list,
    scenario,
    population,
    consumption_per_capita,
    elasticity_of_marginal_utility_of_consumption,
    pure_rate_of_social_time_preference,
    inequality_aversion,
):
    scenario = get_economic_scenario(scenario)

    timestep_list = np.arange(
        0, len(time_horizon.model_time_horizon), time_horizon.timestep
    )

    # Calculate the discount rate
    discount_rate = 1 / (
        np.power(
            (1 + pure_rate_of_social_time_preference),
            (time_horizon.timestep * (timestep_list)),
        )
    )
    discount_rate = np.tile(discount_rate, (len(region_list), 1))
    # Reshape discount_rate adding np.newaxis Changing shape from (timesteps,) to (timesteps, 1)
    discount_rate = discount_rate[:, :, np.newaxis]

    # Calculate the total population for each timestep
    total_population = np.sum(population, axis=0)

    # Calculate the population ratio for each timestep
    population_ratio = population / total_population

    # Fetch Consumption per Capita
    # consumption_per_capita = economy.get_consumption_per_capita(scenario, savings_rate)

    # Calculate the consumption per capita raised to the power of 1 - inequality_aversion
    consumption_per_capita_inequality_aversion = np.power(
        consumption_per_capita, 1 - inequality_aversion
    )

    # Calculate the population weighted consumption per capita
    population_weighted_consumption_per_capita = (
        population_ratio * consumption_per_capita_inequality_aversion
    )

    disentangled_utility = population_weighted_consumption_per_capita

    disentangled_utility_summed = np.sum(
        population_weighted_consumption_per_capita, axis=0
    )

    disentangled_utility_powered = np.power(
        disentangled_utility_summed,
        (
            (1 - elasticity_of_marginal_utility_of_consumption)
            / (1 - inequality_aversion)
        ),
    )

    welfare_utilitarian = np.sum(
        (
            (
                disentangled_utility_powered
                / (1 - elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        )
        * discount_rate,
        axis=0,
    )

    return disentangled_utility, welfare_utilitarian
