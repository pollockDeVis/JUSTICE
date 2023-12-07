"""
This module calculate the utility based on the utilitarian principle.
Derived from RICE50 model which is based on Berger et al. (2020).
* REFERENCES
* Berger, Loic, and Johannes Emmerling (2020): Welfare as Equity Equivalents, Journal of Economic Surveys 34, no. 4 (26 August 2020): 727-752. https://doi.org/10.1111/joes.12368.
"""
from typing import Any
import numpy as np
import pandas as pd
from src.util.enumerations import get_economic_scenario, WelfareFunction
from config.default_parameters import SocialWelfareDefaults


class Utilitarian:
    """
    This class computes the utilitarian welfare for the JUSTICE model.
    """

    def __init__(self, input_dataset, time_horizon, population, **kwargs):
        """
        This method initializes the Utilitarian class.
        """
        self.region_list = input_dataset.REGION_LIST

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon
        # Instantiate the SocialWelfareDefaults class
        social_welfare_defaults = SocialWelfareDefaults()

        # Fetch the defaults for UTILITARIAN
        utilitarian_defaults = social_welfare_defaults.get_defaults(
            WelfareFunction.UTILITARIAN.name
        )

        # Assign the defaults to the class attributes
        self.elasticity_of_marginal_utility_of_consumption = kwargs.get(
            "elasticity_of_marginal_utility_of_consumption",
            utilitarian_defaults["elasticity_of_marginal_utility_of_consumption"],
        )
        self.pure_rate_of_social_time_preference = kwargs.get(
            "pure_rate_of_social_time_preference",
            utilitarian_defaults["pure_rate_of_social_time_preference"],
        )
        self.inequality_aversion = kwargs.get(
            "inequality_aversion", utilitarian_defaults["inequality_aversion"]
        )

        # Time horizon
        timestep_list = np.arange(
            0, len(time_horizon.model_time_horizon), time_horizon.timestep
        )

        # Calculate the discount rate
        discount_rate = 1 / (
            np.power(
                (1 + self.pure_rate_of_social_time_preference),
                (time_horizon.timestep * (timestep_list)),
            )
        )

        # Regionalize the discount rate
        discount_rate = np.tile(discount_rate, (len(self.region_list), 1))

        # Reshape discount_rate adding np.newaxis Changing shape from (timesteps,) to (timesteps, 1)
        self.discount_rate = discount_rate[:, :, np.newaxis]

        # Calculate the total population for each timestep
        total_population = np.sum(population, axis=0)

        # Calculate the population ratio for each timestep
        self.population_ratio = population / total_population

    def calculate_welfare(self, consumption_per_capita):
        """
        This method calculates the utilitarian welfare.
        """
        # Calculate the consumption per capita raised to the power of 1 - inequality_aversion
        consumption_per_capita_inequality_aversion = np.power(
            consumption_per_capita, 1 - self.inequality_aversion
        )

        # Calculate the population weighted consumption per capita
        population_weighted_consumption_per_capita = (
            self.population_ratio * consumption_per_capita_inequality_aversion
        )

        # Calculate the disentangled utility
        disentangled_utility = population_weighted_consumption_per_capita

        # Calculate the regional disentangled utility powered - For regional welfare calculation
        disentangled_utility_regional_powered = np.power(
            disentangled_utility,
            (
                (1 - self.elasticity_of_marginal_utility_of_consumption)
                / (1 - self.inequality_aversion)
            ),
        )

        # Sum the disentangled utility
        disentangled_utility_summed = np.sum(
            population_weighted_consumption_per_capita, axis=0
        )

        # Calculate the disentangled utility powered
        disentangled_utility_powered = np.power(
            disentangled_utility_summed,
            (
                (1 - self.elasticity_of_marginal_utility_of_consumption)
                / (1 - self.inequality_aversion)
            ),
        )

        # Calculate the utilitarian welfare disaggregated temporally
        welfare_utilitarian_temporal = (
            (
                disentangled_utility_powered
                / (1 - self.elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        ) * self.discount_rate[0, :, :]

        # Welfare disaggregated temporally and regionally - For regional welfare calculation
        welfare_utilitarian_regional_temporal = (
            (
                disentangled_utility_regional_powered
                / (1 - self.elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        ) * self.discount_rate

        # Welfare aggregated regionally
        welfare_utilitarian_regional = np.sum(
            welfare_utilitarian_regional_temporal,
            axis=1,
        )

        # Calculate the utilitarian welfare
        welfare_utilitarian = np.sum(
            welfare_utilitarian_temporal,
            axis=0,
        )

        return (
            disentangled_utility,
            welfare_utilitarian_temporal,
            welfare_utilitarian_regional,
            welfare_utilitarian,
        )

    def calculate_stepwise_welfare(self, consumption_per_capita, timestep):
        """
        This method calculates the utilitarian welfare.
        """
        # Calculate the consumption per capita raised to the power of 1 - inequality_aversion
        consumption_per_capita_inequality_aversion = np.power(
            consumption_per_capita, 1 - self.inequality_aversion
        )

        # Calculate the population weighted consumption per capita
        population_weighted_consumption_per_capita = (
            self.population_ratio[:, timestep, :]
            * consumption_per_capita_inequality_aversion
        )

        # Calculate the disentangled utility
        disentangled_utility = population_weighted_consumption_per_capita

        # Sum the disentangled utility
        disentangled_utility_summed = np.sum(
            population_weighted_consumption_per_capita, axis=0
        )

        # Calculate the disentangled utility powered
        disentangled_utility_powered = np.power(
            disentangled_utility_summed,
            (
                (1 - self.elasticity_of_marginal_utility_of_consumption)
                / (1 - self.inequality_aversion)
            ),
        )

        # Calculate disentangled utility regional powered - For regional welfare calculation
        disentangled_utility_regional_powered = np.power(
            disentangled_utility,
            (
                (1 - self.elasticity_of_marginal_utility_of_consumption)
                / (1 - self.inequality_aversion)
            ),
        )

        # Welfare disaggregated temporally and regionally - For regional welfare calculation
        welfare_utilitarian_regional_temporal = (
            (
                disentangled_utility_regional_powered
                / (1 - self.elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        ) * self.discount_rate[:, timestep, :]

        # Welfare aggregated regionally, disaggregated temporally
        welfare_utilitarian_temporal = (
            (
                disentangled_utility_powered
                / (1 - self.elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        ) * self.discount_rate[0, timestep, :]

        return (
            disentangled_utility,
            welfare_utilitarian_regional_temporal,
            welfare_utilitarian_temporal,
        )

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
