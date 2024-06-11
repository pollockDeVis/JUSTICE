"""
This module calculate the welfare based on the different Disributive Justice Principles.
Derived from RICE50 model which is based on Berger et al. (2020).
* REFERENCES
* Berger, Loic, and Johannes Emmerling (2020): Welfare as Equity Equivalents, Journal of Economic Surveys 34, no. 4 (26 August 2020): 727-752. https://doi.org/10.1111/joes.12368.
"""

from typing import Any
import numpy as np
import pandas as pd
from src.util.enumerations import get_economic_scenario, WelfareFunction
from src.objectives.objective_functions import calculate_gini_index


# TODO: Need to change the name of this class to a more general name
class SocialWelfareFunction:
    """
    This class computes the welfare for the JUSTICE model.
    """

    def __init__(
        self,
        input_dataset,
        time_horizon,
        population,
        elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference,
        inequality_aversion,
        sufficiency_threshold,
        egality_strictness,
    ):
        """
        This method initializes the Social Welfare Function class.
        """
        self.region_list = input_dataset.REGION_LIST

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon
        self.elasticity_of_marginal_utility_of_consumption = (
            elasticity_of_marginal_utility_of_consumption
        )
        self.pure_rate_of_social_time_preference = pure_rate_of_social_time_preference
        self.inequality_aversion = inequality_aversion
        self.sufficiency_threshold = sufficiency_threshold
        self.egality_strictness = egality_strictness

        # Time horizon
        timestep_list = np.arange(
            0, len(time_horizon.model_time_horizon), time_horizon.timestep
        )

        # Calculate the discount rate # Validated
        discount_rate = 1 / (
            np.power(
                (1 + self.pure_rate_of_social_time_preference),
                (time_horizon.timestep * (timestep_list)),
            )
        )

        # Regionalize the discount rate
        discount_rate = np.tile(discount_rate, (len(self.region_list), 1))  # Validated

        # Reshape discount_rate adding np.newaxis Changing shape from (timesteps,) to (timesteps, 1) # Validated Shape (57, 286, 1)
        self.discount_rate = discount_rate[:, :, np.newaxis]

        # Calculate the total population for each timestep # Validated
        total_population = np.sum(population, axis=0)

        # Calculate the population ratio for each timestep # Validated
        self.population_ratio = population / total_population

    def calculate_welfare(self, consumption_per_capita):
        """
        This method calculates the welfare.
        """

        # Adjust consumption_per_capita with sufficiency threshold
        # New feature - sufficiency_threshold - subtracted from consumption_per_capita
        consumption_per_capita = consumption_per_capita - self.sufficiency_threshold

        # New feature: consumption_per_capita is checked to have negative values
        # If there are negative values, they are replaced with 1e-6. -inf if it becomes 0
        # This is essential to calculate utility that's not a NaN (or complex number)
        consumption_per_capita = np.where(
            consumption_per_capita < 0, 1e-6, consumption_per_capita
        )

        # Calculate the consumption per capita raised to the power of 1 - inequality_aversion
        consumption_per_capita_inequality_aversion = np.power(  # Validated
            consumption_per_capita, 1 - self.inequality_aversion
        )

        # Calculate the population weighted consumption per capita
        population_weighted_consumption_per_capita = (
            self.population_ratio * consumption_per_capita_inequality_aversion
        )

        # Calculate the disentangled utility # Validated
        disentangled_utility = (
            population_weighted_consumption_per_capita  # # has nans 25th region
        )

        # Get the gini of disentalgled utility
        gini_disentangled_utility = calculate_gini_index(disentangled_utility)

        # Calculate the regional disentangled utility powered - For regional welfare calculation # has nans 25th region
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

        # Applying gini to disentangled utility summed
        # egalitarian measure should incorporate a measure of equality, multiplied or added to a measure of individual welfare
        disentangled_utility_summed = disentangled_utility_summed * (
            1 - gini_disentangled_utility * self.egality_strictness
        )

        # Calculate the disentangled utility powered # TODO- Change this
        disentangled_utility_powered = np.power(
            disentangled_utility_summed,
            (
                (1 - self.elasticity_of_marginal_utility_of_consumption)
                / (1 - self.inequality_aversion)
            ),
        )

        # Calculate the welfare disaggregated temporally # TODO- Change this
        welfare_temporal = (
            (
                (disentangled_utility_powered)
                / (1 - self.elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        ) * self.discount_rate[0, :, :]

        # Welfare disaggregated temporally and regionally - For regional welfare calculation
        welfare_regional_temporal = (
            (
                disentangled_utility_regional_powered
                / (1 - self.elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        ) * self.discount_rate

        # Welfare aggregated regionally
        welfare_regional = np.sum(
            welfare_regional_temporal,
            axis=1,
        )

        # Calculate the welfare
        welfare = np.sum(
            welfare_temporal,
            axis=0,
        )

        return (
            disentangled_utility,
            welfare_regional_temporal,
            welfare_temporal,
            welfare_regional,
            welfare,
        )

    def calculate_stepwise_welfare(self, consumption_per_capita, timestep):
        """
        This method calculates the welfare.
        """
        # New feature: consumption_per_capita is checked to have negative values
        consumption_per_capita = np.where(
            consumption_per_capita < 0, 1e-6, consumption_per_capita
        )
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
        welfare_regional_temporal = (
            (
                disentangled_utility_regional_powered
                / (1 - self.elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        ) * self.discount_rate[:, timestep, :]

        # Welfare aggregated regionally, disaggregated temporally
        welfare_temporal = (
            (
                disentangled_utility_powered
                / (1 - self.elasticity_of_marginal_utility_of_consumption)
            )
            - 1
        ) * self.discount_rate[0, timestep, :]

        return (
            disentangled_utility,
            welfare_regional_temporal,
            welfare_temporal,
        )

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
