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
from src.objectives.objective_functions import (
    calculate_gini_index_c1_2D,
    calculate_gini_index_c1_3D,
)


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
        risk_aversion,
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

        self.risk_aversion = risk_aversion
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
        # TODO: Probably Redundant after dimensions are aggregated in the order described in the paper by Berger & Emmerling
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
        consumption_per_capita = (
            consumption_per_capita - self.sufficiency_threshold
        )  # TODO: This is wrong - threshold should be transformed with utility function

        # New feature: consumption_per_capita is checked to have negative values
        # If there are negative values, they are replaced with 1e-6. -inf if it becomes 0
        # This is essential to calculate utility that's not a NaN (or complex number)
        # TODO: Probably will have to check this step after the utility function is applied
        consumption_per_capita = np.where(
            consumption_per_capita < 0, 1e-6, consumption_per_capita
        )

        (
            disentangled_utility,
            disentangled_utility_regional_powered,
            disentangled_utility_powered,
        ) = self.spatial_aggregator(
            consumption_per_capita,
            self.population_ratio,
            self.elasticity_of_marginal_utility_of_consumption,
            self.inequality_aversion,
            self.egality_strictness,
            self.sufficiency_threshold,
        )

        welfare_regional_temporal, welfare_temporal, welfare_regional, welfare = (
            self.temporal_aggregator(
                disentangled_utility_powered,
                disentangled_utility_regional_powered,
                self.discount_rate,
            )
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

        (
            disentangled_utility,
            disentangled_utility_regional_powered,
            disentangled_utility_powered,
        ) = self.spatial_aggregator(
            consumption_per_capita,
            self.population_ratio[:, timestep, :],
            self.elasticity_of_marginal_utility_of_consumption,
            self.inequality_aversion,
            self.egality_strictness,
            self.sufficiency_threshold,
        )

        # No Temporal aggregation in Stepwise calculation

        return disentangled_utility

    @classmethod
    def utility_function(self, data, parameter):
        """
        This method calculates the isoelastic utility.
        """
        if parameter == 1:
            utility = np.log(data)
        else:
            utility = np.power(data, 1 - parameter) / (1 - parameter)
        return utility

    @classmethod
    def inverse_utility_function(self, data, parameter):
        """
        This method calculates the inverse utility.
        """
        if parameter == 1:
            utility = np.exp(data)
        else:
            utility = np.power((data * (1 - parameter)), 1 / (1 - parameter))
        return utility

    # TODO: Remove this method
    @classmethod
    def spatial_aggregator(
        self,
        data,
        population_ratio,
        elasticity_of_marginal_utility_of_consumption,
        inequality_aversion,
        egality_strictness,
        sufficiency_threshold,
    ):
        """
        This method calculates the spatial aggregator.
        According to Berger & Emmerling (2017), the social welfare function across a dimension is equal to
        the equity equivalent of consumption at the particular dimension
        The aggregated welfare can be calculated in the following steps:
        1. Transform consumption to utility
        2. Weigh the utility with respective weights & sum across the selected dimension
        3. Invert the utility to consumption for next dimension
        """
        # Calculate the consumption per capita raised to the power of 1 - inequality_aversion
        inequality_aversion_transformed_utility = self.utility_function(
            data, inequality_aversion
        )

        # Calculate the population weighted consumption per capita
        population_weighted_utility = (
            population_ratio * inequality_aversion_transformed_utility
        )

        # Calculate the regional disentangled utility powered - This should be used for gini computation (57, 286, 1001)
        declining_marginal_utility_transformed_utility = self.utility_function(
            data, elasticity_of_marginal_utility_of_consumption
        )

        # Aggregate Spatially
        disentangled_utility_summed = np.sum(population_weighted_utility, axis=0)

        # Applying gini to disentangled utility summed
        # egalitarian measure should incorporate a measure of equality, multiplied or added to a measure of individual welfare
        disentangled_utility_summed = disentangled_utility_summed
        # NOTE: Commented out temporarily
        # * (
        #     1 - gini_disentangled_utility * self.egality_strictness
        # )

        # Calculate the disentangled utility powered

        # Invert the utility to consumption
        inequality_aversion_inverted_utility = self.inverse_utility_function(
            disentangled_utility_summed, inequality_aversion
        )
        declining_marginal_utility_transformed_spatially_aggregated_welfare = (
            self.utility_function(
                inequality_aversion_inverted_utility,
                elasticity_of_marginal_utility_of_consumption,
            )
        )

        # returning disaggregated & aggregated version
        return (
            population_weighted_utility,
            declining_marginal_utility_transformed_utility,
            declining_marginal_utility_transformed_spatially_aggregated_welfare,
        )

    @classmethod
    def temporal_aggregator(
        self,
        data,
        data_disaggregated,
        discount_rate,
    ):
        """
        This method calculates the temporal aggregator.
        """
        # Calculate the welfare disaggregated temporally # TODO- Change this
        # TODO: Temporary
        # welfare_temporal = (
        #     ((data) / (1 - elasticity_of_marginal_utility_of_consumption)) - 1
        # ) * discount_rate[0, :, :]

        welfare_temporal = (((data)) - 1) * discount_rate[0, :, :]

        # TODO: Temporary
        # Welfare disaggregated temporally and regionally - For regional welfare calculation
        # welfare_regional_temporal = (
        #     (data_disaggregated / (1 - elasticity_of_marginal_utility_of_consumption))
        #     - 1
        # ) * discount_rate

        welfare_regional_temporal = ((data_disaggregated) - 1) * discount_rate

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
            welfare_regional_temporal,
            welfare_temporal,
            welfare_regional,
            welfare,
        )

    @classmethod
    def states_aggregator(self, data, risk_aversion):
        """
        This method calculates the states aggregator.
        """
        pass

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
