"""
This module calculate the welfare based on the different Disributive Justice Principles.
Derived from RICE50 model which is based on Berger et al. (2020).
* REFERENCES
* Berger, Loic, and Johannes Emmerling (2020): Welfare as Equity Equivalents, Journal of Economic Surveys 34, no. 4 (26 August 2020): 727-752. https://doi.org/10.1111/joes.12368.
"""

from typing import Any
import numpy as np
import pandas as pd
from src.util.enumerations import WelfareFunction
from src.objectives.objective_functions import (
    calculate_gini_index_c1,
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
        climate_ensembles,
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
        self.climate_ensembles = climate_ensembles
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
        self.discount_rate = np.tile(discount_rate, (len(self.region_list), 1))
        # TODO: Temporarily commented out
        # discount_rate = np.tile(discount_rate, (len(self.region_list), 1))  # Validated

        # Reshape discount_rate adding np.newaxis Changing shape from (timesteps,) to (timesteps, 1) # Validated Shape (57, 286, 1)
        # TODO: Probably Redundant after dimensions are aggregated in the order described in the paper by Berger & Emmerling
        # TODO: Temp comment out
        # self.discount_rate = discount_rate[:, :, np.newaxis]

        # Population is exogenously. So we don't need the 1001 copies across the ensemble members. Hence we select the first ensemble member
        population = population[:, :, 0]
        # Calculate the total population for each timestep # Validated
        total_population = np.sum(population, axis=0)

        # Calculate the population ratio for each timestep # Validated
        self.population_ratio = population / total_population

    def calculate_welfare(self, consumption_per_capita):
        """
        This method calculates the welfare.
        """
        # Check if consumption_per_capita has negative values
        consumption_per_capita = np.where(
            consumption_per_capita < 0, 1e-6, consumption_per_capita
        )

        # Aggregate the states dimension
        states_aggregated_consumption_per_capita = self.states_aggregator(
            consumption_per_capita,
            self.climate_ensembles,
            self.risk_aversion,
        )

        # Aggregate the Spatial Dimension
        spatially_aggregated_welfare = self.spatial_aggregator(
            states_aggregated_consumption_per_capita,
            self.population_ratio,
            self.elasticity_of_marginal_utility_of_consumption,
            self.inequality_aversion,
            self.egality_strictness,
            self.sufficiency_threshold,
        )

        # Aggregate the Temporal Dimension
        temporally_disaggregated_welfare, welfare = self.temporal_aggregator(
            data=spatially_aggregated_welfare,
            discount_rate=self.discount_rate,
        )

        return (
            spatially_aggregated_welfare,
            temporally_disaggregated_welfare,
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

        # Aggregate the states dimension
        states_aggregated_consumption_per_capita = self.states_aggregator(
            consumption_per_capita,
            self.climate_ensembles,
            self.risk_aversion,
        )

        spatially_aggregated_welfare = self.spatial_aggregator(
            states_aggregated_consumption_per_capita,
            self.population_ratio[:, timestep],
            self.elasticity_of_marginal_utility_of_consumption,
            self.inequality_aversion,
            self.egality_strictness,
            self.sufficiency_threshold,
        )

        # No Temporal aggregation in Stepwise calculation

        return spatially_aggregated_welfare

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

        # Adjust data with sufficiency threshold # Validated
        data = data - sufficiency_threshold

        # Calculate the consumption per capita raised to the power of 1 - inequality_aversion
        inequality_aversion_transformed_utility = self.utility_function(
            data, inequality_aversion
        )

        # Calculate the gini index of the  declining_marginal_utility_transformed_utility
        gini = calculate_gini_index_c1(data)

        # Adjusted gini with egality strictness parameter #Transforming inequality_aversion_transformed_utility with gini here makes no difference
        gini = gini * egality_strictness

        # Calculate the population weighted consumption per capita
        population_weighted_utility = (
            population_ratio * inequality_aversion_transformed_utility
        )

        # Aggregate Spatially
        weighted_sum_of_utility = np.sum(population_weighted_utility, axis=0)

        # Equality-Prioritarianism by Peterson. (1 - g(w1, w2, ..., wn)) * F(w1, w2, ..., wn), where g is the gini index
        # and F is the prioritarian transformation
        # # Applying gini to spatially aggregated welfare
        # # egalitarian measure should incorporate a measure of equality, multiplied or added to a measure of individual welfare
        weighted_sum_of_utility = weighted_sum_of_utility / ((1 - gini))

        # Invert the utility to consumption
        inequality_aversion_inverted_utility = self.inverse_utility_function(
            weighted_sum_of_utility, inequality_aversion
        )

        # Applying declining marginal utility to spatially aggregated welfare # Adding gini improves the welfare here
        spatially_aggregated_welfare = self.utility_function(
            inequality_aversion_inverted_utility,
            elasticity_of_marginal_utility_of_consumption,
        )

        # Check if sufficiency threshold is present. If it is zero, Can't transform it #TODO
        # if sufficiency_threshold != 0:
        #     # NOTE: In sufficientarian formulation, the welfare becomes positive. Still take absolute value and Minimize it.
        #     # Transform the sufficiency threshold
        #     sufficiency_threshold_transformed_utility = self.utility_function(
        #         sufficiency_threshold, elasticity_of_marginal_utility_of_consumption
        #     )
        # print(
        #     "Sufficiency Threshold Transformed Utility: ",
        #     sufficiency_threshold_transformed_utility,
        #     sufficiency_threshold,
        # )
        # Subtracting the transformed sufficiency threshold (or czero) from the aggregated welfare following Adler (2017)
        # spatially_aggregated_welfare = (
        #     spatially_aggregated_welfare - sufficiency_threshold_transformed_utility
        # )

        # returning disaggregated & aggregated version
        return spatially_aggregated_welfare

    @classmethod
    def temporal_aggregator(
        self,
        data,
        discount_rate,
    ):
        """
        This method calculates the temporal aggregator.
        """
        # TODO: Change that -1 later
        # Calculate the welfare disaggregated temporally
        temporally_disaggregated_welfare = (data - 1) * discount_rate[0, :]  # [0, :, :]

        # Temp
        # temporally_disaggregated_welfare = temporally_disaggregated_welfare #* (-1) #chage this temp

        # Welfare disaggregated temporally and regionally - For regional welfare calculation
        # welfare_regional_temporal = (data_disaggregated - 1) * discount_rate

        # Welfare aggregated regionally
        # welfare_regional = np.sum(
        #     welfare_regional_temporal,
        #     axis=1,
        # )

        # Calculate the welfare
        welfare = np.sum(
            temporally_disaggregated_welfare,
            axis=0,
        )

        return (
            # welfare_regional_temporal,
            temporally_disaggregated_welfare,
            # welfare_regional,
            welfare,
        )

    @classmethod
    def states_aggregator(self, data, climate_ensembles, risk_aversion):
        """
        This method calculates the states aggregator.
        According to Berger & Emmerling (2017), the social welfare function across a dimension is equal to
        the equity equivalent of consumption at the particular dimension
        The aggregated welfare can be calculated in the following steps:
        1. Transform consumption to utility
        2. Weigh the utility with respective weights & sum across the selected dimension
        3. Invert the utility to consumption for next dimension
        """

        # Transform the data according to the risk aversion
        risk_aversion_transformed_utility = self.utility_function(data, risk_aversion)

        # Weight the utility with the respective weight - probability of each states in this case.
        # Probability of each state is 1/climate_ensembles
        weighted_utility = risk_aversion_transformed_utility * (1 / climate_ensembles)

        # Sum the weighted utility across the states. Axis is 2 because data shape is (regions, timesteps, states)
        aggregated_utility = np.sum(weighted_utility, axis=-1)

        # Invert the utility
        risk_aversion_inverted_utility = self.inverse_utility_function(
            aggregated_utility, risk_aversion
        )

        return risk_aversion_inverted_utility

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
