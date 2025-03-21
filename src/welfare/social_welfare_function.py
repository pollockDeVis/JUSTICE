"""
This module calculate the welfare based on the different Disributive Justice Principles.
Derived from RICE50 model which is based on Berger et al. (2020).
* REFERENCES
* Berger, Loic, and Johannes Emmerling (2020): Welfare as Equity Equivalents, Journal of Economic Surveys 34, no. 4 (26 August 2020): 727-752. https://doi.org/10.1111/joes.12368.
"""

from typing import Any
import numpy as np
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
        limitarian_threshold_emissions,
        limitarian_start_year_of_remaining_budget,
    ):
        """
        This method initializes the Social Welfare Function class.
        """
        self.region_list = input_dataset.REGION_LIST

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon
        self.time_horizon = time_horizon
        self.climate_ensembles = climate_ensembles
        self.risk_aversion = risk_aversion
        self.elasticity_of_marginal_utility_of_consumption = (
            elasticity_of_marginal_utility_of_consumption
        )
        self.pure_rate_of_social_time_preference = pure_rate_of_social_time_preference
        self.inequality_aversion = inequality_aversion
        self.sufficiency_threshold = sufficiency_threshold
        self.egality_strictness = egality_strictness
        self.limitarian_threshold_emissions = limitarian_threshold_emissions
        self.limitarian_start_year_of_remaining_budget = (
            limitarian_start_year_of_remaining_budget
        )

        # Population is exogenous. So we don't need the 1001 copies across the ensemble members. Hence we select the first ensemble member
        population = population[:, :, 0]
        # Calculate the total population for each timestep # Validated
        total_population = np.sum(population, axis=0)

        # Calculate the population ratio for each timestep # Validated
        self.population_ratio = population / total_population

    def calculate_welfare(self, consumption_per_capita, emissions=None, **kwargs):
        """
        This method calculates the welfare.
        """
        # Check if consumption_per_capita has negative values # FIXME Optimize this
        consumption_per_capita = np.where(
            consumption_per_capita <= 0, 1e-6, consumption_per_capita
        )

        # Aggregate the states dimension
        states_aggregated_consumption_per_capita = self.states_aggregator(
            consumption_per_capita,
            self.climate_ensembles,
            self.risk_aversion,
        )

        # Limitarian Threshold Check
        if self.limitarian_threshold_emissions > 0 and emissions is not None:

            global_emissions = np.sum(emissions, axis=0)
            cumulative_emissions = np.cumsum(global_emissions, axis=0)

            index_from_year = self.time_horizon.year_to_timestep(
                (self.limitarian_start_year_of_remaining_budget - self.timestep),
                timestep=self.timestep,
            )  # Remaining Budget from the beginning of the year. Hence we take the cumulative emissions from the previous year

            # Taking the ensemble mean of global cumulative emissions
            mean_global_cumulative_emissions = np.mean(cumulative_emissions, axis=1)
            emission_before_start_year = mean_global_cumulative_emissions[
                index_from_year
            ]
            updated_limitarian_threshold = (
                self.limitarian_threshold_emissions + emission_before_start_year
            )
            # Get the index when the global cumulative emissions exceed the updated limitarian threshold
            emission_limit_index = np.argmax(
                mean_global_cumulative_emissions > updated_limitarian_threshold
            )

            # For emission_limit_index onwards, the states_aggregated_consumption_per_capita should be the value of states_aggregated_consumption_per_capita[emission_limit_index - self.timestep] for all time steps
            # Get the slice of states_aggregated_consumption_per_capita from emission_limit_index - self.timestep
            states_aggregated_consumption_per_capita_before_limit = (
                states_aggregated_consumption_per_capita[  # Shape (regions,)
                    :, (emission_limit_index - self.timestep)
                ]
            )

            # Replicate the value of states_aggregated_consumption_per_capita_before_limit for the remaining timesteps in states_aggregated_consumption_per_capita which is of shape (regions, timesteps)
            states_aggregated_consumption_per_capita[:, emission_limit_index:] = (
                states_aggregated_consumption_per_capita_before_limit[:, None]
            )

        # Aggregate the Spatial Dimension
        spatially_aggregated_welfare = self.spatial_aggregator(
            states_aggregated_consumption_per_capita,
            self.population_ratio,
            self.elasticity_of_marginal_utility_of_consumption,
            self.inequality_aversion,
            self.egality_strictness,
            self.sufficiency_threshold,
            **kwargs,
        )

        # Aggregate the Temporal Dimension
        temporally_disaggregated_welfare, welfare = self.temporal_aggregator(
            data=spatially_aggregated_welfare,
            pure_rate_of_social_time_preference=self.pure_rate_of_social_time_preference,
            model_time_horizon=self.model_time_horizon,
            timestep=self.timestep,
        )

        return (
            states_aggregated_consumption_per_capita,
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

        # Reward of individual agents - MARL
        # Applying declining marginal utility to states aggregated welfare
        stepwise_marl_reward = self.utility_function(
            states_aggregated_consumption_per_capita,
            self.elasticity_of_marginal_utility_of_consumption,
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

        return spatially_aggregated_welfare, stepwise_marl_reward

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
        **kwargs,
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
        # Check kwargs for welfare_loss flag
        if (
            "welfare_loss" in kwargs
        ):  # Sufficentarian threshold not to be applied on non-consumption related metric (doesn't make sense)
            welfare_loss = kwargs["welfare_loss"]
        else:
            # Adjust data with sufficiency threshold # Validated
            data = data - sufficiency_threshold

        # Calculate the consumption per capita raised to the power of 1 - inequality_aversion
        inequality_aversion_transformed_utility = self.utility_function(
            data, inequality_aversion
        )

        # Calculate the population weighted consumption per capita
        population_weighted_utility = (
            population_ratio * inequality_aversion_transformed_utility
        )

        # Aggregate Spatially
        weighted_sum_of_utility = np.sum(population_weighted_utility, axis=0)

        # Calculate the gini index of the data
        if egality_strictness != 0:
            gini = calculate_gini_index_c1(data)

            # Adjusted gini with egality strictness parameter #Transforming inequality_aversion_transformed_utility with gini here makes no difference
            gini = gini * egality_strictness
            # Equality-Prioritarianism by Peterson. (1 - g(w1, w2, ..., wn)) * F(w1, w2, ..., wn), where g is the gini index
            # and F is the prioritarian transformation
            # # Applying gini to spatially aggregated welfare
            # [Enflo] egalitarian measure should incorporate a measure of equality, multiplied or added to a measure of individual welfare
            weighted_sum_of_utility = weighted_sum_of_utility * ((1 - gini))

        # Invert the utility to consumption
        inequality_aversion_inverted_utility = self.inverse_utility_function(
            weighted_sum_of_utility, inequality_aversion
        )

        # Applying declining marginal utility to spatially aggregated welfare # Adding gini improves the welfare here
        spatially_aggregated_welfare = self.utility_function(
            inequality_aversion_inverted_utility,
            elasticity_of_marginal_utility_of_consumption,
        )

        # returning disaggregated & aggregated version
        return spatially_aggregated_welfare

    # @classmethod
    def temporal_aggregator(
        self,
        data,
        pure_rate_of_social_time_preference,
        model_time_horizon,
        timestep,
    ):
        """
        This method calculates the temporal aggregator.
        """
        # Get the discount rate array
        discount_rate = self.calculate_discount_rate(
            pure_rate_of_social_time_preference=pure_rate_of_social_time_preference,
            model_time_horizon=model_time_horizon,
            timestep=timestep,
        )

        # TODO: Change that -1 later
        # Calculate the welfare disaggregated temporally
        temporally_disaggregated_welfare = (data - 1) * discount_rate

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

    def calculate_discount_rate(
        self, pure_rate_of_social_time_preference, model_time_horizon, timestep
    ):
        """
        This method calculates the discount rate.
        """

        # Time horizon
        timestep_list = np.arange(len(np.array(model_time_horizon)))
        # np.arange(0, len(model_time_horizon), timestep)

        # Calculate the discount rate
        discount_rate = 1 / (
            np.power(
                (1 + pure_rate_of_social_time_preference),
                (timestep * (timestep_list)),
            )
        )
        return discount_rate

    def calculate_discount_rate_v2(  # Tested. Generates more optimistic welfare values than the previous method
        self, pure_rate_of_social_time_preference, model_time_horizon
    ):
        """
        This method calculates the discount rate with a different mathematical formula
        discount_rate = e^(-pure_rate_of_social_time_preference * timestep_list)

        """

        # Time horizon
        timestep_list = np.arange(len(np.array(model_time_horizon)))

        # Vectorize the computation to get the discounted values
        discount_rate = np.exp(-pure_rate_of_social_time_preference * timestep_list)

        return discount_rate

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
