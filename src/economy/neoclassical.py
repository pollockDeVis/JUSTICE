"""
This file contains the neoclassical economic part of the JUSTICE model.
"""
from typing import Any
from scipy.interpolate import interp1d
import numpy as np
import copy


from src.default_parameters import EconomyDefaults
from src.enumerations import Economy


class NeoclassicalEconomyModel:

    """
    This class describes the neoclassical economic part of the JUSTICE model.
    """

    def __init__(
        self,
        input_dataset,
        time_horizon,
        climate_ensembles,
        **kwargs,  # variable keyword argument so that we can analyze uncertainty range of any parameters
    ):
        # Create an instance of EconomyDefaults
        econ_defaults = EconomyDefaults()

        # Saving the climate ensembles
        self.NUM_OF_ENSEMBLES = climate_ensembles

        # Fetch the defaults for neoclassical submodule
        econ_neoclassical_defaults = econ_defaults.get_defaults(
            Economy.NEOCLASSICAL.name
        )

        # Assign retrieved values to instance variables if kwargs is empty

        self.capital_elasticity_in_production_function = kwargs.get(
            "capital_elasticity_in_production_function",
            econ_neoclassical_defaults["capital_elasticity_in_production_function"],
        )
        self.depreciation_rate_capital = kwargs.get(
            "depreciation_rate_capital",
            econ_neoclassical_defaults["depreciation_rate_capital"],
        )
        self.elasticity_of_marginal_utility_of_consumption = kwargs.get(
            "elasticity_of_marginal_utility_of_consumption",
            econ_neoclassical_defaults["elasticity_of_marginal_utility_of_consumption"],
        )
        self.pure_rate_of_social_time_preference = kwargs.get(
            "pure_rate_of_social_time_preference",
            econ_neoclassical_defaults["pure_rate_of_social_time_preference"],
        )
        self.elasticity_of_output_to_capital = kwargs.get(
            "elasticity_of_output_to_capital",
            econ_neoclassical_defaults["elasticity_of_output_to_capital"],
        )

        self.region_list = input_dataset.REGION_LIST
        self.gdp_dict = copy.deepcopy(input_dataset.GDP_DICT)
        self.population_dict = copy.deepcopy(input_dataset.POPULATION_DICT)

        self.capital_init_arr = input_dataset.CAPITAL_INIT_ARRAY
        self.savings_rate_init_arr = (
            input_dataset.SAVING_RATE_INIT_ARRAY
        )  # Probably won't be used. Can be passed while setting up the Savings Rate Lever
        self.ppp_to_mer = input_dataset.PPP_TO_MER_CONVERSION_FACTOR
        self.mer_to_ppp = 1 / self.ppp_to_mer  # Conversion of MER to PPP

        # mer_to_ppp is a 2D array. Need to convert it to (regions, 1)
        self.mer_to_ppp = self.mer_to_ppp[:, 0:1]

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Check if timestep is not equal to data timestep #If not, then interpolate

        if self.timestep != self.data_timestep:
            # Interpolate GDP
            self._interpolate_gdp()
            self._interpolate_population()

        # List of scenarios
        self.scenario_list = list(self.gdp_dict.keys())

        # Calculate the Optimal long-run Savings Rate
        # This will depend on the input paramters. This is also a upper limit of the savings rate
        self.optimal_long_run_savings_rate = (
            (self.depreciation_rate_capital + self.elasticity_of_output_to_capital)
            / (
                self.depreciation_rate_capital
                + self.elasticity_of_output_to_capital
                * self.elasticity_of_marginal_utility_of_consumption
                + self.pure_rate_of_social_time_preference
            )
        ) * self.capital_elasticity_in_production_function

        # Initializing the capital and TFP array
        self.capital_tfp = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the investment array
        self.investment_tfp = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        # Initializing the TFP array
        self.tfp = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the output array
        self.output = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the damages array
        self.damages = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        # Initial 4D array for gdp and population.
        self.gdp = np.zeros(
            (
                len(self.region_list),
                len(self.model_time_horizon),
                self.NUM_OF_ENSEMBLES,
                len(self.scenario_list),
            )
        )
        self.population = np.zeros(
            (
                len(self.region_list),
                len(self.model_time_horizon),
                self.NUM_OF_ENSEMBLES,
                len(self.scenario_list),
            )
        )

        # Loop through the scenarios to broadcast each one to the ensemble dimension.
        for idx, scenario in enumerate(self.scenario_list):
            self.gdp[:, :, :, idx] = np.broadcast_to(
                self.gdp_dict[scenario][:, :, np.newaxis],
                (
                    self.gdp_dict[scenario].shape[0],
                    self.gdp_dict[scenario].shape[1],
                    self.NUM_OF_ENSEMBLES,
                ),
            )
            self.population[:, :, :, idx] = np.broadcast_to(
                self.population_dict[scenario][:, :, np.newaxis],
                (
                    self.population_dict[scenario].shape[0],
                    self.population_dict[scenario].shape[1],
                    self.NUM_OF_ENSEMBLES,
                ),
            )

    def run(self, scenario, timestep, savings_rate):  # **kwargs
        # Reshaping savings rate
        if len(savings_rate.shape) == 1:
            savings_rate = savings_rate.reshape(-1, 1)

        if timestep == 0:
            self.investment_tfp[:, 0, :] = (
                self.savings_rate_init_arr * self.gdp[:, 0, :, scenario]
            )

            # Initalize capital tfp
            self.capital_tfp[:, 0, :] = self.capital_init_arr * self.mer_to_ppp

            # Calculate the TFP
            self._calculate_tfp(timestep, scenario)

            # Calculate the Output
            self._calculate_output(timestep, scenario)

        else:
            # Calculate the investment_tfp
            self.investment_tfp[:, timestep, :] = (
                savings_rate * self.gdp[:, 0, :, scenario]
            )

            # Calculate capital_tfp

            self.capital_tfp[:, timestep, :] = (
                self.capital_tfp[:, timestep - 1, :]
                * np.power((1 - self.depreciation_rate_capital), timestep)
                + self.investment_tfp[:, timestep - 1, :] * timestep
            )

            # Calculate the TFP
            self._calculate_tfp(timestep, scenario)

            # Calculate the Output based on gross output
            self._calculate_output(timestep, scenario)

        return self.output[:, timestep, :]

    def get_optimal_long_run_savings_rate(self):
        """
        This method returns the optimal long run savings rate.
        """
        return self.optimal_long_run_savings_rate

    def _calculate_tfp(self, timestep, scenario):
        # Calculate the TFP
        self.tfp[:, timestep, :] = self.gdp[:, timestep, :, scenario] / (
            np.power(
                (self.population[:, timestep, :, scenario] / 1000),
                (1 - self.capital_elasticity_in_production_function),
            )
            * np.power(
                self.capital_tfp[:, timestep, :],
                self.capital_elasticity_in_production_function,
            )
        )

    def _calculate_output(self, timestep, scenario):
        # Calculate the Output based on gross output

        self.output[:, timestep, :] = self.tfp[:, timestep, :] * (
            np.power(
                self.capital_tfp[:, timestep, :],
                self.capital_elasticity_in_production_function,
            )
            * np.power(
                (self.population[:, timestep, :, scenario] / 1000),
                (1 - self.capital_elasticity_in_production_function),
            )
        )
        # Subtract damages from output
        self.output[:, timestep, :] = (
            self.output[:, timestep, :] - self.damages[:, timestep, :]
        )

    def apply_damage_to_output(self, timestep, damage):
        """
        This method applies damage to the output.
        Damage calculated
        """
        self.damages[:, timestep, :] = damage

    def _interpolate_gdp(self):
        for keys in self.gdp_dict.keys():
            gdp_SSP = self.gdp_dict[keys]
            interp_data = np.zeros((len(gdp_SSP), len(self.model_time_horizon)))

            for i in range(gdp_SSP.shape[0]):
                f = interp1d(self.data_time_horizon, gdp_SSP[i, :], kind="linear")
                interp_data[i, :] = f(self.model_time_horizon)

            self.gdp_dict[keys] = interp_data

    def _interpolate_population(self):
        for keys in self.population_dict.keys():
            population_SSP = self.population_dict[keys]
            interp_data = np.zeros((len(population_SSP), len(self.model_time_horizon)))

            for i in range(population_SSP.shape[0]):
                f = interp1d(
                    self.data_time_horizon, population_SSP[i, :], kind="linear"
                )
                interp_data[i, :] = f(self.model_time_horizon)

            self.population_dict[keys] = interp_data

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
