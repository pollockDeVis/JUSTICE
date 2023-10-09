"""
This file contains the neoclassical economic part of the JUSTICE model.
"""
from typing import Any
from scipy.interpolate import interp1d
import numpy as np

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
        **kwargs,  # variable keyword argument so that we can analyze uncertainty range of any parameters
    ):
        # Create an instance of EconomyDefaults
        econ_defaults = EconomyDefaults()

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
        self.gdp_dict = input_dataset.GDP_DICT
        self.population_dict = input_dataset.POPULATION_DICT

        self.capital_init_arr = input_dataset.CAPITAL_INIT_ARRAY
        self.savings_rate_init_arr = (
            input_dataset.SAVING_RATE_INIT_ARRAY
        )  # Probably won't be used. Can be passed while setting up the Savings Rate Lever
        self.ppp_to_mer = input_dataset.PPP_TO_MER_CONVERSION_FACTOR
        self.mer_to_ppp = 1 / self.ppp_to_mer  # Conversion of MER to PPP

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Check if timestep is not equal to data timestep #If not, then interpolate

        if self.timestep != self.data_timestep:
            # Interpolate GDP
            self._interpolate_gdp()
            self._interpolate_population()

        # Calculate the Optimal long-run Savings Rate
        # optimal_long_run_savings_rate = ((depriciation_rate_capital + elasticity_of_output_to_capital)/(depriciation_rate_capital + elasticity_of_output_to_capital*elasticity_of_marginal_utility_of_consumption + pure_rate_of_social_time_preference))*capital_elasticity_in_production_function

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
            (len(self.region_list), len(self.model_time_horizon))
        )
        # Initializing the investment array
        self.investment_tfp = np.zeros(
            (len(self.region_list), len(self.model_time_horizon))
        )
        # Initializing the TFP array
        self.tfp = np.zeros((len(self.region_list), len(self.model_time_horizon)))

        # Initializing the output array
        self.output = np.zeros((len(self.region_list), len(self.model_time_horizon)))

        # Loading the initial capital values
        self.capital_tfp[:, 0] = (
            self.capital_init_arr.flatten() * self.mer_to_ppp[:, 0].flatten()
        )

    def run(self, scenario, timestep, savings_rate, **kwargs):
        # Run the calculation
        if timestep == 0:
            self.investment_tfp[:, 0] = (
                savings_rate * self.gdp_dict[scenario][:, 0].flatten()
            )
        else:
            # Calculate the investment_tfp
            self.investment_tfp[:, timestep] = (
                self.savings_rate_init_arr.flatten()
                * self.gdp_dict[scenario][:, 0].flatten()
            )

            # Calculate capital_tfp

            self.capital_tfp[:, timestep] = (
                self.capital_tfp[:, timestep - 1]
                * np.power((1 - self.depreciation_rate_capital), timestep)
                + self.investment_tfp[:, timestep - 1] * timestep
            )

            # Calculate the TFP
            self.tfp[:, timestep] = self.gdp_dict[scenario][:, timestep] / (
                np.power(
                    (self.population_dict[scenario][:, timestep] / 1000),
                    (1 - self.capital_elasticity_in_production_function),
                )
                * np.power(
                    self.capital_tfp[:, timestep],
                    self.capital_elasticity_in_production_function,
                )
            )

            # Calculate the Output based on gross output

            self.output[:, timestep] = self.tfp[:, timestep] * (
                np.power(
                    self.capital_tfp[:, timestep],
                    self.capital_elasticity_in_production_function,
                )
                * np.power(
                    (self.population_dict[scenario][:, timestep] / 1000),
                    (1 - self.capital_elasticity_in_production_function),
                )
            )

            # Check if kwargs has abatement and damage function specified (damage starts from the 2nd time step /maybe abatement too)
            abatement = kwargs.get("abatement")
            damage = kwargs.get("damage")

            if abatement is not None:
                self.output = self.output - abatement

            if damage is not None:
                self.output = self.output - damage

        return self.output

    def get_optimal_long_run_savings_rate(self):
        """
        This method returns the optimal long run savings rate.
        """
        return self.optimal_long_run_savings_rate

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
