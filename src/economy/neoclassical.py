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
        self.savings_rate_init_arr = input_dataset.SAVING_RATE_INIT_ARRAY

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Check if timestep is not equal to data timestep #If not, then interpolate

        if self.timestep != self.data_timestep:
            # Interpolate GDP
            self._interpolate_gdp()
            self._interpolate_population()

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
