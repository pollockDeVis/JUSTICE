"""
This file contains the matter-use part of the JUSTICE model. 
this module is inspired in the DEFINE-MATTER (Dafermos, 2021) set of equations
"""

from typing import Any
from scipy.interpolate import interp1d
import numpy as np
import copy
from config.default_parameters import EconomicSubModules

from src.util.enumerations import get_economic_scenario
from src.economy.neoclassical import NeoclassicalEconomyModel


class MatterDefaults:  # Already moved to default_parameters.py
    def __init__(self):
        self.physical_use_ratio = 0.74  # per year
        self.discard_rate = 0.013  # per year
        self.conversion_rate_material_reserves = 0.0015  # per year
        self.recycling_rate = 0.2  # per year


class MatterUse:
    """
    This class describes the matter-use dynamics in the JUSTICE model.
    """

    def __init__(
        self,
        input_dataset,
        time_horizon,
        climate_ensembles,
        economy,
        recycling_rate=None,
    ):

        # Load the defaults #TODO Angela - you can implement this
        matter_defaults = EconomicSubModules().get_defaults("MATTER")

        # Initialize defaults
        defaults = MatterDefaults()

        # Load the instantiated economy model and set it as an attribute
        self.economy = economy

        # Parameters
        self.physical_use_ratio = defaults.physical_use_ratio
        self.discard_rate = defaults.discard_rate
        self.conversion_rate_material_reserves = (
            defaults.conversion_rate_material_reserves
        )

        # Policy Lever #TODO Angela - should you put it here or in run method? This is because recycling rate is a policy lever
        self.recycling_rate = (
            recycling_rate if recycling_rate is not None else defaults.recycling_rate
        )

        # Saving the climate ensembles ?
        self.NUM_OF_ENSEMBLES = climate_ensembles

        # Saving the scenario
        self.scenario = self.economy.scenario
        # self.scenario = get_economic_scenario(scenario)

        self.region_list = input_dataset.REGION_LIST
        self.material_intensity_array = copy.deepcopy(
            input_dataset.MATERIAL_INTENSITY_ARRAY
        )

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Selecting only the required scenario
        self.material_intensity_array = self.material_intensity_array[
            :, :, self.scenario
        ]

        if self.timestep != self.data_timestep:
            # Interpolate Material Intensity Dictionary
            self._interpolate_material_intensity()

        """
        Initialize matter-use variables arrays
        """

        # TODO This should go into your run() method and not initialization. Net is only calculated in the run
        self.net_output = self.economy.net_output

        # Intializing the material intensity array Unit: kg/USD per year
        self.material_intensity = self.material_intensity_array

        # Intializing the material intensity array Unit: Gt per year
        self.material_consumption = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the in-use stock array Unit: Gt per year
        # TODO check this
        self.in_use_stock = copy.deepcopy(input_dataset.IN_USE_STOCK_INIT_ARRAY)

        # Intializing the discarded material array Unit: Gt per year
        self.discarded_material = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the recycled material array Unit: Gt per year
        self.recycled_material = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the waste array Unit: Gt per year
        self.waste = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the extracted matter array Unit: Gt per year
        self.extracted_matter = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the material reserves array Unit: Gt per year
        self.material_reserves = copy.deepcopy(
            input_dataset.MATERIAL_RESERVES_INIT_ARRAY
        )

        # Intializing the converted material reserves array Unit: Gt per year
        self.converted_material_reserves = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the material resources array Unit: Gt per year
        self.material_resources = copy.deepcopy(
            input_dataset.MATERIAL_RESOURCES_INIT_ARRAY
        )

        # Intializing the depletion ratio
        self.depletion_ratio = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

    def run(self, timestep, recycling_rate):
        if len(recycling_rate.shape) == 1:
            recycling_rate = recycling_rate.reshape(-1, 1)
        self.get_material_consumption(timestep)
        self.get_in_use_stock(timestep)
        self.get_discarded_material(timestep)
        self.get_recycled_material(timestep)
        self.get_waste(timestep)
        self.get_extracted_matter(timestep)
        self.get_material_reserves(timestep)
        self.get_converted_material_reserves(timestep)
        self.get_material_resources(timestep)
        self.get_depletion_ratio(timestep)

    """Matter-use variable calculations functions"""

    def get_material_consumption(self, timestep):
        # Calculate the domestic material consumption based on material intensity and net economic output
        self.material_consumption[:, timestep, :] = (
            self.material_intensity[:, timestep, :]
            * self.net_output[:, timestep, :]
            * 1000
        ) / 1_000_000_000  # Convert to Gt

    def get_in_use_stock(self, timestep):
        if timestep == 0:
            self.in_use_stock[:, timestep, :] = self.in_use_stock[:, timestep, :]
        else:
            self.in_use_stock[:, timestep, :] = (
                self.in_use_stock[:, timestep - 1, :]
                + self.material_consumption[:, timestep, :] * self.physical_use_ratio
                - self.discarded_material[:, timestep, :]
            )

    def get_discarded_material(self, timestep):
        self.discarded_material[:, timestep, :] = (
            self.discard_rate * self.in_use_stock[:, timestep - 1, :]
        )

    def get_recycled_material(self, timestep):
        self.recycled_material[:, timestep, :] = (
            self.recycling_rate * self.discarded_material[:, timestep, :]
        )

    def get_waste(self, timestep):
        self.waste[:, timestep, :] = (
            self.discarded_material[:, timestep, :]
            - self.recycled_material[:, timestep, :]
        )

    def get_extracted_matter(self, timestep):
        self.extracted_matter[:, timestep, :] = (
            self.material_consumption[:, timestep, :]
            - self.recycled_material[:, timestep, :]
        )

    def get_material_reserves(self, timestep):
        if timestep == 0:
            self.material_reserves[:, timestep, :] = self.material_reserves[
                :, timestep, :
            ]
        else:
            self.material_reserves[:, timestep, :] = (
                self.material_reserves[:, timestep - 1, :]
                + self.converted_material_reserves[:, timestep, :]
                - self.extracted_matter[:, timestep, :]
            )

    def get_converted_material_reserves(self, timestep):
        self.converted_material_reserves[:, timestep, :] = (
            self.conversion_rate_material_reserves
            * self.material_resources[:, timestep - 1, :]
        )

    def get_material_resources(self, timestep):
        if timestep == 0:
            self.material_resources[:, timestep, :] = self.material_resources[
                :, timestep, :
            ]
        else:
            self.material_resources[:, timestep, :] = (
                self.material_resources[:, timestep - 1, :]
                - self.converted_material_reserves[:, timestep, :]
            )

    def get_depletion_ratio(self, timestep):
        self.depletion_ratio[:, timestep, :] = (
            self.extracted_matter[:, timestep, :]
            / self.material_resources[:, timestep - 1, :]
        )

    # TODO check ALL THIS BELOW I just copy this from the other modules
    def _interpolate_material_intensity(self):
        interp_data = np.zeros(
            (
                self.material_intensity_array.shape[0],
                len(self.model_time_horizon),
                # self.gdp_array.shape[2],
            )
        )
        for i in range(self.material_intensity_array.shape[0]):
            f = interp1d(
                self.data_time_horizon,
                self.material_intensity_array[i, :],
                kind="linear",  # , j
            )
            interp_data[i, :] = f(self.model_time_horizon)  # , j

        self.material_intensity_array = interp_data

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
