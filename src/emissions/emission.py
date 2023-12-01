"""
This is the emissions module that converts the output into emissions.
#'CO2 emissions in [GtCO2/year]'
"""
from typing import Any
import numpy as np
from scipy.interpolate import interp1d
import copy
from src.util.enumerations import get_economic_scenario


class OutputToEmissions:
    """
    This class converts the output into emissions.
    """

    def __init__(self, input_dataset, time_horizon, climate_ensembles):
        """
        This method initializes the OutputToEmissions class.
        """
        # Saving the climate ensembles
        self.NUM_OF_ENSEMBLES = climate_ensembles

        self.emissions_array = copy.deepcopy(input_dataset.EMISSIONS_ARRAY)
        self.gdp_array = copy.deepcopy(input_dataset.GDP_ARRAY)
        self.region_list = input_dataset.REGION_LIST

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Initializing the carbon intensity 3D array
        self.carbon_intensity_array = np.zeros(
            (
                len(self.region_list),
                len(self.data_time_horizon),
                self.gdp_array.shape[2],
            )
        )

        for idx in range(self.gdp_array.shape[2]):
            self.carbon_intensity_array[:, :, idx] = (
                self.emissions_array[:, :, idx] / self.gdp_array[:, :, idx]
            )

        if self.timestep != self.data_timestep:
            # Interpolate Carbon Intensity Dictionary
            self._interpolate_carbon_intensity()

        # Initializing the emissions array
        self.emissions = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initialize 4D array for carbon intensity.
        self.carbon_intensity = np.zeros(
            (
                len(self.region_list),
                len(self.model_time_horizon),
                self.NUM_OF_ENSEMBLES,
                self.carbon_intensity_array.shape[2],
            )
        )

        # Loop through the scenarios in carbon intensity array to broadcast each one to the ensemble dimension.
        for idx in range(self.carbon_intensity_array.shape[2]):
            self.carbon_intensity[:, :, :, idx] = np.broadcast_to(
                self.carbon_intensity_array[:, :, idx, np.newaxis],
                (
                    self.carbon_intensity_array.shape[0],
                    self.carbon_intensity_array.shape[1],
                    self.NUM_OF_ENSEMBLES,
                ),
            )

    def run(self, scenario, timestep, output, emission_control_rate):
        """
        This method calculates the emissions for the economic output of a given scenario.
        carbon intensity shape (57, 1001)
        output shape (57, 1001)
        """

        scenario = get_economic_scenario(scenario)
        # Calculate emissions
        self.emissions[:, timestep, :] = (
            self.carbon_intensity[:, timestep, :, scenario]
            * output
            * (
                1 - emission_control_rate[:, np.newaxis]
            )  # Emisison Control Rate is a lever and might have to take timestep
        )

        return self.emissions[:, timestep, :]

    def emission_downscaler(self, aggregated_emissions):
        """
        This method downscales total emissions per timestep to regional emissions.
        """
        emissions_sum = np.sum(self.emissions, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            emissions_ratio = np.divide(
                self.emissions,
                emissions_sum[None, :, :],
                out=np.zeros_like(self.emissions),
                where=emissions_sum[None, :, :] != 0,
            )

        # Multiply the aggregated emissions by the ratio
        downscaled_emissions = emissions_ratio * aggregated_emissions

        return downscaled_emissions

    def get_fossil_and_land_use_emissions(self, land_use_emissions):
        """
        This method calculates the total CO2 fossil fuel and CO2 Land Use emissions.
        """
        # print("Land use before", land_use_emissions[0, 0])
        # Get downscaled land use emissions
        land_use_emissions = self.emission_downscaler(land_use_emissions)
        # print("Land use after", (land_use_emissions.sum(axis=0))[0, 0])

        # Sum CO2 fossil fuel and CO2 Land Use emissions
        fossil_and_land_use_emissions = self.emissions + land_use_emissions
        # print("Emission before", self.emissions[0, 0, 0])
        # print("Emission after", fossil_and_land_use_emissions[0, 0, 0])

        return fossil_and_land_use_emissions

    def _interpolate_carbon_intensity(self):
        interp_data = np.zeros(
            (
                self.carbon_intensity_array.shape[0],
                len(self.model_time_horizon),
                self.carbon_intensity_array.shape[2],
            )
        )

        for i in range(self.carbon_intensity_array.shape[0]):
            for j in range(self.carbon_intensity_array.shape[2]):
                f = interp1d(
                    self.data_time_horizon,
                    self.carbon_intensity_array[i, :, j],
                    kind="linear",
                )
                interp_data[i, :, j] = f(self.model_time_horizon)

        self.carbon_intensity_array = interp_data

    def get_emissions(self):
        """
        This method returns the emissions.
        """
        return self.emissions

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
