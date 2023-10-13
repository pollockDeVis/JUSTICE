"""
This is the emissions module that converts the output into emissions.
"""

import numpy as np
from scipy.interpolate import interp1d
import copy


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

        self.emissions_dict = copy.deepcopy(input_dataset.EMISSIONS_DICT)
        self.gdp_dict = copy.deepcopy(input_dataset.GDP_DICT)
        self.region_list = input_dataset.REGION_LIST

        self.carbon_intensity_dict = {}

        # List of scenarios
        self.scenario_list = list(self.gdp_dict.keys())

        for scenarios in self.scenario_list:  # self.gdp_dict.keys()
            self.carbon_intensity_dict[scenarios] = (
                np.array(self.emissions_dict[scenarios]) / self.gdp_dict[scenarios]
            )

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

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
                len(self.scenario_list),
            )
        )

        # Loop through the scenarios in carbon intensity dict to broadcast each one to the ensemble dimension.
        for idx, scenario in enumerate(self.scenario_list):
            self.carbon_intensity[:, :, :, idx] = np.broadcast_to(
                self.carbon_intensity_dict[scenario][:, :, np.newaxis],
                (
                    self.carbon_intensity_dict[scenario].shape[0],
                    self.carbon_intensity_dict[scenario].shape[1],
                    self.NUM_OF_ENSEMBLES,
                ),
            )

    def run_emissions(self, scenario, timestep, output, emission_control_rate):
        """
        This method calculates the emissions for the economic output of a given scenario.
        """
        # Calculate emissions
        self.emissions[:, timestep, :] = (
            self.carbon_intensity[:, timestep, :, scenario]
            * output
            * (
                1 - emission_control_rate
            )  # Emisison Control Rate is a lever and might have to take timestep
        )

        return self.emissions

    def _interpolate_carbon_intensity(self):
        for keys in self.carbon_intensity_dict.keys():
            carbon_intensity_SSP = self.carbon_intensity_dict[keys]
            interp_data = np.zeros(
                (len(carbon_intensity_SSP), len(self.model_time_horizon))
            )

            for i in range(carbon_intensity_SSP.shape[0]):
                f = interp1d(
                    self.data_time_horizon, carbon_intensity_SSP[i, :], kind="linear"
                )
                interp_data[i, :] = f(self.model_time_horizon)

            self.carbon_intensity_dict[keys] = interp_data

    # def _interpolate_gdp(self):
    #     for keys in self.gdp_dict.keys():
    #         gdp_SSP = self.gdp_dict[keys]
    #         interp_data = np.zeros((len(gdp_SSP), len(self.model_time_horizon)))

    #         for i in range(gdp_SSP.shape[0]):
    #             f = interp1d(self.data_time_horizon, gdp_SSP[i, :], kind="linear")
    #             interp_data[i, :] = f(self.model_time_horizon)

    #         self.gdp_dict[keys] = interp_data
