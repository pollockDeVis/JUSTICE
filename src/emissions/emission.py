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

    def __init__(
        self,
        input_dataset,
        time_horizon,
    ):
        """
        This method initializes the OutputToEmissions class.
        """

        self.emissions_dict = copy.deepcopy(input_dataset.EMISSIONS_DICT)
        self.gdp_dict = copy.deepcopy(input_dataset.GDP_DICT)

        self.carbon_intensity_dict = {}

        for scenarios in self.gdp_dict.keys():
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
            # # interpolate GDP Dictionary
            # self._interpolate_gdp()

    def run_emissions(self, timestep, scenario, output, emission_control_rate):
        """
        This method calculates the emissions for the economic output of a given scenario.
        """

        emissions = (
            self.carbon_intensity_dict[scenario][:, timestep]
            * output
            * (1 - emission_control_rate)
        )

        return emissions

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
