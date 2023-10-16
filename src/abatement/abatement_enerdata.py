"""
This file computes the Marginal Abatement Cost (MAC) for the JUSTICE model using the Enerdata dataset.
#From RICE50:
#Regional Matginal Abatement Cost (RMAC) curves
#Three periods: Near Future (2025-2040)- continuous curves on EnerData-EnerFuture Data Projections based on process based model POLES
#For rest of the century (2040-2100) - Emissions and Abatement potential based on IAMs reviewed in IPCC SR1.5 (IPCC 2018)
#Post 2100, model assumptions converge to DICE trend driven by backstop technology

#Transitionary phase happens at 2045 for common backstop technology
"""

from typing import Any
from scipy.interpolate import interp1d
import numpy as np


class AbatementEnerdata:
    """
    This class computes the abatement costs for the JUSTICE model.
    """

    def __init__(self, input_dataset, time_horizon):
        """
        This method initializes the Abatement class.
        """
        self.abatement_coefficient_a = input_dataset.ABATEMENT_COEFFICIENT_A
        self.abatement_coefficient_b = input_dataset.ABATEMENT_COEFFICIENT_B
        self.region_list = input_dataset.REGION_LIST

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        if self.timestep != self.data_timestep:
            # Interpolate GDP
            self._interpolate_coefficients()

    """
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
    """

    # def _interpolate_coefficients(self):
    #     interp_data = np.zeros((len(self.region_list), len(self.model_time_horizon)))
    #     print(interp_data.shape)
    #     print(self.model_time_horizon)

    #     for i in range(len(self.region_list)):
    #         f = interp1d(
    #             self.data_time_horizon,
    #             self.abatement_coefficient_a[i, :],
    #             kind="linear",
    #         )
    #         interp_data[i, :] = f(self.model_time_horizon)

    #     self.abatement_coefficient_a = interp_data

    #     for i in range(len(self.region_list)):
    #         f = interp1d(
    #             self.data_time_horizon,
    #             self.abatement_coefficient_b[i, :],
    #             kind="linear",
    #         )
    #         interp_data[i, :] = f(self.model_time_horizon)
    #     self.abatement_coefficient_b = interp_data

    def _interpolate_coefficients(self):
        interp_data_a = np.zeros((len(self.region_list), len(self.model_time_horizon)))
        interp_data_b = np.zeros((len(self.region_list), len(self.model_time_horizon)))

        for i in range(self.abatement_coefficient_a.shape[0]):
            f = interp1d(
                self.data_time_horizon,
                self.abatement_coefficient_a[i, :],
                kind="linear",
            )
            interp_data_a[i, :] = f(self.model_time_horizon)
        self.abatement_coefficient_a = interp_data_a

        for i in range(self.abatement_coefficient_b.shape[0]):
            f = interp1d(
                self.data_time_horizon,
                self.abatement_coefficient_b[i, :],
                kind="linear",
            )
            interp_data_b[i, :] = f(self.model_time_horizon)
        self.abatement_coefficient_b = interp_data_b

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
