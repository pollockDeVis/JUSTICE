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

from config.default_parameters import AbatementDefaults
from src.util.enumerations import Abatement


class AbatementEnerdata:
    """
    This class computes the abatement costs for the JUSTICE model.
    """

    def __init__(
        self, input_dataset, time_horizon, **kwargs
    ):  # TODO maybe this has to move to the calculate abatement
        """
        This method initializes the Abatement class.
        """
        # Create an instance of the AbatementDefaults class
        abatement_defaults = AbatementDefaults()

        self.abatement_coefficient_a = input_dataset.ABATEMENT_COEFFICIENT_A
        self.abatement_coefficient_b = input_dataset.ABATEMENT_COEFFICIENT_B
        self.region_list = input_dataset.REGION_LIST

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Fetch the defaults for ENERDATA
        abatement_enerdata_defaults = abatement_defaults.get_defaults(
            Abatement.ENERDATA.name
        )

        # Assign the defaults to the class attributes
        self.calibrated_correction_multiplier_starting_value = (
            abatement_enerdata_defaults[
                "calibrated_correction_multiplier_starting_value"
            ]
        )
        # self.exponential_control_cost_function = abatement_enerdata_defaults[
        #     "exponential_control_cost_function"
        # ] #TODO this is used for another abatement calculation - need to check later
        self.backstop_cost = kwargs.get(
            "backstop_cost", abatement_enerdata_defaults["backstop_cost"]
        )
        self.backstop_cost_decline_rate_per_5_year = kwargs.get(
            "backstop_cost_decline_rate_per_5_year",
            abatement_enerdata_defaults["backstop_cost_decline_rate_per_5_year"],
        )
        self.transition_year_start = kwargs.get(
            "transition_year_start",
            abatement_enerdata_defaults["transition_year_start"],
        )
        self.transition_year_end = kwargs.get(
            "transition_year_end", abatement_enerdata_defaults["transition_year_end"]
        )
        self.logistic_transition_speed_per_5_year = kwargs.get(
            "logistic_transition_speed_per_5_year",
            abatement_enerdata_defaults["logistic_transition_speed_per_5_year"],
        )

        if self.timestep != self.data_timestep:
            # Interpolate GDP
            self._interpolate_coefficients()

        # Start here

        # Backstop Calculation

        if self.timestep != self.data_timestep:
            backstop_cost_decline_rate = np.power(
                1 - self.backstop_cost_decline_rate_per_5_year, 1 / self.data_timestep
            )
        else:
            backstop_cost_decline_rate = 1 - self.backstop_cost_decline_rate_per_5_year

        # This is pbacktime in RICE50
        global_backstop_cost_curve = self.backstop_cost * np.power(
            backstop_cost_decline_rate, np.arange(len(self.model_time_horizon))
        )

        # Calculate calibrated_correction_multiplier
        calibrated_correction_multiplier = global_backstop_cost_curve[np.newaxis, :] / (
            self.abatement_coefficient_a + self.abatement_coefficient_b
        )

        # Calculate multiplier_difference
        calibrated_correction_multiplier_starting_value_arr = np.full(
            (len(self.region_list), len(self.model_time_horizon)),
            self.calibrated_correction_multiplier_starting_value,
        )
        multiplier_difference = np.maximum(
            calibrated_correction_multiplier_starting_value_arr
            - calibrated_correction_multiplier,
            0,
        )

        # Backstop Transition Calculation

        if self.timestep != self.data_timestep:  # If Timestep is not 1 year
            logistic_transition_speed = (
                self.logistic_transition_speed_per_5_year / self.data_timestep
            )
        else:
            logistic_transition_speed = self.logistic_transition_speed_per_5_year

        backstop_transition_period = time_horizon.year_to_timestep(
            self.transition_year_start, self.timestep
        ) + (
            (
                time_horizon.year_to_timestep(self.transition_year_end, self.timestep)
                - time_horizon.year_to_timestep(
                    self.transition_year_start, self.timestep
                )
            )
            / 2
        )

        # validated
        transition_coefficient = 1 / (
            1
            + np.exp(
                -logistic_transition_speed
                * (np.arange(len(self.model_time_horizon)) - backstop_transition_period)
            )
        )

        self.coefficient_multiplier = (
            calibrated_correction_multiplier_starting_value_arr
            - multiplier_difference * transition_coefficient
        )

    def calculate_abatement(self, timestep, emissions, emission_control_rate):
        """
        This method calculates the abatement for the emissions of a given timestep.
        * y ~ ax + bx^4   --with multiplier-->   y ~ mx (ax + bx^4)
        * with constraints (R-colf optim package) avoiding negative costs

        * Abatement Cost ::   mx * (a(x^2)/2 + b(x^5)/5) * bau   :: [$/tCO2]x[GtCO2] ->  [ G$ ]
        Shape of coefficient_multiplier:  (57,)
        Shape of abatement_coefficient_a:  (57,)
        Shape of abatement_coefficient_b:  (57,)
        Shape of emission_control_rate:  (57,)
        Shape of emissions:  (57, 1001)

        @return: abatement [Trill 2005 USD / year]
        """
        # Calculate abatement
        abatement = (
            self.coefficient_multiplier[:, timestep, np.newaxis]
            * (
                self.abatement_coefficient_a[:, timestep, np.newaxis]
                * ((np.power(emission_control_rate, 2)) / 2)  # [:, np.newaxis]
                + self.abatement_coefficient_b[:, timestep, np.newaxis]
                * ((np.power(emission_control_rate, 5)) / 5)  # [:, np.newaxis]
            )
            * (emissions / 1000)  #   Conversion:  [ G$ ] / 1000 -> [Trill $]
        )
        return abatement

    def calculate_carbon_price(self, timestep, emission_control_rate):
        """
        This method calculates the carbon price (Carbon Price Enerdata) for the emissions of a given timestep.
        * Carbon Price ::   y ~ mx (ax + bx^4)
        * CPrice will result in [$/tCO2] by construction
        * Carbon Price [ 2005 USD $/tCO2 ]
        """

        # Calculate carbon price
        carbon_price = self.coefficient_multiplier[:, timestep] * (
            self.abatement_coefficient_a[:, timestep] * (emission_control_rate)
            + self.abatement_coefficient_b[:, timestep]
            * (np.power(emission_control_rate, 4))
        )

        return carbon_price

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
