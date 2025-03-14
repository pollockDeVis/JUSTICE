"""
Kalkuhl damage function taken from RICE50. It is a recursive function with the following parameters:

1. kw_DT:  `short_run_temperature_change_coefficient`.

2. kw_DT_lag: `lagged_short_run_temperature_change_coefficient`.

3. kw_TDT: `interaction_term_temperature_change_coefficient`.

4. kw_TDT_lag:  `lagged_interaction_term_temperature_change_coefficient`.

5. kw_T:  `temperature_dependent_coefficient`.

Kalkuhl Reference: https://www.sciencedirect.com/science/article/pii/S0095069620300838
"""

import numpy as np
from typing import Any
from config.default_parameters import DamageDefaults


class DamageKalkuhl:
    def __init__(self, input_dataset, time_horizon, climate_ensembles):
        self.region_list = input_dataset.REGION_LIST
        self.data_timestep = time_horizon.data_timestep
        self.model_timestep = time_horizon.timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        damage_defaults = DamageDefaults().get_defaults("KALKUHL")

        self.short_run_temp_change_coefficient = damage_defaults[
            "short_run_temp_change_coefficient"
        ]
        self.lagged_short_run_temp_change_coefficient = damage_defaults[
            "lagged_short_run_temp_change_coefficient"
        ]
        self.interaction_term_temp_change_coefficient = damage_defaults[
            "interaction_term_temp_change_coefficient"
        ]
        self.lagged_interaction_term_temp_change_coefficient = damage_defaults[
            "lagged_interaction_term_temp_change_coefficient"
        ]
        self.temperature_dependent_coefficient = damage_defaults[
            "temperature_dependent_coefficient"
        ]
        self.damage_window = damage_defaults["damage_window"]

        self.damage_gdp_ratio_with_threshold = damage_defaults[
            "damage_gdp_ratio_with_threshold"
        ]
        self.temperature_threshold_for_damage = damage_defaults[
            "temperature_threshold_for_damage"
        ]
        self.temperature_threshold_variation = damage_defaults[
            "temperature_threshold_variation"
        ]
        self.damage_gdp_ratio_with_gradient = damage_defaults[
            "damage_gdp_ratio_with_gradient"
        ]
        self.temperature_difference_scaling_factor = damage_defaults[
            "temperature_difference_scaling_factor"
        ]
        self.damage_growth_rate = damage_defaults["damage_growth_rate"]

        self.coefficient_a = (
            self.short_run_temp_change_coefficient
            + self.lagged_short_run_temp_change_coefficient
        ) / self.model_timestep  # NOTE: # Kalkuhl uses annual rates
        self.coefficient_b = (
            self.interaction_term_temp_change_coefficient
            + self.lagged_interaction_term_temp_change_coefficient
        ) / self.model_timestep  # NOTE: # Kalkuhl uses annual rates

        self.climate_ensembles = climate_ensembles

        # Create an temperature array of shape (region_list, damage window, climate_ensembles)
        self.temperature_array = np.zeros(
            (len(self.region_list), self.damage_window, self.climate_ensembles)
        )

        # Create an damage array of shape (region_list, damage window, climate_ensembles)
        # The window is of shape 2, to hold values for current and previous timestep
        self.damage_specification = np.zeros(
            (len(self.region_list), self.damage_window, self.climate_ensembles)
        )

        # Create economic damage factor array of shape (region_list, damage window, climate_ensembles)
        self.economic_damage_factor = np.zeros(
            (
                len(self.region_list),
                len(self.model_time_horizon),
                self.climate_ensembles,
            )
        )

        # Create damage to output array of shape (region_list, climate_ensembles)
        self.total_damage_fraction = np.zeros(
            (
                len(self.region_list),
                self.climate_ensembles,
            )
        )

    # def calculate_damage(self, temperature, timestep):
    #     """
    #     Returns the damage as a percentage for the given temperature and time step.
    #     Damages [Trill 2005 USD / year]
    #     Damages can be calculated using output - Damage = gross output * total_damage_fraction
    #     """
    #     # Assert that temperature is of shape (region_list, climate_ensembles)
    #     assert temperature.shape == (
    #         len(self.region_list),
    #         self.climate_ensembles,
    #     ), "Temperature array is not of shape (region_list, climate_ensembles)"

    #     # Initialize the temperature array
    #     if timestep == 0:
    #         # Fill the temperature array with the current temperature data
    #         self.temperature_array[:, 0, :] = temperature
    #     elif timestep > 0 and timestep < (self.damage_window + 1):
    #         # Fill the temperature array with the current temperature data
    #         self.temperature_array[:, 1, :] = temperature
    #         # Calculate the temperature difference
    #         temperature_difference = (
    #             self.temperature_array[:, 1, :] - self.temperature_array[:, 0, :]
    #         )
    #         self.temperature_array[:, 0, :] = self.temperature_array[:, 1, :]

    #     elif timestep >= (self.damage_window + 1):
    #         # Fill the temperature array with the current temperature data
    #         self.temperature_array[:, 1, :] = temperature
    #         # print(f"temperature_array: {self.temperature_array[:, 1, :].shape}")

    #         # Calculate the temperature difference
    #         temperature_difference = (
    #             self.temperature_array[:, 1, :] - self.temperature_array[:, 0, :]
    #         )

    #         # Calculate the damage specification based on Kalkuhl params. Damage coefficient for current timestep is based on previous temperature #BIMPACT
    #         self.damage_specification[:, 1, :] = (
    #             self.coefficient_a * temperature_difference
    #             + self.coefficient_b
    #             * temperature_difference
    #             * self.temperature_array[:, 0, :]
    #         )

    #         # Calculate economic_damage_factor for current timestep #OMEGA
    #         np.divide(
    #             (1 + (self.economic_damage_factor[:, timestep - 1, :])),
    #             np.power(
    #                 (1 + self.damage_specification[:, 0, :]), self.model_timestep
    #             ),  # self.data_timestep
    #             out=self.economic_damage_factor[:, timestep, :],
    #         )
    #         self.economic_damage_factor[
    #             :, timestep, :
    #         ] -= 1  # "subtract 1 from each element in the slice of `economic_damage_factor`

    #         unbounded_damage_fraction = 1 - (
    #             np.divide(1, (1 + self.economic_damage_factor[:, timestep, :]))
    #         )

    #         # TODO: implement this later. Current calculations are not working
    #         # Threshold Damage (57,1001)
    #         # threshold_damage_fraction = self.damage_gdp_ratio_with_threshold * erfc(
    #         #     np.divide(
    #         #         (
    #         #             self.temperature_array[:, 1, :]
    #         #             - self.temperature_threshold_for_damage
    #         #         ),
    #         #         self.temperature_threshold_variation,
    #         #     )
    #         # )

    #         # Gradient Damage (57,1001)
    #         gradient_damage_fraction = self.damage_gdp_ratio_with_gradient * np.power(
    #             np.abs(
    #                 np.divide(
    #                     temperature_difference,
    #                     self.temperature_difference_scaling_factor,
    #                 )
    #             ),
    #             self.damage_growth_rate,
    #         )

    #         self.total_damage_fraction = (
    #             unbounded_damage_fraction + gradient_damage_fraction
    #         )

    #         # Update the first column of the temperature array and damage coefficient array for the next timestep
    #         self.temperature_array[:, 0, :] = self.temperature_array[:, 1, :]
    #         self.damage_specification[:, 0, :] = self.damage_specification[:, 1, :]

    #     return self.total_damage_fraction

    def calculate_damage(self, temperature, timestep):
        """
        Calculate and return the damage fraction for a given temperature array and time step.

        Damage (in USD/year Trillions) is computed via an economic damage factor that is updated
        based on the change in temperature. The damage specification is computed through a linear
        combination of temperature difference and its product with the previous temperature, and then
        the economic damage factor is updated. Damages are then given by:

            total_damage_fraction = unbounded_damage_fraction + gradient_damage_fraction

        The temperature array is stored with two columns (index 0 and 1) representing the previous and
        current timestep values respectively.

        Parameters:
            temperature : np.ndarray
                An array with shape (number_of_regions, climate_ensembles) representing the current temperature.
            timestep : int
                The simulation time step.

        Returns:
            np.ndarray : The computed total damage fraction.
        """
        # Validate that the temperature array has the expected shape.
        expected_shape = (len(self.region_list), self.climate_ensembles)
        assert (
            temperature.shape == expected_shape
        ), f"Temperature array must be of shape {expected_shape} but got {temperature.shape}"

        # For performance, use local variables for arrays to reduce repeated attribute lookup.
        temp_arr = self.temperature_array  # shape: (region, 2, ensembles)
        damage_spec = self.damage_specification  # shape: (region, 2, ensembles)
        econ_damage = (
            self.economic_damage_factor
        )  # shape: (region, timesteps, ensembles)

        # --------------------------
        # Case 1: Initial Timestep
        # --------------------------
        if timestep == 0:
            # Initialize the first column of temperature.
            temp_arr[:, 0, :] = temperature
            # For an initial timestep, damage might not be computed yet.
            # (self.total_damage_fraction may be preinitialized or remain unchanged.)
            return self.total_damage_fraction

        # --------------------------
        # Case 2: Warm-up Period before Full Damage Calculation
        # --------------------------
        # (Assuming self.damage_window defines a number of time steps needed before full damage calc.)
        if timestep < (self.damage_window + 1):
            # Update the current temperature slot.
            temp_arr[:, 1, :] = temperature

            # Compute the temperature difference between "new" and "old" values.
            temperature_difference = temp_arr[:, 1, :] - temp_arr[:, 0, :]

            # Update the previous temperature with the current one for the next timestep.
            temp_arr[:, 0, :] = temp_arr[:, 1, :]

            # In the warm-up window, the full damage calculation might not be desired.
            return self.total_damage_fraction

        # --------------------------
        # Case 3: Full Damage Calculation
        # --------------------------
        # Update the current temperature.
        temp_arr[:, 1, :] = temperature

        # Compute the temperature difference using a temporary variable.
        temperature_difference = temp_arr[:, 1, :] - temp_arr[:, 0, :]

        # Calculate damage specification:
        # Damage specification for the current timestep is based on the temperature difference and
        # the previous temperature stored in temp_arr[:, 0, :].
        damage_spec[:, 1, :] = (
            self.coefficient_a * temperature_difference
            + self.coefficient_b * temperature_difference * temp_arr[:, 0, :]
        )

        # Update the economic damage factor:
        # The economic damage factor is computed as:
        #   (1 + economic_damage_factor_previous) / (1 + damage_specification_previous)^(model_timestep) - 1
        #
        # Using the previous economic damage factor stored at timestep-1 and the previous damage specification.
        prev_econ = econ_damage[:, timestep - 1, :]
        denom = np.power(1 + damage_spec[:, 0, :], self.model_timestep)
        np.divide(1 + prev_econ, denom, out=econ_damage[:, timestep, :])
        econ_damage[:, timestep, :] -= 1  # subtract 1 in place

        # Compute the unbounded damage fraction.
        # This is 1 minus the reciprocal of (1 + current economic damage factor).
        unbounded_damage_fraction = 1 - (1 / (1 + econ_damage[:, timestep, :]))

        # Compute gradient damage fraction:
        # Damage due to the gradient (or rate of change) in temperature.
        gradient_damage_fraction = self.damage_gdp_ratio_with_gradient * np.power(
            np.abs(temperature_difference / self.temperature_difference_scaling_factor),
            self.damage_growth_rate,
        )

        # Total damage fraction is the sum of the unbounded and gradient components.
        self.total_damage_fraction = (
            unbounded_damage_fraction + gradient_damage_fraction
        )

        # Update the stored temperature and damage specification for use in the next timestep.
        temp_arr[:, 0, :] = temp_arr[:, 1, :]
        damage_spec[:, 0, :] = damage_spec[:, 1, :]

        return self.total_damage_fraction

    def reset(self):
        """
        Reset self.total_damage_fraction to zeros.
        """
        self.total_damage_fraction = np.zeros(
            (
                len(self.region_list),
                self.climate_ensembles,
            )
        )

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
