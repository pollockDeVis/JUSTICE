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

# from scipy.special import erfc
from typing import Any
from config.default_parameters import DamageDefaults


class DamageKalkuhl:
    def __init__(self, input_dataset, time_horizon, climate_ensembles):
        self.region_list = input_dataset.REGION_LIST
        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
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
        ) / self.data_timestep
        self.coefficient_b = (
            self.interaction_term_temp_change_coefficient
            + self.lagged_interaction_term_temp_change_coefficient
        ) / self.data_timestep

        # self.climate_timestep_index = climate_model.__getattribute__(
        #     "justice_start_index"
        # )
        self.climate_ensembles = climate_ensembles
        # climate_model.__getattribute__("number_of_ensembles")
        # print(self.climate_timestep_index)

        # Create an temperature array of shape (region_list, damage window, climate_ensembles)
        self.temperature_array = np.zeros(
            (len(self.region_list), self.damage_window, self.climate_ensembles)
        )

        # Create an damage array of shape (region_list, damage window, climate_ensembles)
        # The window is of shape 2, to hold values for current and previous timestep
        self.damage_coefficient = np.zeros(
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

    def calculate_damage(self, temperature, timestep):
        """
        Returns the damage as a percentage for the given temperature and time step.
        Damages [Trill 2005 USD / year]
        Damages can be calculated using output - Damage = gross output * total_damage_fraction
        """
        # Assert that temperature is of shape (region_list, climate_ensembles)
        assert temperature.shape == (
            len(self.region_list),
            self.climate_ensembles,
        ), "Temperature array is not of shape (region_list, climate_ensembles)"

        # print(f"temperature: {temperature[:, 0]}")

        # DO a if else for the first timestep
        if timestep == 0:
            # Fill the temperature array with the current temperature data
            self.temperature_array[:, 0, :] = temperature
        elif timestep > 0 and timestep < (self.damage_window + 1):
            # Fill the temperature array with the current temperature data
            self.temperature_array[:, 1, :] = temperature
            # Calculate the temperature difference
            temperature_difference = (
                self.temperature_array[:, 1, :] - self.temperature_array[:, 0, :]
            )
            self.temperature_array[:, 0, :] = self.temperature_array[:, 1, :]

        elif timestep >= (self.damage_window + 1):
            # Fill the temperature array with the current temperature data
            self.temperature_array[:, 1, :] = temperature
            # print(f"temperature_array: {self.temperature_array[:, 1, :].shape}")

            # Calculate the temperature difference
            temperature_difference = (
                self.temperature_array[:, 1, :] - self.temperature_array[:, 0, :]
            )

            # Calculate the damage coefficient. Damage coefficient for current timestep is based on previous temperature #BIMPACT
            self.damage_coefficient[:, 1, :] = (
                self.coefficient_a * temperature_difference
                + self.coefficient_b
                * temperature_difference
                * self.temperature_array[:, 0, :]
            )

            # print(f"damage_coefficient: {self.damage_coefficient[:, 1, 0]}")

            # Calculate economic_damage_factor for current timestep #OMEGA
            np.divide(
                (1 + (self.economic_damage_factor[:, timestep - 1, :])),
                np.power((1 + self.damage_coefficient[:, 0, :]), self.data_timestep),
                out=self.economic_damage_factor[:, timestep, :],
            )
            self.economic_damage_factor[
                :, timestep, :
            ] -= 1  # "subtract 1 from each element in the slice of `economic_damage_factor`

            unbounded_damage_fraction = 1 - (
                np.divide(1, (1 + self.economic_damage_factor[:, timestep, :]))
            )

            # TODO: implement this later. Current calculations are not working
            # Threshold Damage (57,1001)
            # threshold_damage_fraction = self.damage_gdp_ratio_with_threshold * erfc(
            #     np.divide(
            #         (
            #             self.temperature_array[:, 1, :]
            #             - self.temperature_threshold_for_damage
            #         ),
            #         self.temperature_threshold_variation,
            #     )
            # )

            # Gradient Damage (57,1001)
            gradient_damage_fraction = self.damage_gdp_ratio_with_gradient * np.power(
                np.abs(
                    np.divide(
                        temperature_difference,
                        self.temperature_difference_scaling_factor,
                    )
                ),
                self.damage_growth_rate,
            )

            self.total_damage_fraction = (
                unbounded_damage_fraction + gradient_damage_fraction
            )
            # total_damage_fraction = unbounded_damage_fraction + gradient_damage_fraction
            # self.damage_to_output = 1 - total_damage_fraction

            # Update the first column of the temperature array and damage coefficient array for the next timestep
            self.temperature_array[:, 0, :] = self.temperature_array[:, 1, :]
            self.damage_coefficient[:, 0, :] = self.damage_coefficient[:, 1, :]

        return self.total_damage_fraction

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
