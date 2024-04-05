"""
This is a helper module that contains the emission control constraint function for the model.
"""

import numpy as np


class EmissionControlConstraint:
    """
    A class that represents an emission control constraint.

    Attributes:
        max_annual_growth_rate (float): The maximum annual growth rate allowed.
        emission_control_start_timestep (int): The timestep at which emission control starts.
        previous_emission_control_rate (ndarray): The previous emission control rate.

    Methods:
        constrain_emission_control_rate(emission_control_rate, timestep, allow_fallback=False):
            Constrains the emission control rate based on the given timestep and maximum growth rate.
    """

    def __init__(
        self,
        max_annual_growth_rate,
        emission_control_start_timestep,
        min_emission_control_rate=0.01,
    ):
        self.max_annual_growth_rate = max_annual_growth_rate
        self.emission_control_start_timestep = emission_control_start_timestep
        self.previous_emission_control_rate = None
        self.constrained_emission_control_rate = None
        self.min_emission_control_rate = min_emission_control_rate

    def constrain_emission_control_rate(
        self, emission_control_rate, timestep, allow_fallback=False
    ):
        """
        Constrains the emission control rate based on the given timestep and maximum growth rate.

        Args:
            emission_control_rate (ndarray): The emission control rate.
            timestep (int): The current timestep.
            allow_fallback (bool, optional): Whether to allow fallback if growth rate is negative. Defaults to False.

        Returns:
            ndarray: The constrained emission control rate.
        """

        if timestep >= self.emission_control_start_timestep:

            with np.errstate(divide="ignore", invalid="ignore"):
                annual_growth_rate = (
                    emission_control_rate - self.previous_emission_control_rate
                ) / np.where(
                    self.previous_emission_control_rate != 0,
                    self.previous_emission_control_rate,
                    np.nan,
                )

                # Setting it to 1 which is the highest growth rate if there are nan or inf values
                annual_growth_rate[np.isnan(annual_growth_rate)] = 1
                annual_growth_rate[np.isinf(annual_growth_rate)] = 1

            if np.any(annual_growth_rate > self.max_annual_growth_rate):
                mask_high_growth = annual_growth_rate > self.max_annual_growth_rate
                emission_control_rate[mask_high_growth] = (
                    self.previous_emission_control_rate[mask_high_growth]
                    * (1 + self.max_annual_growth_rate)
                )

            if allow_fallback:
                self.constrained_emission_control_rate = emission_control_rate
            else:
                if np.any(annual_growth_rate < 0):
                    mask_fallback = annual_growth_rate < 0
                    emission_control_rate[mask_fallback] = (
                        self.previous_emission_control_rate[mask_fallback]
                    )
                    self.constrained_emission_control_rate = emission_control_rate

            # Check if emission_control_rate has any negative elements. If yes, set to 0.01
            if np.any(emission_control_rate < 0):
                mask_negative = emission_control_rate < 0
                emission_control_rate[mask_negative] = self.min_emission_control_rate

            self.constrained_emission_control_rate = emission_control_rate

            self.previous_emission_control_rate = self.constrained_emission_control_rate

        # Timesteps before the emission_control_start_timestep
        elif timestep < self.emission_control_start_timestep:
            self.constrained_emission_control_rate = np.zeros_like(
                emission_control_rate
            )
            self.previous_emission_control_rate = self.constrained_emission_control_rate

            if timestep == (self.emission_control_start_timestep - 1):
                self.previous_emission_control_rate = emission_control_rate

        return self.constrained_emission_control_rate
