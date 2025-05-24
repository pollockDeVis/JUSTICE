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

            # Check if emission_control_rate has any negative elements. If yes, set to zero
            if np.any(emission_control_rate < 0):
                mask_negative = emission_control_rate < 0
                emission_control_rate[mask_negative] = 0

            # Initialize the constrained_emission_control_rate
            if timestep == self.emission_control_start_timestep:
                global_min_ecr = np.min(emission_control_rate)
                global_max_ecr = np.max(emission_control_rate)
                global_range_ecr = global_max_ecr - global_min_ecr

                # Check if the range is smaller than 1e-16
                if global_range_ecr < 1e-16:
                    # If the range is too small, set the emission_control_rate to a small value
                    scaled_emission_control_rate = np.zeros_like(emission_control_rate)
                else:
                    scaled_emission_control_rate = (
                        emission_control_rate - global_min_ecr
                    ) / global_range_ecr
                # scaled_emission_control_rate = (
                #     emission_control_rate - np.min(emission_control_rate)
                # ) / (
                #     np.max(emission_control_rate) - np.min(emission_control_rate)
                # )  # TODO: Requires logic to avoid division by zero
                # Adjusting the feature range
                scaled_emission_control_rate = (
                    scaled_emission_control_rate
                    * (self.max_annual_growth_rate - self.min_emission_control_rate)
                ) + self.min_emission_control_rate

                # Setting the constrained_emission_control_rate to the scaled_emission_control_rate
                self.constrained_emission_control_rate = (
                    scaled_emission_control_rate  # emission_control_rate
                )
                annual_growth_rate = 0  # Initial value
            elif timestep > self.emission_control_start_timestep:
                annual_growth_rate = (
                    emission_control_rate - self.previous_emission_control_rate
                )

                if np.any(annual_growth_rate > self.max_annual_growth_rate):
                    mask_high_growth = annual_growth_rate > self.max_annual_growth_rate
                    emission_control_rate[mask_high_growth] = (
                        self.previous_emission_control_rate[mask_high_growth]
                        + self.max_annual_growth_rate
                    )
                    self.constrained_emission_control_rate = emission_control_rate

                if allow_fallback == False:
                    if np.any(annual_growth_rate < 0):
                        mask_fallback = annual_growth_rate < 0
                        self.constrained_emission_control_rate[mask_fallback] = (
                            self.previous_emission_control_rate[mask_fallback]
                        )

            self.previous_emission_control_rate = self.constrained_emission_control_rate

        # Timesteps before the emission_control_start_timestep
        elif timestep < self.emission_control_start_timestep:
            # Initialize the constrained_emission_control_rate to zeros same as the emission_control_rate shape
            self.constrained_emission_control_rate = np.zeros_like(
                emission_control_rate
            )

        return self.constrained_emission_control_rate
