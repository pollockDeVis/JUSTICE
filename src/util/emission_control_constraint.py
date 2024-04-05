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

    def __init__(self, max_annual_growth_rate, emission_control_start_timestep):
        self.max_annual_growth_rate = max_annual_growth_rate
        self.emission_control_start_timestep = emission_control_start_timestep
        self.previous_emission_control_rate = None
        self.constrained_emission_control_rate = None

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
            # if timestep == self.emission_control_start_timestep:
            #     # Initialize the previous_emission_control_rate with a self.max_annual_growth_rate small value of 1e-6
            #     self.previous_emission_control_rate = (
            #         np.ones_like(emission_control_rate)
            #         * self.max_annual_growth_rate  # 1e-6
            #     )

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
                emission_control_rate[mask_negative] = 0.01

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


# def emission_control_constraint(
#     emission_control_rate,
#     previous_emission_control_rate,
#     timestep,
#     max_annual_growth_rate=0.04,
#     emission_control_start_timestep=9,
#     allow_fallback=False,
# ):
#     """
#     This function constrains the emission control rate based on the maximum annual growth rate and the emission control start timestep.

#     Parameters
#     ----------
#     emission_control_rate : numpy.ndarray
#         The emission control rate for each region at each timestep.
#     timestep : int
#         The current timestep.
#     max_annual_growth_rate : float, optional
#         The maximum annual growth rate, by default 0.4.
#     emission_control_start_timestep : int, optional
#         The emission control start timestep, by default 9, which is 2025 if the start year is 2015
#     allow_fallback : bool, optional
#         A flag to allow fallback to the previous emission control rate, by default False. Fallback prevents the emission control rate from decreasing from previous timestep.
#     """
#     constrained_emission_control_rate = emission_control_rate
#     previous_emission_control_rate

#     if timestep >= emission_control_start_timestep:
#         # Calculate the annual growth rate and handle ZeroDivisionError
#         with np.errstate(divide="ignore", invalid="ignore"):
#             annual_growth_rate = (
#                 emission_control_rate - previous_emission_control_rate
#             ) / np.where(
#                 previous_emission_control_rate != 0,
#                 previous_emission_control_rate,
#                 np.nan,
#             )  # * 100

#             # Replace NaN and Inf values with zero
#             annual_growth_rate[np.isnan(annual_growth_rate)] = 1
#             annual_growth_rate[np.isinf(annual_growth_rate)] = 1

#         # Check if the annual growth rate is greater than the maximum annual growth rate, then set the emission control rate to the previous emission control rate with a growth rate of the maximum annual growth rate
#         if np.any(annual_growth_rate > max_annual_growth_rate):
#             mask_high_growth = annual_growth_rate > max_annual_growth_rate
#             emission_control_rate[mask_high_growth] = previous_emission_control_rate[
#                 mask_high_growth
#             ] * (1 + max_annual_growth_rate)

#         # else

#         # If the allow_fallback is set to True, then set the constrained emission control rate to the current timestep emission control rate
#         if allow_fallback:
#             constrained_emission_control_rate = emission_control_rate
#         else:
#             # If the allow_fallback is set to False, then set the constrained emission control rate to the previous emission control rate
#             if np.any(annual_growth_rate < 0):
#                 mask_fallback = annual_growth_rate < 0
#                 emission_control_rate[mask_fallback] = previous_emission_control_rate[
#                     mask_fallback
#                 ]
#                 constrained_emission_control_rate = emission_control_rate

#         previous_emission_control_rate = constrained_emission_control_rate

#     elif timestep < emission_control_start_timestep:
#         # If the timestep is less than the emission control start timestep, then set the constrained emission control rate to zero

#         constrained_emission_control_rate = np.zeros_like(emission_control_rate)
#         previous_emission_control_rate = (
#             emission_control_rate  # constrained_emission_control_rate
#         )

#     return constrained_emission_control_rate, previous_emission_control_rate
