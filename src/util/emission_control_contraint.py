"""
This is a helper module that contains the emission control constraint function for the model.
"""

import numpy as np


def emission_control_constraint(
    emission_control_rate,
    timestep,
    max_annual_growth_rate=0.4,
    emission_control_start_timestep=9,
    allow_fallback=False,
):
    """
    This function constrains the emission control rate based on the maximum annual growth rate and the emission control start timestep.

    Parameters
    ----------
    emission_control_rate : numpy.ndarray
        The emission control rate for each region at each timestep.
    timestep : int
        The current timestep.
    max_annual_growth_rate : float, optional
        The maximum annual growth rate, by default 0.4.
    emission_control_start_timestep : int, optional
        The emission control start timestep, by default 9, which is 2025 if the start year is 2015
    allow_fallback : bool, optional
        A flag to allow fallback to the previous emission control rate, by default False. Fallback prevents the emission control rate from decreasing from previous timestep.
    """
    if timestep >= emission_control_start_timestep:
        # Calculate the annual growth rate and handle ZeroDivisionError
        with np.errstate(divide="ignore", invalid="ignore"):
            annual_growth_rate = (
                (emission_control_rate - previous_emission_control_rate)
                / np.where(
                    previous_emission_control_rate != 0,
                    previous_emission_control_rate,
                    np.nan,
                )
            ) * 100

            # Replace NaN and Inf values with zero
            annual_growth_rate[np.isnan(annual_growth_rate)] = 0
            annual_growth_rate[np.isinf(annual_growth_rate)] = 0

        # Check if the annual growth rate is greater than the maximum annual growth rate, then set the emission control rate to the previous emission control rate with a growth rate of the maximum annual growth rate
        mask_high_growth = annual_growth_rate > max_annual_growth_rate
        emission_control_rate[mask_high_growth] = previous_emission_control_rate[
            mask_high_growth
        ] * (1 + max_annual_growth_rate)

        # If the allow_fallback is set to True, then set the constrained emission control rate to the current timestep emission control rate
        if allow_fallback:
            constrained_emission_control_rate = emission_control_rate
        else:
            # If the allow_fallback is set to False, then set the constrained emission control rate to the previous emission control rate
            mask_fallback = annual_growth_rate < 0
            emission_control_rate[mask_fallback] = previous_emission_control_rate[
                mask_fallback
            ]
            constrained_emission_control_rate = emission_control_rate

    else:
        # If the timestep is less than the emission control start timestep, then set the constrained emission control rate to zero
        previous_emission_control_rate = emission_control_rate
        constrained_emission_control_rate = np.zeros_like(
            emission_control_rate[:, timestep, :]
        )

    return constrained_emission_control_rate
