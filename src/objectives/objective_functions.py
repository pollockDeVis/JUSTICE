import numpy as np


# Compute the GINI coefficient
def calculate_gini_index(array):
    """Calculate the Gini index of a numpy array."""
    # Can be used to calculate both spatial and temporal inequality
    # O indicates perfect equality and 1 maximal inequality
    # based on bottom eq:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    # TODO: Can check spatial inequality by timestep or temporal inequality
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    # Check if array contains 0
    elif np.amin(array) == 0:
        array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def years_above_temperature_threshold(temperature, threshold):
    """Calculate the number of years above a temperature threshold."""
    # Temperature array should be 2D. Check if it is 2D, else throw an error.
    if temperature.ndim != 2:
        raise ValueError("Temperature array should be 2D.")
    # Check if threshold is a float, else throw an error.
    if not isinstance(threshold, float):
        raise ValueError("Threshold should be a float.")

    # Calculate the number of years above the threshold.
    number_of_years_above_threshold = np.mean(
        np.sum(np.greater(temperature, threshold), axis=0)
    )
    return number_of_years_above_threshold


def total_damage_cost(damage_cost):
    """Calculate the total damage cost."""

    # If damage_cost is not 3D, throw an error.
    if damage_cost.ndim != 3:
        raise ValueError("Damage cost should be 3D.")

    # Calculate the total damage cost across first two dimensions.
    total_damage_cost = np.sum(damage_cost, axis=(0, 1))

    # Calculate the mean of the total damage cost across ensemble members.
    total_damage_cost = np.mean(total_damage_cost)

    return total_damage_cost


def total_abatement_cost(abatement_cost):
    """Calculate the total abatement cost."""

    # If abatement_cost is not 3D, throw an error.
    if abatement_cost.ndim != 3:
        raise ValueError("Abatement cost should be 3D.")

    # Calculate the total abatement cost across first two dimensions.
    total_abatement_cost = np.sum(abatement_cost, axis=(0, 1))

    # Calculate the mean of the total abatement cost across ensemble members.
    total_abatement_cost = np.mean(total_abatement_cost)

    return total_abatement_cost
