import numpy as np


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


def calculate_gini_index_c1(data):  # Vectorized in 2D
    """
    Gini Calculation based on Milanovic's Concept 1 Inequality
    which is unweighted (population) international inequality.
    Concept 1 answers whether nations are converging in terms of their income levels.
    Here we are not interested in individuals, but countries. See Worlds Apart by Branko Milanovic.
    -------------------------------------------------------------------------------------------------
    @param data: 2D numpy array where data has shape (regions, timesteps)
    Vectorized implementation of Gini Index calculation for 2D data
    returns gini_coefficient over time
    """

    if data.ndim == 2:
        # Calculate the mean of the data
        mean_data = np.mean(data, axis=0)

        # Get the number of samples/regions
        sample_size = data.shape[0]

        # Difference matrix with broadcasting
        difference_matrix = data[:, np.newaxis, :] - data[np.newaxis, :, :]

        # Only consider positive differences
        positive_differences_sum = np.sum(
            difference_matrix * (difference_matrix > 0), axis=(0, 1)
        )

        gini_coefficient = (
            (1 / mean_data) * (1 / sample_size**2) * positive_differences_sum
        )

    elif data.ndim == 1:  # TESTED
        # Calculate the mean of the data
        mean_data = np.mean(data)

        # Get the number of samples/regions
        sample_size = data.shape[0]

        # Difference matrix with broadcasting
        difference_matrix = data[:, np.newaxis] - data[np.newaxis, :]

        # Only consider positive differences
        positive_differences_sum = np.sum(difference_matrix * (difference_matrix > 0))

        gini_coefficient = (
            (1 / mean_data) * (1 / sample_size**2) * positive_differences_sum
        )

    return gini_coefficient


def calculate_gini_index_c1_3D(data):  # Vectorized in 3D
    """
    Gini Calculation based on Milanovic's Concept 1 Inequality
    which is unweighted (population) international inequality.
    Concept 1 answers whether nations are converging in terms of their income levels.
    Here we are not interested in individuals, but countries. See Worlds Apart by Branko Milanovic.
    -------------------------------------------------------------------------------------------------
    @param data: 3D numpy array where data has shape (regions, timesteps, scenarios)
    Vectorized implementation of Gini Index calculation for 3D data
    returns gini_coefficient over time over scenarios
    """

    # Assert if data is not 3D
    assert data.ndim == 3, "Data must be 3D"

    # Calculate the mean of the data
    mean_data = np.mean(data, axis=0)  # Shape: (timesteps, scenarios)

    # Get the number of samples/regions
    sample_size = data.shape[0]  # Number of regions

    # Difference matrix with broadcasting
    difference_matrix = data[:, np.newaxis, :, :] - data[np.newaxis, :, :, :]
    # Shape: (regions, regions, timesteps, scenarios)

    # Only consider positive differences
    positive_differences_sum = np.sum(
        difference_matrix * (difference_matrix > 0), axis=(0, 1)
    )
    # Shape: (timesteps, scenarios)

    gini_coefficient = (1 / mean_data) * (1 / sample_size**2) * positive_differences_sum

    return gini_coefficient


def calculate_gini_index_c2(consumption_per_capita, population_ratio):
    """
    Gini Calculation based on Milanovic's Concept 2 Inequality
    which is population-weighted international inequality.
    Key assumption is “within country distribution is equal”, also often referred to as “world” income distribution.
    Concept 2 is in the middle and deals with neither nations nor individuals.
    It’s key advantage is it is a proxy/approximate for concept 3 inequality (the "true" world inequality), which is most difficult to compute
    See Worlds Apart by Branko Milanovic.
    -------------------------------------------------------------------------------------------------

    A full non vectorized implementation of Gini Index calculation will look like this:
    sum_of_differences = 0
    for j in reversed(range(0, data.shape[0])):
        for i in reversed(range(0, data.shape[0])):
            if data[j] > data[i]:
                sum_of_differences += (population_ratio[j] * population_ratio[i]) * (data[j] - data[i])

    gini_coefficient = mean_pop_weighted_cpc * sum_of_differences
    -------------------------------------------------------------------------------------------------
    @param consumption_per_capita: 2D numpy array where data has shape (regions, timesteps)
    @param population_ratio: 2D numpy array where data has shape (regions, timesteps)
    Not vectorized over timesteps or scenarios but vectorized over regions

    returns gini_coefficient over time
    """

    mean_population_weighted_consumption_per_capita = np.mean(
        (consumption_per_capita * population_ratio), axis=0
    )  # sum instead mean
    print(mean_population_weighted_consumption_per_capita.shape)
    # Create gini_coefficient array of same shape as consumption_per_capita
    sum_of_differences = np.zeros(consumption_per_capita.shape[1])
    for i in range(consumption_per_capita.shape[1]):
        consumption_per_capita_difference = np.subtract.outer(
            consumption_per_capita[:, i], consumption_per_capita[:, i]
        )
        mask = consumption_per_capita_difference > 0
        population_weighted_consumption_per_capita = np.outer(
            population_ratio[:, i], population_ratio[:, i]
        )
        sum_of_differences[i] = np.sum(
            population_weighted_consumption_per_capita[mask]
            * consumption_per_capita_difference[mask]
        )

    # Gini calculated in percentage. Hence, divide by 100
    gini_coefficient = (
        (1 / mean_population_weighted_consumption_per_capita) * sum_of_differences
    ) / 100

    return gini_coefficient
