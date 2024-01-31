"""
This module wraps the model for uncertainty analysis experiments using EMA.
"""
import numpy as np

from src.model import JUSTICE
from src.util.enumerations import *
from emodps.rbf import RBF

# Scaling Values
max_temperature = 16.0
min_temperature = 0.0
max_difference = 2.0
min_difference = 0.0


def model_wrapper_emodps(**kwargs):
    scenario = kwargs.pop("ssp_rcp_scenario")
    elasticity_of_marginal_utility_of_consumption = kwargs.pop(
        "elasticity_of_marginal_utility_of_consumption"
    )
    pure_rate_of_social_time_preference = kwargs.pop(
        "pure_rate_of_social_time_preference"
    )
    inequality_aversion = kwargs.pop("inequality_aversion")

    economy_type = kwargs.pop("economy_type", (Economy.NEOCLASSICAL,))
    damage_function_type = kwargs.pop("damage_function_type", (DamageFunction.KALKUHL,))
    abatement_type = kwargs.pop("abatement_type", (Abatement.ENERDATA,))
    welfare_function = kwargs.pop("welfare_function", (WelfareFunction.UTILITARIAN,))

    n_regions = kwargs.pop("n_regions")
    n_timesteps = kwargs.pop("n_timesteps")

    n_inputs_rbf = kwargs.pop("n_inputs_rbf")
    n_outputs_rbf = kwargs.pop("n_outputs_rbf")

    rbf = RBF(n_rbfs=(n_inputs_rbf + 2), n_inputs=n_inputs_rbf, n_outputs=n_outputs_rbf)

    centers_shape, radii_shape, weights_shape = rbf.get_shape()

    centers = np.zeros(centers_shape)
    radii = np.zeros(radii_shape)
    weights = np.zeros(weights_shape)

    for i in range(centers_shape[0]):
        centers[i] = kwargs.pop(f"center {i}")
        radii[i] = kwargs.pop(f"radii {i}")

    for i in range(weights_shape[0]):
        weights[i] = kwargs.pop(f"weights {i}")

    # Populating the decision variables
    centers_flat = centers.flatten()
    radii_flat = radii.flatten()
    weights_flat = weights.flatten()

    decision_vars = np.concatenate((centers_flat, radii_flat, weights_flat))

    rbf.set_decision_vars(decision_vars)

    model = JUSTICE(
        scenario=scenario,
        economy_type=economy_type,
        damage_function_type=damage_function_type,
        abatement_type=abatement_type,
        social_welfare_function=welfare_function,
        # Declaring for endogenous fixed savings rate
        elasticity_of_marginal_utility_of_consumption=elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference=pure_rate_of_social_time_preference,
        inequality_aversion=inequality_aversion,
    )

    no_of_ensembles = model.__getattribute__("no_of_ensembles")

    datasets = {}
    # Initialize emissions control rate
    emissions_control_rate = np.zeros((n_regions, n_timesteps, no_of_ensembles))
    previous_temperature = 0
    difference = 0

    for timestep in range(n_timesteps):
        model.stepwise_run(
            emission_control_rate=emissions_control_rate[:, timestep, :],
            timestep=timestep,
            endogenous_savings_rate=True,
        )
        datasets = model.stepwise_evaluate(timestep=timestep)
        temperature = datasets["global_temperature"][timestep, :]

        if timestep % 5 == 0:
            difference = temperature - previous_temperature
            # Do something with the difference variable
            previous_temperature = temperature

        # Apply Min Max Scaling to temperature and difference
        scaled_temperature = (temperature - min_temperature) / (
            max_temperature - min_temperature
        )
        scaled_difference = (difference - min_difference) / (
            max_difference - min_difference
        )

        rbf_input = np.array([scaled_temperature, scaled_difference])

        # Check if this is not the last timestep
        if timestep < n_timesteps - 1:
            emissions_control_rate[:, timestep + 1, :] = rbf.apply_rbfs(rbf_input)

    datasets = model.evaluate()
    # Calculate the mean of ["welfare_utilitarian"]
    datasets["welfare_utilitarian"] = np.mean(datasets["welfare_utilitarian"])

    return datasets


def model_wrapper(**kwargs):
    scenario = kwargs.pop("ssp_rcp_scenario")
    elasticity_of_marginal_utility_of_consumption = kwargs.pop(
        "elasticity_of_marginal_utility_of_consumption"
    )
    pure_rate_of_social_time_preference = kwargs.pop(
        "pure_rate_of_social_time_preference"
    )
    inequality_aversion = kwargs.pop("inequality_aversion")
    economy_type = (Economy.NEOCLASSICAL,)
    damage_function_type = (DamageFunction.KALKUHL,)
    abatement_type = (Abatement.ENERDATA,)
    welfare_function = (WelfareFunction.UTILITARIAN,)

    n_regions = kwargs.pop("n_regions")
    n_timesteps = kwargs.pop("n_timesteps")

    savings_rate = np.zeros((n_regions, n_timesteps))
    emissions_control_rate = np.zeros((n_regions, n_timesteps))

    # TODO temporarily commented out
    for i in range(n_regions):
        for j in range(n_timesteps):
            savings_rate[i, j] = kwargs.pop(f"savings_rate {i} {j}")
            emissions_control_rate[i, j] = kwargs.pop(f"emissions_control_rate {i} {j}")

    # Optimal savings rate and emissions control rate RUNS from RICE50

    # TODO: Loading policy levers here - RICE50 optimal runs

    # ssp_scenario = get_economic_scenario(scenario)
    # optimal_emissions_control = np.load(
    #     "./data/input/solved_RICE50_data/interpolated_emissions_control.npy",
    #     allow_pickle=True,
    # )
    # optimal_savings_rate = np.load(
    #     "./data/input/solved_RICE50_data/interpolated_savings_rate.npy",
    #     allow_pickle=True,
    # )
    # savings_rate = optimal_savings_rate[ssp_scenario, :, :]
    # emissions_control_rate = optimal_emissions_control[ssp_scenario, :, :]

    model = JUSTICE(
        scenario=scenario,
        economy_type=economy_type,
        damage_function_type=damage_function_type,
        abatement_type=abatement_type,
        social_welfare_function=welfare_function,
    )

    model.run(savings_rate=savings_rate, emission_control_rate=emissions_control_rate)
    datasets = model.evaluate(
        welfare_function=welfare_function,
        elasticity_of_marginal_utility_of_consumption=elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference=pure_rate_of_social_time_preference,
        inequality_aversion=inequality_aversion,
    )

    return datasets


def get_outcome_names():
    return [
        "net_economic_output",
        "consumption_per_capita",
        "emissions",
        "global_temperature",
        "economic_damage",
        "abatement_cost",
        "disentangled_utility",
    ]
