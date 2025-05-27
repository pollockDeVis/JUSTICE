"""
This module wraps the model for uncertainty analysis experiments using EMA.
"""

# TODO: Move this to solvers. Create a clean solving interface for JUSTICE model using EMA Workbench

import numpy as np

from justice.model import JUSTICE
from justice.util.enumerations import *
from solvers.emodps.rbf import RBF
from justice.objectives.objective_functions import (
    years_above_temperature_threshold,
    fraction_of_ensemble_above_threshold,
)

from justice.util.emission_control_constraint import EmissionControlConstraint

# Scaling Values
max_temperature = 16.0
min_temperature = 0.0
max_difference = 2.0
min_difference = 0.0


def model_wrapper_emodps(**kwargs):
    scenario = kwargs.pop("ssp_rcp_scenario")
    social_welfare_function_type = kwargs.pop("social_welfare_function_type")

    # Check kwargs for "economy_type", "damage_function_type", "abatement_type
    if "economy_type" in kwargs:
        economy_type = Economy.from_index(kwargs["economy_type"])
    if "damage_function_type" in kwargs:
        damage_function_type = DamageFunction.from_index(kwargs["damage_function_type"])
    if "abatement_type" in kwargs:
        abatement_type = Abatement.from_index(kwargs["abatement_type"])
    if "stochastic_run" in kwargs:
        stochastic_run = kwargs.pop("stochastic_run")
    # Loading the constants
    n_regions = kwargs.pop("n_regions")
    n_timesteps = kwargs.pop("n_timesteps")
    emission_control_start_timestep = kwargs.pop("emission_control_start_timestep")

    n_inputs_rbf = kwargs.pop("n_inputs_rbf")
    n_outputs_rbf = kwargs.pop("n_outputs_rbf")

    temperature_year_of_interest_index = kwargs.pop(
        "temperature_year_of_interest_index"
    )

    climate_ensemble_members = kwargs.pop("climate_ensemble_members")

    rbf = RBF(
        n_rbfs=(n_inputs_rbf + 2), n_inputs=n_inputs_rbf, n_outputs=n_outputs_rbf
    )  # n_inputs_rbf is a rule of thumb. Hasn't been verified yet for complex models

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

    # NOTE: Could get the growth rate and min_emission_control_rate from the kwargs
    emission_constraint = EmissionControlConstraint(
        max_annual_growth_rate=0.04,
        emission_control_start_timestep=emission_control_start_timestep,
        min_emission_control_rate=0.01,
    )

    # --- Singleton logic for JUSTICE ---
    if not hasattr(model_wrapper_emodps, "justice_instance"):
        # First call: create the instance (this does heavy initialization)
        model_wrapper_emodps.justice_instance = JUSTICE(
            scenario=scenario,
            economy_type=economy_type,
            damage_function_type=damage_function_type,
            abatement_type=abatement_type,
            social_welfare_function_type=social_welfare_function_type,
            stochastic_run=stochastic_run,
            climate_ensembles=climate_ensemble_members,
        )
    else:
        # Subsequent calls: perform only a light reset
        model_wrapper_emodps.justice_instance.reset_model()

    # Reuse the JUSTICE instance from here on
    model = model_wrapper_emodps.justice_instance

    # getattr(model, "no_of_ensembles")
    no_of_ensembles = model.__getattribute__("no_of_ensembles")

    datasets = {}
    # Initialize emissions control rate
    emissions_control_rate = np.zeros((n_regions, n_timesteps, no_of_ensembles))
    constrained_emission_control_rate = np.zeros(
        (n_regions, n_timesteps, no_of_ensembles)
    )
    previous_temperature = 0
    difference = 0

    for timestep in range(n_timesteps):
        # TODO: save this constrained emission control rate
        # Constrain the emission control rate
        constrained_emission_control_rate[:, timestep, :] = (
            emission_constraint.constrain_emission_control_rate(
                emissions_control_rate[:, timestep, :],
                timestep,
                allow_fallback=False,  # Default is False
            )
        )
        model.stepwise_run(
            emission_control_rate=constrained_emission_control_rate[:, timestep, :],
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

    # Calculate the mean of ["welfare"] over the 1000 ensembles
    welfare = np.abs(datasets["welfare"])

    # Get the years above temperature threshold
    # years_above_threshold = years_above_temperature_threshold(
    #     datasets["global_temperature"], 2.0
    # )

    # Calculate the fraction of ensemble above the temperature threshold temperature, temperature_year_index, threshold
    fraction_above_threshold = fraction_of_ensemble_above_threshold(
        temperature=datasets["global_temperature"],
        temperature_year_index=temperature_year_of_interest_index,
        threshold=2.0,
    )

    # TODO: Temporarily commented out for bi-objective optimization

    # Transform the damage cost per capita to welfare loss value
    # _, _, _, welfare_loss_damage = model.welfare_function.calculate_welfare(
    #     datasets["damage_cost_per_capita"], welfare_loss=True
    # )
    # welfare_loss_damage = np.abs(welfare_loss_damage)

    # # Transform the abatement cost to welfare loss value
    # _, _, _, welfare_loss_abatement = model.welfare_function.calculate_welfare(
    #     datasets["abatement_cost_per_capita"], welfare_loss=True
    # )
    # welfare_loss_abatement = np.abs(welfare_loss_abatement)

    return (
        welfare,
        fraction_above_threshold,
        # years_above_threshold,
        # welfare_loss_damage,
        # welfare_loss_abatement,
    )  # ,


def model_wrapper(**kwargs):
    # TODO - Need to update this wrapper [Deprecated]
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


def model_wrapper_static_optimization(**kwargs):
    # TODO - Need to update this wrapper [Deprecated]
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

    emissions_control_rate = np.zeros((n_regions, n_timesteps))

    # TODO temporarily commented out
    for i in range(n_regions):
        for j in range(n_timesteps):
            emissions_control_rate[i, j] = kwargs.pop(f"emissions_control_rate {i} {j}")

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

    datasets = {}

    model.run(
        emission_control_rate=emissions_control_rate, endogenous_savings_rate=True
    )

    datasets = model.evaluate()
    # Calculate the mean of ["welfare_utilitarian"] over the 1000 ensembles
    datasets["welfare_utilitarian"] = np.mean(datasets["welfare_utilitarian"])

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
