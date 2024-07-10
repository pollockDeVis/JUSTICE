"""
This module wraps the model for uncertainty analysis experiments using EMA.
"""

# TODO: Move this to solvers. Create a clean solving interface for JUSTICE model using EMA Workbench

import numpy as np
import pandas as pd
from src.model import JUSTICE
from src.util.enumerations import *
from solvers.emodps.rbf import RBF
from src.objectives.objective_functions import (
    years_above_temperature_threshold,
)

from src.util.emission_control_constraint import EmissionControlConstraint

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

    # Loading the constants
    n_regions = kwargs.pop("n_regions")
    n_timesteps = kwargs.pop("n_timesteps")
    emission_control_start_timestep = kwargs.pop("emission_control_start_timestep")

    n_inputs_rbf = kwargs.pop("n_inputs_rbf")
    n_outputs_rbf = kwargs.pop("n_outputs_rbf")

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

    model = JUSTICE(
        scenario=scenario,
        economy_type=economy_type,
        damage_function_type=damage_function_type,
        abatement_type=abatement_type,
        social_welfare_function_type=social_welfare_function_type,
    )

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
    years_above_threshold = years_above_temperature_threshold(
        datasets["global_temperature"], 2.0
    )

    # Transform the damage cost per capita to welfare loss value
    _, _, welfare_loss_damage = model.welfare_function.calculate_welfare(
        datasets["damage_cost_per_capita"], welfare_loss=True
    )
    welfare_loss_damage = np.abs(welfare_loss_damage)
    # Transform the abatement cost to welfare loss value
    _, _, welfare_loss_abatement = model.welfare_function.calculate_welfare(
        datasets["abatement_cost_per_capita"], welfare_loss=True
    )
    welfare_loss_abatement = np.abs(welfare_loss_abatement)

    return welfare, years_above_threshold, welfare_loss_damage, welfare_loss_abatement


def model_wrapper(**kwargs):

    scenario = kwargs.pop("ssp_rcp_scenario")
    damage_share_ratio_tfp = kwargs.pop("damage_share_ratio_tfp")
    # economy_type = (Economy.NEOCLASSICAL,)
    # damage_function_type = (DamageFunction.KALKUHL,)
    # abatement_type = (Abatement.ENERDATA,)
    # welfare_function = (WelfareFunction.UTILITARIAN,)

    # Initialize the model
    model = JUSTICE(
        scenario=scenario,
        # economy_type=economy_type,
        # damage_function_type=damage_function_type,
        # abatement_type=abatement_type,
        # social_welfare_function=welfare_function,
        enable_damage_function=True,
        enable_abatement=True,
        economy_endogenous_growth=False,
        damage_share_ratio_tfp=damage_share_ratio_tfp,
    )

    # TODO: Harcoded values for now
    rbf_policy_index = 32
    n_inputs_rbf = 2
    path_to_rbf_weights = "data/optimized_rbf_weights/150k/UTIL/150373.csv"
    max_annual_growth_rate = 0.04
    emission_control_start_timestep = 10
    min_emission_control_rate = 0.01
    allow_emission_fallback = False
    previous_temperature = 0
    difference = 0
    max_temperature = 16.0
    min_temperature = 0.0
    max_difference = 2.0
    min_difference = 0.0
    endogenous_savings_rate = True
    enable_mitigation = False

    time_horizon = model.__getattribute__("time_horizon")
    data_loader = model.__getattribute__("data_loader")
    no_of_ensembles = model.__getattribute__("no_of_ensembles")
    n_regions = len(data_loader.REGION_LIST)
    n_timesteps = len(time_horizon.model_time_horizon)

    # Setting up the RBF. Note: this depends on the setup of the optimization run
    rbf = setup_RBF_for_emission_control(
        region_list=data_loader.REGION_LIST,
        rbf_policy_index=rbf_policy_index,
        n_inputs_rbf=n_inputs_rbf,
        path_to_rbf_weights=path_to_rbf_weights,
    )
    emission_constraint = EmissionControlConstraint(
        max_annual_growth_rate=max_annual_growth_rate,
        emission_control_start_timestep=emission_control_start_timestep,
        min_emission_control_rate=min_emission_control_rate,
    )

    # Initialize datasets to store the results
    datasets = {}

    # Initialize emissions control rate
    emissions_control_rate = np.zeros((n_regions, n_timesteps, no_of_ensembles))
    constrained_emission_control_rate = np.zeros(
        (n_regions, n_timesteps, no_of_ensembles)
    )

    for timestep in range(n_timesteps):

        if enable_mitigation:
            # Constrain the emission control rate
            constrained_emission_control_rate[:, timestep, :] = (
                emission_constraint.constrain_emission_control_rate(
                    emissions_control_rate[:, timestep, :],
                    timestep,
                    allow_fallback=allow_emission_fallback,
                )
            )
        else:
            constrained_emission_control_rate[:, timestep, :] = emissions_control_rate[
                :, timestep, :
            ]

        model.stepwise_run(
            emission_control_rate=constrained_emission_control_rate[:, timestep, :],
            timestep=timestep,
            endogenous_savings_rate=endogenous_savings_rate,
        )
        datasets = model.stepwise_evaluate(timestep=timestep)

        if enable_mitigation:
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
    datasets["constrained_emission_control_rate"] = constrained_emission_control_rate

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


def setup_RBF_for_emission_control(
    region_list,
    rbf_policy_index,
    n_inputs_rbf,
    path_to_rbf_weights,
):

    # Read the csv file
    rbf_decision_vars = pd.read_csv(path_to_rbf_weights)

    # select 6810 row
    rbf_decision_vars = rbf_decision_vars.iloc[rbf_policy_index, :]

    # Read the columns starting with name 'center'
    center_columns = rbf_decision_vars.filter(regex="center")

    # Read the columns starting with name 'radii'
    radii_columns = rbf_decision_vars.filter(regex="radii")

    # Read the columns starting with name 'weights'
    weights_columns = rbf_decision_vars.filter(regex="weights")

    # Coverting the center columns to a numpy array
    center_columns = center_columns.to_numpy()

    # Coverting the radii columns to a numpy array
    radii_columns = radii_columns.to_numpy()

    # Coverting the weights columns to a numpy array
    weights_columns = weights_columns.to_numpy()

    # centers = n_rbfs x n_inputs # radii = n_rbfs x n_inputs
    # weights = n_outputs x n_rbfs

    n_outputs_rbf = len(region_list)

    rbf = RBF(n_rbfs=(n_inputs_rbf + 2), n_inputs=n_inputs_rbf, n_outputs=n_outputs_rbf)

    # Populating the decision variables
    centers_flat = center_columns.flatten()
    radii_flat = radii_columns.flatten()
    weights_flat = weights_columns.flatten()

    decision_vars = np.concatenate((centers_flat, radii_flat, weights_flat))

    rbf.set_decision_vars(decision_vars)

    return rbf


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
