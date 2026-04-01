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

from justice.util.regional_configuration import aggregate_by_macro_region
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
    )  # n_inputs_rbf +2  is a rule of thumb.

    centers_shape, radii_shape, weights_shape = rbf.get_shape()

    centers = np.zeros(centers_shape)
    radii = np.zeros(radii_shape)
    weights = np.zeros(weights_shape)

    for i in range(centers_shape[0]):
        centers[i] = kwargs.pop(f"center_{i}")
        radii[i] = kwargs.pop(f"radii_{i}")

    for i in range(weights_shape[0]):
        weights[i] = kwargs.pop(f"weights_{i}")

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

    # Calculate the fraction of ensemble above the temperature threshold temperature, temperature_year_index, threshold
    fraction_above_threshold = fraction_of_ensemble_above_threshold(
        temperature=datasets["global_temperature"],
        temperature_year_index=temperature_year_of_interest_index,
        threshold=2.0,
    )

    return (
        welfare,
        fraction_above_threshold,
    )


################################################################################################################################
# --- Normalization helper ---------------------------------------------------------
def _compute_inverse_range(min_value: float, max_value: float) -> float:
    span = max_value - min_value
    if span <= 0:
        raise ValueError(
            f"Invalid normalization bounds: min={min_value}, max={max_value}"
        )
    return 1.0 / span


def _extract_vector(kwargs_dict, prefix, size, macro_idx):
    """Pull a flat vector of length `size` from kwargs named `{prefix}_{macro_idx}_{i}`."""
    vector = np.empty(size, dtype=float)
    for i in range(size):
        vector[i] = kwargs_dict.pop(f"{prefix}_{macro_idx}_{i}")
    return vector


# --- MOMA Wrapper ----------------------------------------------------------------------
def model_wrapper_momadps(**kwargs):
    scenario = kwargs.pop("ssp_rcp_scenario")
    social_welfare_function_type = kwargs.pop("social_welfare_function_type")

    economy_type = Economy.from_index(kwargs.pop("economy_type"))
    damage_function_type = DamageFunction.from_index(kwargs.pop("damage_function_type"))
    abatement_type = Abatement.from_index(kwargs.pop("abatement_type"))
    stochastic_run = kwargs.pop("stochastic_run")

    n_regions = kwargs.pop("n_regions")
    n_timesteps = kwargs.pop("n_timesteps")
    emission_control_start_timestep = kwargs.pop("emission_control_start_timestep")

    n_inputs_rbf = kwargs.pop("n_inputs_rbf")  # should be 3
    n_outputs_rbf = kwargs.pop("n_outputs_rbf")
    temperature_year_of_interest_index = kwargs.pop(
        "temperature_year_of_interest_index"
    )
    climate_ensemble_members = kwargs.pop("climate_ensemble_members")

    # Normalization parameters (all provided via config)
    min_temperature = kwargs.pop("min_temperature")
    max_temperature = kwargs.pop("max_temperature")
    min_temperature_change = kwargs.pop("min_temperature_change")
    max_temperature_change = kwargs.pop("max_temperature_change")
    consumption_min = kwargs.pop("consumption_min")
    consumption_max = kwargs.pop("consumption_max")

    inv_temperature_range = _compute_inverse_range(min_temperature, max_temperature)
    inv_temperature_change_range = _compute_inverse_range(
        min_temperature_change, max_temperature_change
    )
    inv_consumption_range = _compute_inverse_range(consumption_min, consumption_max)

    # Macro-region setup
    region_to_macro = np.asarray(kwargs.pop("region_to_macro"), dtype=np.intp)
    n_macro_regions = kwargs.pop("n_macro_regions")

    macro_region_counts = np.bincount(
        region_to_macro, minlength=n_macro_regions
    ).astype(float)
    macro_region_counts = macro_region_counts[:, None]  # For broadcasting

    # RBF instantiation: one per macro region, each with its own parameter block
    rbf_template = RBF(
        n_rbfs=(n_inputs_rbf + 2), n_inputs=n_inputs_rbf, n_outputs=n_outputs_rbf
    )
    centers_shape, radii_shape, weights_shape = rbf_template.get_shape()
    centers_len, radii_len, weights_len = (
        centers_shape[0],
        radii_shape[0],
        weights_shape[0],
    )

    macro_rbfs = []
    for macro_idx in range(n_macro_regions):
        centers = _extract_vector(kwargs, "center", centers_len, macro_idx)
        radii = _extract_vector(kwargs, "radii", radii_len, macro_idx)
        weights = _extract_vector(kwargs, "weights", weights_len, macro_idx)

        rbf = RBF(
            n_rbfs=(n_inputs_rbf + 2), n_inputs=n_inputs_rbf, n_outputs=n_outputs_rbf
        )
        decision_vars = np.concatenate((centers, radii, weights))
        rbf.set_decision_vars(decision_vars)
        macro_rbfs.append(rbf)

    emission_constraint = EmissionControlConstraint(
        max_annual_growth_rate=0.04,
        emission_control_start_timestep=emission_control_start_timestep,
        min_emission_control_rate=0.01,
    )

    # --- Singleton JUSTICE handling -------------------------------------------------
    if not hasattr(model_wrapper_momadps, "justice_instance"):
        model_wrapper_momadps.justice_instance = JUSTICE(
            scenario=scenario,
            economy_type=economy_type,
            damage_function_type=damage_function_type,
            abatement_type=abatement_type,
            social_welfare_function_type=social_welfare_function_type,
            stochastic_run=stochastic_run,
            climate_ensembles=climate_ensemble_members,
        )
    else:
        model_wrapper_momadps.justice_instance.reset_model()

    model = model_wrapper_momadps.justice_instance
    no_of_ensembles = model.__getattribute__("no_of_ensembles")

    # --- Policy buffers -------------------------------------------------------------
    regional_emission_control_rate = np.zeros(
        (n_regions, n_timesteps, no_of_ensembles), dtype=float
    )
    constrained_emission_control_rate = np.zeros_like(regional_emission_control_rate)
    macro_emission_control_rate = np.zeros(
        (n_macro_regions, n_timesteps, no_of_ensembles), dtype=float
    )

    # Feedback state
    previous_temperature = np.zeros(no_of_ensembles, dtype=float)
    previous_temperature_change = np.zeros(no_of_ensembles, dtype=float)

    # Temporary buffer reused for RBF inputs
    rbf_input_buffer = np.empty((n_inputs_rbf, no_of_ensembles), dtype=float)

    # Storage for macro-level consumption history
    macro_consumption_per_capita_history = np.zeros(
        (n_macro_regions, n_timesteps, no_of_ensembles), dtype=float
    )

    population = model.economy.get_population()  # Getting population from the model
    population = aggregate_by_macro_region(
        population, region_to_macro
    )  # Aggregating to macro regions
    # --- Simulation loop ------------------------------------------------------------
    for timestep in range(n_timesteps):
        constrained_emission_control_rate[:, timestep, :] = (
            emission_constraint.constrain_emission_control_rate(
                regional_emission_control_rate[:, timestep, :],
                timestep,
                allow_fallback=False,
            )
        )

        model.stepwise_run(
            emission_control_rate=constrained_emission_control_rate[:, timestep, :],
            timestep=timestep,
            endogenous_savings_rate=True,
        )
        datasets = model.stepwise_evaluate(timestep=timestep)

        global_temperature = datasets["global_temperature"][timestep, :]

        consumption = datasets["consumption"][:, timestep, :]
        consumption = consumption * 1e3

        # Temperature and rate of change (shared inputs)
        if timestep == 0:
            temperature_change = np.zeros_like(global_temperature)
            previous_temperature = global_temperature.copy()
            previous_temperature_change = temperature_change.copy()
        elif timestep % 5 == 0:
            temperature_change = global_temperature - previous_temperature
            previous_temperature = global_temperature.copy()
            previous_temperature_change = temperature_change.copy()
        else:
            temperature_change = previous_temperature_change

        rbf_input_buffer[0, :] = np.clip(
            (global_temperature - min_temperature) * inv_temperature_range, 0.0, 1.0
        )
        rbf_input_buffer[1, :] = np.clip(
            (temperature_change - min_temperature_change)
            * inv_temperature_change_range,
            0.0,
            1.0,
        )

        # Aggregate consumption to macro level, normalize, and store full history
        aggregated_consumption = aggregate_by_macro_region(  # Use consumption here
            consumption, region_to_macro
        )

        aggregated_consumption_per_capita = (
            aggregated_consumption / population[:, timestep, :]
        )

        macro_consumption_per_capita_history[:, timestep, :] = (
            aggregated_consumption_per_capita
        )

        normalized_aggregated_consumption_per_capita = (
            np.clip(  # TODO Check this normalization
                (aggregated_consumption_per_capita - consumption_min)
                * inv_consumption_range,
                0.0,
                1.0,
            )
        )

        # RBFs observe current step signals and set emissions for next step
        if timestep < n_timesteps - 1:
            for macro_idx, rbf in enumerate(macro_rbfs):
                rbf_input_buffer[2, :] = normalized_aggregated_consumption_per_capita[
                    macro_idx, :
                ]
                macro_output = rbf.apply_rbfs(rbf_input_buffer)
                macro_emission_control_rate[macro_idx, timestep + 1, :] = macro_output

            regional_emission_control_rate[:, timestep + 1, :] = (
                macro_emission_control_rate[region_to_macro, timestep + 1, :]
            )

    datasets = model.evaluate()

    spatially_disaggregated_welfare = (
        model.welfare_function.calculate_spatially_disaggregated_welfare(
            macro_consumption_per_capita_history
        )
    )

    fraction_above_threshold = fraction_of_ensemble_above_threshold(
        temperature=datasets["global_temperature"],
        temperature_year_index=temperature_year_of_interest_index,
        threshold=2.0,
    )

    return (
        float(spatially_disaggregated_welfare[0]),
        float(spatially_disaggregated_welfare[1]),
        float(spatially_disaggregated_welfare[2]),
        float(spatially_disaggregated_welfare[3]),
        float(spatially_disaggregated_welfare[4]),
        float(fraction_above_threshold),
    )


# ---------------------------MOMA Single Agent----------------------------------------------------------------#


def model_wrapper_momadps_single_agent(**kwargs):
    """
    Variant of the MOMADPS wrapper for single-agent re-optimization.
    Only the RBF of `variable_macro_index` is free to change; all others are fixed
    to a prior (Pareto-Nash) solution supplied via fixed_centers/radii/weights.
    """
    scenario = kwargs.pop("ssp_rcp_scenario")
    social_welfare_function_type = kwargs.pop("social_welfare_function_type")

    economy_type = Economy.from_index(kwargs.pop("economy_type"))
    damage_function_type = DamageFunction.from_index(kwargs.pop("damage_function_type"))
    abatement_type = Abatement.from_index(kwargs.pop("abatement_type"))
    stochastic_run = kwargs.pop("stochastic_run")

    n_regions = kwargs.pop("n_regions")
    n_timesteps = kwargs.pop("n_timesteps")
    emission_control_start_timestep = kwargs.pop("emission_control_start_timestep")

    n_inputs_rbf = kwargs.pop("n_inputs_rbf")
    n_outputs_rbf = kwargs.pop("n_outputs_rbf")
    temperature_year_of_interest_index = kwargs.pop(
        "temperature_year_of_interest_index"
    )
    climate_ensemble_members = kwargs.pop("climate_ensemble_members")

    # Normalization parameters
    min_temperature = kwargs.pop("min_temperature")
    max_temperature = kwargs.pop("max_temperature")
    min_temperature_change = kwargs.pop("min_temperature_change")
    max_temperature_change = kwargs.pop("max_temperature_change")
    consumption_min = kwargs.pop("consumption_min")
    consumption_max = kwargs.pop("consumption_max")

    inv_temperature_range = _compute_inverse_range(min_temperature, max_temperature)
    inv_temperature_change_range = _compute_inverse_range(
        min_temperature_change, max_temperature_change
    )
    inv_consumption_range = _compute_inverse_range(consumption_min, consumption_max)

    # Macro-region setup
    region_to_macro = np.asarray(kwargs.pop("region_to_macro"), dtype=np.intp)
    n_macro_regions = kwargs.pop("n_macro_regions")

    # Which macro region is being (re)optimized
    variable_macro_index = kwargs.pop("variable_macro_index")

    # Fixed RBF parameters from the Pareto-Nash solution (constants)
    fixed_centers = np.asarray(kwargs.pop("fixed_centers"), dtype=float)
    fixed_radii = np.asarray(kwargs.pop("fixed_radii"), dtype=float)
    fixed_weights = np.asarray(kwargs.pop("fixed_weights"), dtype=float)

    macro_region_counts = np.bincount(
        region_to_macro, minlength=n_macro_regions
    ).astype(float)
    macro_region_counts = macro_region_counts[:, None]

    # RBF parameter shapes
    rbf_template = RBF(
        n_rbfs=(n_inputs_rbf + 2), n_inputs=n_inputs_rbf, n_outputs=n_outputs_rbf
    )
    centers_shape, radii_shape, weights_shape = rbf_template.get_shape()
    centers_len, radii_len, weights_len = (
        centers_shape[0],
        radii_shape[0],
        weights_shape[0],
    )

    # Build the RBF set: only one macro uses lever values; others use fixed
    macro_rbfs = []
    for macro_idx in range(n_macro_regions):
        if macro_idx == variable_macro_index:
            centers = _extract_vector(kwargs, "center", centers_len, macro_idx)
            radii = _extract_vector(kwargs, "radii", radii_len, macro_idx)
            weights = _extract_vector(kwargs, "weights", weights_len, macro_idx)
        else:
            centers = fixed_centers[macro_idx]
            radii = fixed_radii[macro_idx]
            weights = fixed_weights[macro_idx]

        rbf = RBF(
            n_rbfs=(n_inputs_rbf + 2), n_inputs=n_inputs_rbf, n_outputs=n_outputs_rbf
        )
        decision_vars = np.concatenate((centers, radii, weights))
        rbf.set_decision_vars(decision_vars)
        macro_rbfs.append(rbf)

    emission_constraint = EmissionControlConstraint(
        max_annual_growth_rate=0.04,
        emission_control_start_timestep=emission_control_start_timestep,
        min_emission_control_rate=0.01,
    )

    # JUSTICE singleton handling
    if not hasattr(model_wrapper_momadps_single_agent, "justice_instance"):
        model_wrapper_momadps_single_agent.justice_instance = JUSTICE(
            scenario=scenario,
            economy_type=economy_type,
            damage_function_type=damage_function_type,
            abatement_type=abatement_type,
            social_welfare_function_type=social_welfare_function_type,
            stochastic_run=stochastic_run,
            climate_ensembles=climate_ensemble_members,
        )
    else:
        model_wrapper_momadps_single_agent.justice_instance.reset_model()

    model = model_wrapper_momadps_single_agent.justice_instance
    no_of_ensembles = model.no_of_ensembles

    # Buffers
    regional_emission_control_rate = np.zeros(
        (n_regions, n_timesteps, no_of_ensembles), dtype=float
    )
    constrained_emission_control_rate = np.zeros_like(regional_emission_control_rate)
    macro_emission_control_rate = np.zeros(
        (n_macro_regions, n_timesteps, no_of_ensembles), dtype=float
    )

    previous_temperature = np.zeros(no_of_ensembles, dtype=float)
    previous_temperature_change = np.zeros(no_of_ensembles, dtype=float)

    rbf_input_buffer = np.empty((n_inputs_rbf, no_of_ensembles), dtype=float)
    macro_consumption_per_capita_history = np.zeros(
        (n_macro_regions, n_timesteps, no_of_ensembles), dtype=float
    )

    population = model.economy.get_population()
    population = aggregate_by_macro_region(population, region_to_macro)

    # Simulation loop
    for timestep in range(n_timesteps):
        constrained_emission_control_rate[:, timestep, :] = (
            emission_constraint.constrain_emission_control_rate(
                regional_emission_control_rate[:, timestep, :],
                timestep,
                allow_fallback=False,
            )
        )

        model.stepwise_run(
            emission_control_rate=constrained_emission_control_rate[:, timestep, :],
            timestep=timestep,
            endogenous_savings_rate=True,
        )
        datasets = model.stepwise_evaluate(timestep=timestep)

        global_temperature = datasets["global_temperature"][timestep, :]
        consumption = datasets["consumption"][:, timestep, :] * 1e3

        if timestep == 0:
            temperature_change = np.zeros_like(global_temperature)
            previous_temperature = global_temperature.copy()
            previous_temperature_change = temperature_change.copy()
        elif timestep % 5 == 0:
            temperature_change = global_temperature - previous_temperature
            previous_temperature = global_temperature.copy()
            previous_temperature_change = temperature_change.copy()
        else:
            temperature_change = previous_temperature_change

        rbf_input_buffer[0, :] = np.clip(
            (global_temperature - min_temperature) * inv_temperature_range, 0.0, 1.0
        )
        rbf_input_buffer[1, :] = np.clip(
            (temperature_change - min_temperature_change)
            * inv_temperature_change_range,
            0.0,
            1.0,
        )

        aggregated_consumption = aggregate_by_macro_region(consumption, region_to_macro)
        aggregated_consumption_per_capita = (
            aggregated_consumption / population[:, timestep, :]
        )
        macro_consumption_per_capita_history[:, timestep, :] = (
            aggregated_consumption_per_capita
        )
        normalized_consumption = np.clip(
            (aggregated_consumption_per_capita - consumption_min)
            * inv_consumption_range,
            0.0,
            1.0,
        )

        if timestep < n_timesteps - 1:
            for macro_idx, rbf in enumerate(macro_rbfs):
                rbf_input_buffer[2, :] = normalized_consumption[macro_idx, :]
                macro_output = rbf.apply_rbfs(rbf_input_buffer)
                macro_emission_control_rate[macro_idx, timestep + 1, :] = macro_output

            regional_emission_control_rate[:, timestep + 1, :] = (
                macro_emission_control_rate[region_to_macro, timestep + 1, :]
            )

    datasets = model.evaluate()

    spatial_welfare = model.welfare_function.calculate_spatially_disaggregated_welfare(
        macro_consumption_per_capita_history
    )
    agent_welfare = float(spatial_welfare[variable_macro_index])

    fraction_above_threshold = fraction_of_ensemble_above_threshold(
        temperature=datasets["global_temperature"],
        temperature_year_index=temperature_year_of_interest_index,
        threshold=2.0,
    )

    return (
        agent_welfare,
        float(fraction_above_threshold),
    )


##################################################################################################################################


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
