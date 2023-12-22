"""
This module contains the uncertainty analysis for the JUSTICE model using EMA Workbench.
"""
import functools

import numpy as np
import os

# stat = "mean"  # mean, median, 5th, 95th

# EMA
from ema_workbench import (
    Model,
    RealParameter,
    ArrayOutcome,
    TimeSeriesOutcome,
    CategoricalParameter,
    ema_logging,
    MultiprocessingEvaluator,
    Constant,
    SequentialEvaluator,
)
from ema_workbench.util.utilities import save_results, load_results

ema_logging.log_to_stderr(ema_logging.INFO)

# JUSTICE
from src.enumerations import Scenario
from src.util.EMA_model_wrapper import model_wrapper, model_wrapper_emodps
from src.model_time import TimeHorizon
from src.data_loader import DataLoader

# Instantiate the DataLoader class
data_loader = DataLoader()
# Instantiate the TimeHorizon class
time_horizon = TimeHorizon(start_year=2015, end_year=2300, data_timestep=5, timestep=1)


def run_optimization_adaptive(
    n_rbfs=4, n_inputs=2, nfe=5000, filename=None, folder=None
):
    model = Model("JUSTICE", function=model_wrapper_emodps)

    # Define constants, uncertainties and levers
    model.constants = [
        Constant("n_regions", len(data_loader.REGION_LIST)),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
        Constant("n_rbfs", n_rbfs),
        Constant("n_inputs_rbf", n_inputs),
        Constant("n_outputs_rbf", len(data_loader.REGION_LIST)),
    ]

    # Speicify uncertainties
    model.uncertainties = [
        CategoricalParameter(
            "ssp_rcp_scenario", (0, 1, 2, 3, 4, 5, 6, 7)
        ),  # 8 SSP-RCP scenario combinations
        # TODO temporarily commented out
        # RealParameter("elasticity_of_marginal_utility_of_consumption", 0.0, 2.0),
        # RealParameter(
        #     "pure_rate_of_social_time_preference", 0.0001, 0.020
        # ),  # 0.1 to 3% in RICE50 gazzotti2
        # RealParameter("inequality_aversion", 0.0, 2.0),  # 0.2 -2.5
    ]

    # Set the model levers, which are the RBF parameters
    # These are the formula to calculate the number of centers, radii and weights

    centers_shape = (
        n_rbfs * n_inputs
    )  # centers = n_rbfs x n_inputs # radii = n_rbfs x n_inputs
    weights_shape = (
        len(data_loader.REGION_LIST) * n_rbfs
    )  # weights = n_outputs x n_rbfs

    centers_levers = []
    radii_levers = []
    weights_levers = []

    for i in range(centers_shape):
        centers_levers.append(RealParameter(f"center {i}", -1.0, 1.0))
        radii_levers.append(RealParameter(f"radii {i}", 0.0, 1.0))

    for i in range(weights_shape):
        weights_levers.append(RealParameter(f"weights {i}", 0.0, 1.0))

    # Set the model levers
    model.levers = centers_levers + radii_levers + weights_levers

    # Reference Scenario?
    # Hyperparameters -deap
    # Epsilon
    # Outcomes?


def perform_exploratory_analysis(number_of_experiments=10, filename=None, folder=None):
    # Instantiate the model

    model = Model("JUSTICE", function=model_wrapper)
    model.constants = [
        Constant("n_regions", len(data_loader.REGION_LIST)),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
    ]

    # Speicify uncertainties
    model.uncertainties = [
        CategoricalParameter(
            "ssp_rcp_scenario", (0, 1, 2, 3, 4, 5, 6, 7)
        ),  # 8 SSP-RCP scenario combinations
        RealParameter("elasticity_of_marginal_utility_of_consumption", 0.0, 2.0),
        RealParameter(
            "pure_rate_of_social_time_preference", 0.0001, 0.020
        ),  # 0.1 to 3% in RICE50 gazzotti2
        RealParameter("inequality_aversion", 0.0, 2.0),  # 0.2 -2.5
    ]

    # Set model levers - has to be 2D array of shape (57, 286) 57 regions and 286 timesteps

    # TODO temporarily commented out
    # sr_levers = []
    # ecr_levers = []
    # for i in range(len(data_loader.REGION_LIST)):
    #     for j in range(len(time_horizon.model_time_horizon)):
    #         sr_levers.append(RealParameter(f"savings_rate {i} {j}", 0.05, 0.5))
    #         ecr_levers.append(
    #             RealParameter(f"emissions_control_rate {i} {j}", 0.00, 1.0)
    #         )

    # model.levers = sr_levers + ecr_levers

    # Specify outcomes #All outcomes have shape (57, 286, 1001) except global_temperature which has shape (286, 1001)
    model.outcomes = [
        ArrayOutcome(
            "mean_net_economic_output",
            function=functools.partial(np.mean, axis=2),
            variable_name="net_economic_output",
        ),
        ArrayOutcome(
            "5p_net_economic_output",
            function=functools.partial(np.percentile, q=5, axis=2),
            variable_name="net_economic_output",
        ),
        ArrayOutcome(
            "95p_net_economic_output",
            function=functools.partial(np.percentile, q=95, axis=2),
            variable_name="net_economic_output",
        ),
        ArrayOutcome(
            "mean_consumption_per_capita",
            function=functools.partial(np.mean, axis=2),
            variable_name="consumption_per_capita",
        ),
        ArrayOutcome(
            "5p_consumption_per_capita",
            function=functools.partial(np.percentile, q=5, axis=2),
            variable_name="consumption_per_capita",
        ),
        ArrayOutcome(
            "95p_consumption_per_capita",
            function=functools.partial(np.percentile, q=95, axis=2),
            variable_name="consumption_per_capita",
        ),
        ArrayOutcome(
            "mean_emissions",
            function=functools.partial(np.mean, axis=2),
            variable_name="emissions",
        ),
        ArrayOutcome(
            "5p_emissions",
            function=functools.partial(np.percentile, q=5, axis=2),
            variable_name="emissions",
        ),
        ArrayOutcome(
            "95p_emissions",
            function=functools.partial(np.percentile, q=95, axis=2),
            variable_name="emissions",
        ),
        ArrayOutcome(
            "mean_economic_damage",
            function=functools.partial(np.mean, axis=2),
            variable_name="economic_damage",
        ),
        ArrayOutcome(
            "5p_economic_damage",
            function=functools.partial(np.percentile, q=5, axis=2),
            variable_name="economic_damage",
        ),
        ArrayOutcome(
            "95p_economic_damage",
            function=functools.partial(np.percentile, q=95, axis=2),
            variable_name="economic_damage",
        ),
        ArrayOutcome(
            "mean_abatement_cost",
            function=functools.partial(np.mean, axis=2),
            variable_name="abatement_cost",
        ),
        ArrayOutcome(
            "5p_abatement_cost",
            function=functools.partial(np.percentile, q=5, axis=2),
            variable_name="abatement_cost",
        ),
        ArrayOutcome(
            "95p_abatement_cost",
            function=functools.partial(np.percentile, q=95, axis=2),
            variable_name="abatement_cost",
        ),
        ArrayOutcome(
            "mean_disentangled_utility",
            function=functools.partial(np.mean, axis=2),
            variable_name="disentangled_utility",
        ),
        ArrayOutcome(
            "5p_disentangled_utility",
            function=functools.partial(np.percentile, q=5, axis=2),
            variable_name="disentangled_utility",
        ),
        ArrayOutcome(
            "95p_disentangled_utility",
            function=functools.partial(np.percentile, q=95, axis=2),
            variable_name="disentangled_utility",
        ),
        ArrayOutcome(
            "mean_consumption",
            function=functools.partial(np.mean, axis=2),
            variable_name="consumption",
        ),
        ArrayOutcome(
            "5p_consumption",
            function=functools.partial(np.percentile, q=5, axis=2),
            variable_name="consumption",
        ),
        ArrayOutcome(
            "95p_consumption",
            function=functools.partial(np.percentile, q=95, axis=2),
            variable_name="consumption",
        ),
        # ArrayOutcome("consumption", function=functools.partial(np.mean, axis=2)),
        # ArrayOutcome("welfare_utilitarian"),  # (286, 1001) #, function=get_mean_2D
        # ArrayOutcome("consumption_per_capita", function=apply_statistical_functions),
        # ArrayOutcome("emissions", function=apply_statistical_functions),
        ArrayOutcome(
            "mean_global_temperature",
            function=functools.partial(np.mean, axis=1),
            variable_name="global_temperature",
        ),  # (286, 1001)
        ArrayOutcome(
            "5p_global_temperature",
            function=functools.partial(np.percentile, q=5, axis=1),
            variable_name="global_temperature",
        ),  # (286, 1001)
        ArrayOutcome(
            "95p_global_temperature",
            function=functools.partial(np.percentile, q=95, axis=1),
            variable_name="global_temperature",
        ),
        ArrayOutcome(
            "mean_welfare_utilitarian",
            function=functools.partial(np.mean, axis=1),
            variable_name="welfare_utilitarian",
        ),  # (286, 1001)
        ArrayOutcome(
            "5p_welfare_utilitarian",
            function=functools.partial(np.percentile, q=5, axis=1),
            variable_name="welfare_utilitarian",
        ),  # (286, 1001)
        ArrayOutcome(
            "95p_welfare_utilitarian",
            function=functools.partial(np.percentile, q=95, axis=1),
            variable_name="welfare_utilitarian",
        ),  # (286, 1001)
        # (286, 1001)
        # ArrayOutcome("economic_damage", function=apply_statistical_functions),
        # ArrayOutcome("abatement_cost", function=apply_statistical_functions),
        # ArrayOutcome("disentangled_utility", function=apply_statistical_functions),
    ]

    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(
            scenarios=number_of_experiments,
            reporting_frequency=100,  # policies=2,TODO temporarily commented out
        )

        if filename is None:
            file_name = f"optimal_open_exploration_{number_of_experiments}.tar.gz"

        if folder is None:
            target_directory = os.path.join(os.getcwd(), "data/output", file_name)
        else:
            target_directory = os.path.join(folder, file_name)

        save_results(results, file_name=target_directory)


if __name__ == "__main__":
    perform_exploratory_analysis(number_of_experiments=10, filename=None, folder=None)
