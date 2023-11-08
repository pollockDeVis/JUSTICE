"""
This module contains the uncertainty analysis for the JUSTICE model using EMA Workbench.
"""
import numpy as np
import os

stat = "mean"  # mean, median, 5th, 95th

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
from src.util.EMA_model_wrapper import model_wrapper
from src.model_time import TimeHorizon
from src.data_loader import DataLoader

# Instantiate the DataLoader class
data_loader = DataLoader()
# Instantiate the TimeHorizon class
time_horizon = TimeHorizon(start_year=2015, end_year=2300, data_timestep=5, timestep=1)


def apply_statistical_functions(results):
    if stat == "mean":
        if len(results.shape) == 3:
            # Return mean of results
            return np.mean(results, axis=2)
        elif len(results.shape) == 2:
            # Return results
            return np.mean(results, axis=1)
        elif len(results.shape) == 1:
            return np.mean(results)
    elif stat == "median":
        if len(results.shape) == 3:
            # Return mean of results
            return np.median(results, axis=2)
        elif len(results.shape) == 2:
            # Return results
            return np.median(results, axis=1)
        elif len(results.shape) == 1:
            return np.median(results)

    elif stat == "95th":
        if len(results.shape) == 3:
            # Return mean of results
            return np.percentile(results, 95, axis=2)
        elif len(results.shape) == 2:
            # Return results
            return np.percentile(results, 95, axis=1)
        elif len(results.shape) == 1:
            return np.percentile(results, 95)

    elif stat == "5th":
        if len(results.shape) == 3:
            # Return mean of results
            return np.percentile(results, 5, axis=2)
        elif len(results.shape) == 2:
            # Return results
            return np.percentile(results, 5, axis=1)
        elif len(results.shape) == 1:
            return np.percentile(results, 5)


def get_mean_3D(results):
    # Check if results is a 3D array or a 2D array
    if len(results.shape) == 3:
        # Return mean of results
        return np.mean(results, axis=2)
    elif len(results.shape) == 2:
        # Return results
        return np.mean(results, axis=1)


def get_mean_2D(results):
    if len(results.shape) == 2:
        # Return mean of results
        return np.mean(results, axis=1)
    elif len(results.shape) == 1:
        return np.mean(results)


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
        RealParameter("inequality_aversion", 0.0, 2.0),
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
        # ArrayOutcome("net_economic_output", function=get_mean),
        # ArrayOutcome("consumption", function=get_mean),
        # ArrayOutcome("welfare_utilitarian"),  # (286, 1001) #, function=get_mean_2D
        ArrayOutcome("consumption_per_capita", function=apply_statistical_functions),
        ArrayOutcome("emissions", function=apply_statistical_functions),
        TimeSeriesOutcome(
            "global_temperature", function=apply_statistical_functions
        ),  # (286, 1001)
        ArrayOutcome("economic_damage", function=apply_statistical_functions),
        ArrayOutcome("abatement_cost", function=apply_statistical_functions),
        ArrayOutcome("disentangled_utility", function=apply_statistical_functions),
    ]

    with MultiprocessingEvaluator(model, n_processes=28) as evaluator:
        results = evaluator.perform_experiments(
            scenarios=number_of_experiments,
            reporting_frequency=100,  # policies=2,TODO temporarily commented out
        )

        if filename is None:
            file_name = (
                f"optimal_open_exploration_{number_of_experiments}_{stat}.tar.gz"
            )

        if folder is None:
            target_directory = os.path.join(os.getcwd(), "data/output", file_name)
        else:
            target_directory = os.path.join(folder, file_name)

        save_results(results, file_name=target_directory)


if __name__ == "__main__":
    perform_exploratory_analysis(number_of_experiments=10, filename=None, folder=None)
