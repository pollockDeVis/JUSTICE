"""
This module contains the uncertainty analysis for the JUSTICE model using EMA Workbench.
"""
import numpy as np
import os

# EMA
from ema_workbench import (
    Model,
    RealParameter,
    ArrayOutcome,
    CategoricalParameter,
    Policy,
    ema_logging,
    MultiprocessingEvaluator,
    SequentialEvaluator,
    Constant
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


def get_mean_median_5th_95th(results):
    mean_array = np.mean(results, axis=2)
    median_array = np.median(results, axis=2)
    fifth_percentile_array = np.percentile(results, 5, axis=2)
    ninety_fifth_percentile_array = np.percentile(results, 95, axis=2)

    # Return list of arrays
    return [
        fifth_percentile_array,
        mean_array,
        median_array,
        ninety_fifth_percentile_array,
    ]


def perform_exploratory_analysis(number_of_experiments=10, filename=None, folder=None):
    # scenario,
    # savings_rate,
    # emissions_control_rate,
    # elasticity_of_marginal_utility_of_consumption,
    # pure_rate_of_social_time_preference,
    # inequality_aversion,

    # Instantiate the model
    model = Model("JUSTICE", function=model_wrapper)
    model.constants = [Constant("n_regions", len(data_loader.REGION_LIST)),
                       Constant("n_timesteps", len(time_horizon.model_time_horizon))]

    # Speicify uncertainties
    model.uncertainties = [
        CategoricalParameter(
            "ssp_rcp_scenario", (0, 1, 2, 3, 4, 5, 6, 7)
        ),  # 8 SSP-RCP scenario combinations
        RealParameter("elasticity_of_marginal_utility_of_consumption", 0.0, 2.0),
        RealParameter("pure_rate_of_social_time_preference", 0.0, 0.020),
        RealParameter("inequality_aversion", 0.0, 2.0),
    ]

    # Set model levers - has to be 2D array of shape (57, 286) 57 regions and 286 timesteps
    sr_levers = []
    ecr_levers = []
    for i in range(len(data_loader.REGION_LIST)):
        for j in range(len(time_horizon.model_time_horizon)):
            sr_levers.append(RealParameter(f"savings_rate {i} {j}", 0.05, 0.5))
            ecr_levers.append(RealParameter(f"emissions_control_rate {i} {j}", 0.00, 1.0))

    model.levers = sr_levers + ecr_levers

    # Specify outcomes
    model.outcomes = [
        ArrayOutcome("net_economic_output"),
        ArrayOutcome("consumption"),
        ArrayOutcome("consumption_per_capita"),
        ArrayOutcome("emissions"),
        ArrayOutcome("global_temperature"),
        ArrayOutcome("economic_damage"),
        ArrayOutcome("abatement_cost"),
        ArrayOutcome("disentangled_utility"),
    ]

    with SequentialEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(
            number_of_experiments, policies=2, reporting_frequency=100
        )

        if filename != None:
            file_name = f"results_open_exploration_{number_of_experiments}"

        if folder is None:
            target_directory = os.path.join(os.getcwd(), "data/output", file_name)

        save_results(results, file_name=target_directory)


if __name__ == "__main__":
    perform_exploratory_analysis(number_of_experiments=10, filename=None, folder=None)
