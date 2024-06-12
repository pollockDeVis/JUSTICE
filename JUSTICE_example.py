import pandas as pd
from emodps.rbf import RBF
import numpy as np

from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
from src.util.emission_control_constraint import EmissionControlConstraint


def get_linear_emission_control():
    """
    Linear emission control problem
    """
    data_loader = DataLoader()

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(
        start_year=2015, end_year=2300, data_timestep=5, timestep=1
    )

    # emissions control rate borrowed from emissions module

    # Variables to be changed/deleted later
    miu_initial = 0.0
    min_miu = 0.0  # 0.2  # 0.0 #1.0
    min_miu_year = 2055  # 9-original #8 in this model  # 2060
    max_miu = 1.0  # 1.0  # 1.2
    max_miu_year = 2200  # 38-original #37 in this model #2205

    t_min_miu = time_horizon.year_to_timestep(min_miu_year, timestep=1)
    t_max_miu = time_horizon.year_to_timestep(max_miu_year, timestep=1)

    # Initialize emissions control rate
    emissions_control_rate = np.zeros(
        (len(data_loader.REGION_LIST), len(time_horizon.model_time_horizon))
    )

    for t in range(len(time_horizon.model_time_horizon)):
        if t < t_min_miu:  # Before time of transition
            emissions_control_rate[:, t] = min_miu
        elif t <= t_max_miu:  # Transition
            # During the transition
            emissions_control_rate[:, t] = min_miu + (max_miu - min_miu) * (
                t - t_min_miu
            ) / (t_max_miu - t_min_miu)
        else:  # After the transition
            emissions_control_rate[:, t] = max_miu
    return emissions_control_rate


def JUSTICE_run(scenarios=0):
    """
    Run the JUSTICE model for a given scenario
    """
    model = JUSTICE(
        scenario=scenarios,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=WelfareFunction.UTILITARIAN,
        # climate_ensembles=570,
        # Declaring for endogenous fixed savings rate
        elasticity_of_marginal_utility_of_consumption=1.45,
        pure_rate_of_social_time_preference=0.015,
        inequality_aversion=0.5,
    )

    # Get example emissions control rate
    emissions_control_rate = get_linear_emission_control()

    # Run the model
    model.run(
        emission_control_rate=emissions_control_rate, endogenous_savings_rate=True
    )

    # Get the results
    datasets = model.evaluate()

    return datasets


def JUSTICE_stepwise_run(
    scenarios=0,
    elasticity_of_marginal_utility_of_consumption=1.45,
    pure_rate_of_social_time_preference=0.015,
    inequality_aversion=0.5,
    path_to_rbf_weights="data/optimized_rbf_weights/100k_Util_4Obj_JUSTICE_dps_archive_1-4-24/100027.csv",
    path_to_output="data/output/",
    saving=False,
    output_file_name=None,
    rbf_policy_index=6809,
    n_inputs_rbf=2,
    max_annual_growth_rate=0.04,
    emission_control_start_timestep=10,
    min_emission_control_rate=0.01,
    allow_emission_fallback=False,  # Default is False
    endogenous_savings_rate=True,
    max_temperature=16.0,
    min_temperature=0.0,
    max_difference=2.0,
    min_difference=0.0,
):
    """
    Run the JUSTICE model for a given scenario
    """
    model = JUSTICE(
        scenario=scenarios,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=WelfareFunction.UTILITARIAN,
        elasticity_of_marginal_utility_of_consumption=elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference=pure_rate_of_social_time_preference,
        inequality_aversion=inequality_aversion,
    )

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

    previous_temperature = 0
    difference = 0
    max_temperature = max_temperature
    min_temperature = min_temperature
    max_difference = max_difference
    min_difference = min_difference

    for timestep in range(n_timesteps):

        # Constrain the emission control rate
        constrained_emission_control_rate[:, timestep, :] = (
            emission_constraint.constrain_emission_control_rate(
                emissions_control_rate[:, timestep, :],
                timestep,
                allow_fallback=allow_emission_fallback,
            )
        )

        model.stepwise_run(
            emission_control_rate=constrained_emission_control_rate[:, timestep, :],
            timestep=timestep,
            endogenous_savings_rate=endogenous_savings_rate,
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
    datasets["constrained_emission_control_rate"] = constrained_emission_control_rate

    # Save the datasets
    if saving:
        if output_file_name is not None:
            np.save(path_to_output + output_file_name + rbf_policy_index, datasets)
    # np.save(
    #     "data/output/optimized_emissions_control_rate.npy",
    #     constrained_emission_control_rate,
    # )

    return datasets


# TODO: Under Construction - Not implemented yet
def get_scaled_temperature_difference(
    timestep,
    temperature,
    previous_temperature,
    difference,
    min_temperature,
    max_temperature,
    min_difference,
    max_difference,
):
    """
    Get the scaled temperature and difference
    """
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

    return scaled_temperature, scaled_difference


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


if __name__ == "__main__":
    datasets = JUSTICE_run(scenarios=0)
    # datasets = JUSTICE_stepwise_run(scenarios=2)
    # Print the keys of the datasets
    print(datasets.keys())
