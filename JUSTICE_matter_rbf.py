import pandas as pd
import h5py
from emodps.rbf import RBF
import numpy as np

from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
from src.util.emission_control_constraint import EmissionControlConstraint

#Define time horizon
time_horizon = TimeHorizon(
        start_year=2015, end_year=2300, data_timestep=5, timestep=1
    )

# Define Recycling rate input
with h5py.File(
    "data/input/recycling/recycling_rate_linear_proyection.hdf5",
      'r'
    ) as f:
    recycling_rate_linear_projections = f['recycling_rate_linear_proyection'][:]

#Running the model with the optimized emissions control rate
def JUSTICE_stepwise_run(
    scenarios=0,
    elasticity_of_marginal_utility_of_consumption=1.45,
    pure_rate_of_social_time_preference=0.015,
    inequality_aversion=0.5,
    path_to_rbf_weights="data/input/optimized_rbf/100049.csv",
    path_to_output="data/output/",
    saving=False,
    output_file_name=None,
    rbf_policy_index=1033,
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
        matter=EconomySubModules.MATTER,
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
            recycling_rate=recycling_rate_linear_projections
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


def setup_RBF_for_emission_control(
    region_list,
    rbf_policy_index,
    n_inputs_rbf,
    path_to_rbf_weights,
):

    # Read the csv file
    rbf_decision_vars = pd.read_csv(path_to_rbf_weights)

    # select row
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
    #datasets = JUSTICE_run(scenarios=0)
    datasets = JUSTICE_stepwise_run(scenarios=0)
    # Print the keys of the datasets
    print(datasets.keys())
