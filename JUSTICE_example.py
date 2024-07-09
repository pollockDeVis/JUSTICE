import pandas as pd
from solvers.emodps.rbf import RBF
import numpy as np
import h5py

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


def JUSTICE_run(
    scenarios=2,
    climate_ensembles=None,
    social_welfare_function=None,
    enable_damage_function=True,
    enable_abatement=True,
    economy_endogenous_growth=False,
):
    """
    Run the JUSTICE model for a given scenario

    @param scenarios: Scenario to run the model
    @param climate_ensembles: Climate ensembles. Default is None. Select an index from 0 to 1000 ensembles. Only one ensemble is selected.

    """
    model = JUSTICE(
        scenario=scenarios,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=social_welfare_function,  # WelfareFunction.UTILITARIAN,
        climate_ensembles=climate_ensembles,
        enable_damage_function=enable_damage_function,
        enable_abatement=enable_abatement,
        economy_endogenous_growth=economy_endogenous_growth,
    )

    # Get example emissions control rate
    emissions_control_rate = get_linear_emission_control()

    # Run the model
    model.run(
        emission_control_rate=emissions_control_rate, endogenous_savings_rate=True
    )

    # Get the results
    datasets = model.evaluate()

    # Temp #TODO: Remove this
    _, _, damages = model.welfare_function.calculate_welfare(
        datasets["damage_cost_per_capita"], welfare_loss=True
    )

    print("Wloss Damage: ", damages)
    _, _, abatement = model.welfare_function.calculate_welfare(
        datasets["abatement_cost_per_capita"], welfare_loss=True
    )
    print("Wloss Abatement: ", abatement)

    return datasets


def JUSTICE_stepwise_run(
    scenarios=0,
    social_welfare_function=WelfareFunction.UTILITARIAN,
    path_to_rbf_weights=None,
    path_to_output="data/output/",
    saving=False,
    output_file_name=None,
    rbf_policy_index=500,
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
    enable_damage_function=True,
    enable_abatement=True,
    economy_endogenous_growth=False,
):
    """
    Run the JUSTICE model for a given scenario

    @param scenarios: Scenario to run the model
    @param social_welfare_function: Social welfare function. Default is UTILITARIAN
    @param path_to_rbf_weights: Path to the RBF weights
    @param path_to_output: Path to save the output
    @param saving: Flag to save the output
    @param output_file_name: Output file name
    @param rbf_policy_index: RBF policy index - the index of the policy to be used inside the csv archive
    @param n_inputs_rbf: Number of inputs for the RBF
    @param max_annual_growth_rate: Maximum annual growth rate of emission control rate. Default is 0.04 or 4%
    @param emission_control_start_timestep: Emission control start timestep. Default is 10, which is 2025
    @param min_emission_control_rate: Minimum emission control rate. Default is 0.01 or 1%
    @param allow_emission_fallback: Flag to allow emission fallback - that is going back on Mitigation. Default is False
    @param endogenous_savings_rate: Flag to use endogenous savings rate. Default is True
    @param max_temperature: Maximum future temperature in 2300. Default is 16.0 - Needed for Min Max Scaling
    @param min_temperature: Minimum future temperature in 2300. Default is 0.0 - Needed for Min Max Scaling
    @param max_difference: Maximum difference in temperature. Default is 2.0 - Needed for Min Max Scaling
    @param min_difference: Minimum difference in temperature. Default is 0.0 - Needed for Min Max Scaling
    """

    # Assert if the path to the RBF weights is provided
    assert path_to_rbf_weights is not None, "Path to RBF weights is not provided"

    # Initialize the model
    model = JUSTICE(
        scenario=scenarios,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=social_welfare_function,
        enable_damage_function=enable_damage_function,
        enable_abatement=enable_abatement,
        economy_endogenous_growth=economy_endogenous_growth,
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

    datasets = JUSTICE_run(
        scenarios=2,
        social_welfare_function=WelfareFunction.UTILITARIAN,
        enable_damage_function=True,
        enable_abatement=True,
        economy_endogenous_growth=True,
    )

    # datasets = JUSTICE_stepwise_run(
    #     scenarios=2,
    #     social_welfare_function=WelfareFunction.UTILITARIAN,
    #     rbf_policy_index=32,
    #     path_to_rbf_weights="data/optimized_rbf_weights/150k/UTIL/150373.csv",
    #     enable_damage_function=True,
    #     enable_abatement=True,
    #     economy_endogenous_growth=True,
    # )
    # # Print the keys of the datasets
    # print(datasets.keys())

    # # Save the datasets as hdf5 file
    # with h5py.File("data/reevaluation/justice_output_util_150_endo.h5", "w") as f:
    #     for key, value in datasets.items():
    #         f.create_dataset(key, data=value)
