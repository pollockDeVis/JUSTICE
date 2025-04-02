import pandas as pd
from solvers.emodps.rbf import RBF
import numpy as np

from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
from src.util.emission_control_constraint import EmissionControlConstraint


from src.welfare.social_welfare_function import SocialWelfareFunction
from config.default_parameters import SocialWelfareDefaults
from src.util.enumerations import get_economic_scenario


# This example can be used to profile JUSTICE model
# Run python -m cProfile -o profile_output.prof JUSTICE_example.py
# Visualize with python -m snakeviz profile_output.prof


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
    min_miu_year = 2060  # 9-original #8 in this model  # 2060
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


def JUSTICE_run(scenarios=2, climate_ensembles=None, social_welfare_function=None):
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
    )

    time_horizon = model.__getattribute__("time_horizon")
    data_loader = model.__getattribute__("data_loader")
    no_of_ensembles = model.__getattribute__("no_of_ensembles")
    n_regions = len(data_loader.REGION_LIST)
    n_timesteps = len(time_horizon.model_time_horizon)
    population = model.economy.get_population()

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

    # Call the function within the JUSTICE_stepwise_run method #NOTE: This is optional for data analysis
    # baseline_emissions = calculate_baseline_emissions(model, datasets, scenarios)

    # df_welfare_util_prior = calculate_welfare_for_different_swfs(
    #     datasets, data_loader, time_horizon, no_of_ensembles, population
    # )

    # Example usage within JUSTICE_stepwise_run #NOTE: This is optional for data analysis
    # save_constrained_emission_control_rate_at_percentile(
    #     datasets=datasets,
    #     constrained_emission_control_rate=constrained_emission_control_rate,
    #     time_horizon=time_horizon,
    #     path_to_output=path_to_output,
    #     output_file_name=output_file_name,
    #     rbf_policy_index=rbf_policy_index,
    #     year=2100,
    #     percentile=95,
    # )

    # Save the datasets
    if saving:
        # np.save(
        #     path_to_output + "baseline_emissions_" + str(rbf_policy_index),
        #     baseline_emissions,
        # )

        # if output_file_name:
        #     # Save the df
        #     df_welfare_util_prior.to_csv(
        #         path_to_output + output_file_name + "_" + str(rbf_policy_index) + ".csv"
        #     )

        np.save(
            path_to_output + output_file_name + "_" + str(rbf_policy_index), datasets
        )
        # np.save(
        #     "data/output/optimized_emissions_control_rate.npy",
        #     constrained_emission_control_rate,
        # )

    return datasets, model


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


def save_constrained_emission_control_rate_at_percentile(
    datasets,
    constrained_emission_control_rate,
    time_horizon,
    path_to_output,
    output_file_name,
    rbf_policy_index,
    year=2100,
    percentile=95,
):
    """
    Save the constrained emission control rate at a given percentile for a specific year.

    @param datasets: The datasets generated by the model
    @param constrained_emission_control_rate: The constrained emission control rate
    @param time_horizon: The time horizon object
    @param path_to_output: Path to save the output
    @param output_file_name: Output file name
    @param rbf_policy_index: RBF policy index
    @param year: The year for which the percentile is calculated. Default is 2100
    @param percentile: The desired percentile (e.g., 50 for median, 25 for 25th percentile, 95 for 95th percentile). Default is 95
    """
    timestep_from_year = time_horizon.year_to_timestep(year, timestep=1)

    # Calculate the temperature at the desired percentile
    temperature_percentile = np.percentile(
        datasets["global_temperature"][timestep_from_year, :], percentile
    )

    # Check which index is closest to the temperature at the desired percentile
    ensemble_index = np.argmin(
        np.abs(
            datasets["global_temperature"][timestep_from_year, :]
            - temperature_percentile
        )
    )

    print(f"{percentile}th Percentile Temperature: ", temperature_percentile)
    print("Ensemble Index: ", ensemble_index)

    # Select the ensemble index for the constrained emission control rate
    constrained_emission_control_rate_indexed = constrained_emission_control_rate[
        :, :, ensemble_index
    ]

    # Save the constrained emission control rate indexed as npy
    np.save(
        path_to_output
        + output_file_name
        + "_"
        + f"{percentile}th_percentile_"
        + "constrained_emission_control_rate_"
        + str(rbf_policy_index)
        + "ensembleidx"
        + str(ensemble_index),
        constrained_emission_control_rate_indexed,
    )


def calculate_baseline_emissions(model, datasets, scenarios):
    """
    Calculate baseline emissions based on the model, datasets, and scenarios.

    @param model: The JUSTICE model instance
    @param datasets: The datasets generated by the model
    @param scenarios: The scenario index
    @return: Baseline emissions
    """
    carbon_intensity = model.emissions.carbon_intensity[
        :, :, :, get_economic_scenario(scenarios)
    ]
    print("Carbon Intensity Shape: ", carbon_intensity.shape)

    gross_economic_output = datasets["gross_economic_output"]

    # Find baseline emissions
    baseline_emissions = carbon_intensity * gross_economic_output
    print("Baseline Emissions Shape: ", baseline_emissions.shape)

    return baseline_emissions


def calculate_welfare_for_different_swfs(
    datasets, data_loader, time_horizon, no_of_ensembles, population
):
    social_welfare_defaults = SocialWelfareDefaults()

    # Fetch the defaults for Social Welfare Function
    welfare_defaults_utilitarian = social_welfare_defaults.get_defaults(
        WelfareFunction.UTILITARIAN.name
    )

    # Fetch the defaults for Social Welfare Function
    welfare_defaults_prioritarian = social_welfare_defaults.get_defaults(
        WelfareFunction.PRIORITARIAN.name
    )

    welfare_function_utilitarian = SocialWelfareFunction(
        input_dataset=data_loader,
        time_horizon=time_horizon,
        climate_ensembles=no_of_ensembles,
        population=population,
        risk_aversion=welfare_defaults_utilitarian["risk_aversion"],
        elasticity_of_marginal_utility_of_consumption=welfare_defaults_utilitarian[
            "elasticity_of_marginal_utility_of_consumption"
        ],
        pure_rate_of_social_time_preference=welfare_defaults_utilitarian[
            "pure_rate_of_social_time_preference"
        ],
        inequality_aversion=welfare_defaults_utilitarian["inequality_aversion"],
        sufficiency_threshold=welfare_defaults_utilitarian["sufficiency_threshold"],
        egality_strictness=welfare_defaults_utilitarian["egality_strictness"],
    )

    welfare_function_prioritarian = SocialWelfareFunction(
        input_dataset=data_loader,
        time_horizon=time_horizon,
        climate_ensembles=no_of_ensembles,
        population=population,
        risk_aversion=welfare_defaults_prioritarian["risk_aversion"],
        elasticity_of_marginal_utility_of_consumption=welfare_defaults_prioritarian[
            "elasticity_of_marginal_utility_of_consumption"
        ],
        pure_rate_of_social_time_preference=welfare_defaults_prioritarian[
            "pure_rate_of_social_time_preference"
        ],
        inequality_aversion=welfare_defaults_prioritarian["inequality_aversion"],
        sufficiency_threshold=welfare_defaults_prioritarian["sufficiency_threshold"],
        egality_strictness=welfare_defaults_prioritarian["egality_strictness"],
    )

    _, _, _, datasets["welfare_utilitarian"] = (
        welfare_function_utilitarian.calculate_welfare(
            consumption_per_capita=datasets["consumption_per_capita"]
        )
    )
    _, _, _, datasets["welfare_prioritarian"] = (
        welfare_function_prioritarian.calculate_welfare(
            consumption_per_capita=datasets["consumption_per_capita"]
        )
    )

    timestep_2100 = time_horizon.year_to_timestep(2100, timestep=1)
    print("Timestep 2100: ", timestep_2100)

    net_output = datasets["net_economic_output"]
    # Sum over all regions
    net_output = np.sum(net_output, axis=0)
    # Select the 2100 timestep
    net_output = net_output[timestep_2100, :]

    damages = datasets["economic_damage"]
    # Sum over all regions
    damages = np.sum(damages, axis=0)
    # Select the 2100 timestep
    damages = damages[timestep_2100, :]

    abatement = datasets["abatement_cost"]
    # Sum over all regions
    abatement = np.sum(abatement, axis=0)
    # Select the 2100 timestep
    abatement = abatement[timestep_2100, :]

    temperature = datasets["global_temperature"]
    # Select the 2100 timestep
    temperature = temperature[timestep_2100, :]

    consumption_per_capita = datasets["consumption_per_capita"]
    # Sum over all regions
    consumption_per_capita = np.sum(consumption_per_capita, axis=0)
    # Select the 2100 timestep
    consumption_per_capita = consumption_per_capita[timestep_2100, :]

    # Print the shapes of net_output, damages, abatement, temperature
    print("Net Output Shape: ", net_output.shape)
    print("Damages Shape: ", damages.shape)
    print("Abatement Shape: ", abatement.shape)
    print("Temperature Shape: ", temperature.shape)

    # Net Output, Damages, Abatement, Temperature have shape (1001,). Combine them in a single dataframe with 4 columns
    df = pd.DataFrame(
        {
            "Net Output": net_output,
            "Consumption Per Capita": consumption_per_capita,
            "Damages": damages,
            "Abatement": abatement,
            "Temperature": temperature,
        }
    )

    return df


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

    # Print the welfare and years_above_temperature_threshold values # Diagnostics
    print("Welfare: ", rbf_decision_vars["welfare"])
    # print(
    #     "Years Above Temperature Threshold: ",
    #     rbf_decision_vars["years_above_temperature_threshold"],
    # )

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
    )

    # Stepwise run
    # datasets, _ = JUSTICE_stepwise_run(
    #     scenarios=2,
    #     social_welfare_function=WelfareFunction.UTILITARIAN,
    #     rbf_policy_index=66,
    #     path_to_rbf_weights="data/convergence_metrics/UTILITARIAN_reference_set.csv",
    #     saving=False,
    #     path_to_output="data/reevaluation/util_90_welfare_temp/",
    #     output_file_name="UTILITARIAN",
    # )

    # Print the keys of the datasets
    print("Welfare", datasets["welfare"])
    # print("Welfare Utilitarian", datasets["welfare_utilitarian"])
    # print("Welfare Prioritarian", datasets["welfare_prioritarian"])
