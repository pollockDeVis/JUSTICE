# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:08:58 2024

@author: apoujon
"""

import pandas as pd
import numpy as np

from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
#from src.model_abm import OLD_ABM_JUSTICE
from src.model_abm_justice import AbmJustice
import matplotlib
from matplotlib import pyplot as plt
from alive_progress import alive_bar
from src.util.emission_control_constraint import EmissionControlConstraint
matplotlib.rcParams['figure.dpi']=300


# matplotlib.use('Qt5Agg') #For real time display


###############################################################################
########################       ABM-JUSTICE      ###############################
###############################################################################
# Get list of Scenarios from Enum
for idx, scenarios in enumerate(list(Scenario.__members__.keys())):
    print(idx, scenarios)



scenarios = 7
model = AbmJustice(
    start_year=2015,  # Model is only tested for start year 2015
    end_year=2300,  # Model is only tested for end year 2300
    timestep=1,  # Model is only tested for timestep 1
    scenario=scenarios,
    economy_type=Economy.NEOCLASSICAL,
    damage_function_type=DamageFunction.KALKUHL,
    abatement_type=Abatement.ENERDATA,
    social_welfare_function=WelfareFunction.UTILITARIAN,
    # climate_ensembles=570, # This is to select a specific climate ensemble
    # Declaring for endogenous fixed savings rate
    elasticity_of_marginal_utility_of_consumption=1.45,
    pure_rate_of_social_time_preference=0.015,
    seed=None,
    utility_params=[0,0,0,0,0],
)


###############################################################################
#####################       Step-by-Step Run        ###########################
###############################################################################
print("Step-by-step run:")
with alive_bar(len(model.time_horizon.model_time_horizon), force_tty=True) as bar:
    for timestep in range(len(model.time_horizon.model_time_horizon)):

        model.abm_stepwise_run(
            timestep=timestep, endogenous_savings_rate=True
        )  # savings_rate = fixed_savings_rate[:, timestep],
        datasets = model.stepwise_evaluate(timestep=timestep)

        bar()


model.close_files()
print("DONE! :D")

# Shape of disentangled_utility, disentangled_utility_summed, disentangled_utility_powered, welfare_utilitarian
# Shape of discount_rate
# (57, 286, 1)
# (57, 286, 1001) (286, 1001) (286, 1001) (286, 1001)
# (57, 286, 1001) (286, 1001) (286, 1001) (1001,)
# Shape of discount_rate
# (286, 1)

print("-> Gathering data")
net_output = datasets["net_economic_output"]
consumption = datasets["consumption"]  # (57, 286, 1001)
cpc = datasets["consumption_per_capita"]  # (57, 286, 1001)
emis = datasets["emissions"]  # (57, 286, 1001)
reg_temp = datasets["regional_temperature"]
temp = datasets["global_temperature"]  # (286, 1001)
damages = datasets["economic_damage"]  # (57, 286, 1001)
abatecost = datasets["abatement_cost"]  # (57, 286, 1001)
disentangled_utility = datasets["disentangled_utility"]  # (57, 286, 1001)
welfare_utilitarian = datasets["welfare_utilitarian"]  # (1001,)
welfare_utilitarian_temporal = datasets["welfare_utilitarian_temporal"]  # (286, 1001)
welfare_utilitarian_regional = datasets["welfare_utilitarian_regional"]  # (57, 1001)
welfare_utilitarian_regional_temporal = datasets[
    "welfare_utilitarian_regional_temporal"
]  # (57, 286, 1001)

emission_cutting_rate_temporal = datasets["emission_cutting_rate"]
print("   OK")

import logging

import seaborn as sns

# Create list of all the data arrays
data_list = [
    net_output,
    cpc,
    emis,
    disentangled_utility,
    damages,
    abatecost,
    emission_cutting_rate_temporal,
    reg_temp,
]
titles = [
    "Net Economic Output",
    "Consumption per Capita",
    "Emissions",
    "Disaggregated Utility",
    "Economic Damages",
    "Abatement Cost",
    "Emission rate",
    "Temperature (regional)"
]


# data_list = [net_output, cpc, emis, emission_cutting_rate_temporal]
# titles = ['Net Economic Output', 'Consumption per Capita', 'Emissions', 'Emission rate', '']


list_region_index = [0, 1]

for region_index in list_region_index:
    # Create a figure with 2 rows and 3 columns
    fig, axs = plt.subplots(4, 2, figsize=(15, 10))

    # Flatten the axs array to iterate over it
    axs = axs.flatten()

    print("-> Drawing figures")

    # Iterate over the data arrays and plot them

    for i, data in enumerate(data_list):
        # Select the region based on region_index
        region_data = data[region_index, :, :]

        # Create a line plot for each scenario
        for j in range(region_data.shape[1]):
            sns.lineplot(
                x=model.time_horizon.model_time_horizon,
                y=region_data[:, j],
                ax=axs[i],
            )

        # Set the title and axis labels
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Year")
        axs[i].set_ylabel("Value")



    # Remove the unused subplots
    for i in range(len(data_list), len(axs)):
        fig.delaxes(axs[i])

    # Adjust the layout and spacing
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    fig.show()

print("   OK")
