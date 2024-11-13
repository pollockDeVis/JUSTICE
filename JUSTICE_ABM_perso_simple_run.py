# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:08:58 2024

@author: apoujon
"""
import time


import numpy as np
from matplotlib import pyplot as plt

from src.drafts_and_tests.utils_visualize_HK_opinion_dynamics import visualize_HK_OpDyn
from src.util.model_time import TimeHorizon
from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.LogFiles import print_log
from src.drafts_and_tests.utils_visualize_policy import visualize_policy
from src.drafts_and_tests.utils_save_household_thresholds import (
    visualize_household_thresholds,
)

from src.util.enumerations import *


# from src.model_abm import OLD_ABM_JUSTICE
# loading XML file
from src.model_abm_justice import AbmJustice
import matplotlib
import seaborn as sns

matplotlib.rcParams["figure.dpi"] = 300


# matplotlib.use('Qt5Agg') #For real time display


###############################################################################
########################       ABM-JUSTICE      ###############################
###############################################################################
# Get list of Scenarios from Enum
for idx, scenarios in enumerate(list(Scenario.__members__.keys())):
    print(idx, scenarios)


# np.seterr(invalid='warn')
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
    seed=XML_init_values.dict["seed"],
    Region_alpha1=75,
    Region_alpha2=0.1,
    Region_beta1=1,
    Region_beta2=1,
    Region_gamma=1,
)

###############################################################################
#####################       Step-by-Step Run        ###########################
###############################################################################
model.full_run(max_time_steps=85)

region_list = [32]
print("--> Visualizing results for regions: ", region_list)
print("   -> Save directory is: ", print_log.path)
for region in region_list:
    print("      -> Region ", region)
    time.sleep(1)
    visualize_HK_OpDyn(print_log.path, region)
    time.sleep(1)
    visualize_policy(print_log.path, region)
    print("         L> OK")


