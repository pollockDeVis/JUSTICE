import pandas as pd
from ema_workbench import Model, RealParameter, ScalarOutcome, perform_experiments
from ema_workbench.util import ema_logging

from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.LogFiles import print_log, LogFiles
from src.drafts_and_tests.utils_save import visualize_policy
from src.drafts_and_tests.utils_save_household_thresholds import (
    visualize_household_thresholds,
)
from src.util.enumerations import *

# from src.model_abm import OLD_ABM_JUSTICE
# loading XML file
from src.model_abm_justice import AbmJustice
import matplotlib
from scipy.stats import qmc


def full_run_justice_abm(
    Region_alpha1, Region_alpha2, Region_beta1, Region_beta2, Region_gamma
):
    scenarios = 7
    print_log.__init__()
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
        Region_alpha1=Region_alpha1,
        Region_alpha2=Region_alpha2,
        Region_beta1=Region_beta1,
        Region_beta2=Region_beta2,
        Region_gamma=Region_gamma,
    )

    ###############################################################################
    #####################       Step-by-Step Run        ###########################
    ###############################################################################
    model.full_run(max_time_steps=85)

    ###############################################################################
    #       Additional Data Saves in addition to saves by print_log-by-Step       #
    ###############################################################################

    return {"year_global_net_zero": model.two_levels_game.year_global_net_zero}


sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=100)
l_bounds = [75, 0, 0, 0, 0]
u_bounds = [100, 2, 2, 2, 2]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
results = []
for sample in sample_scaled:
    results.append(full_run_justice_abm(
                        sample[0], sample[1], sample[2], sample[3], sample[4]
                    )
    )
    print(results[-1])

print(results)


"""
ema_logging.log_to_stderr(ema_logging.INFO)
analyzer_model = Model("justiceAbmModel", function=full_run_justice_abm)
analyzer_model.uncertainties = [
    RealParameter("Region_alpha1", 0, 100),
    RealParameter("Region_alpha2", 0, 2),
    RealParameter("Region_beta1", 0, 2),
    RealParameter("Region_beta2", 0, 0),
    RealParameter("Region_gamma", 0, 10)
]
analyzer_model.outcomes = [ScalarOutcome("year_global_net_zero")]
results = perform_experiments(analyzer_model, 10, )
"""
