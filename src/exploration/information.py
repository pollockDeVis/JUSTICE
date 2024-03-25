# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:45:43 2024

@author: apoujon
"""

import numpy as np
import pandas as pd
from typing import Any
import copy

from src.util.data_loader import DataLoader
from src.util.enumerations import Economy, DamageFunction, Abatement, WelfareFunction
from src.util.model_time import TimeHorizon
from src.economy.neoclassical import NeoclassicalEconomyModel
from src.emissions.emission import OutputToEmissions
from src.damage.kalkuhl import DamageKalkuhl
from src.climate.coupled_fair import CoupledFAIR
from src.climate.temperature_downscaler import TemperatureDownscaler
from src.abatement.abatement_enerdata import AbatementEnerdata
from src.welfare.utilitarian import Utilitarian
from src.exploration.twolevelsgame import TwoLevelsGame
from src.exploration.household import Household
from src.model import JUSTICE
from src.JusticeProjection import JusticeProjection


class Information:

    def __init__(
        self,
        justice_model,
        start_year=2015,  # Model is only tested for start year 2015
        end_year=2300,  # Model is only tested for end year 2300
        timestep=1,  # Model is only tested for timestep 1
        scenario=0,
        climate_ensembles=None,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=WelfareFunction.UTILITARIAN,
        **kwargs,
    ):

        self.justice_model = justice_model

        # self.elasticity_of_marginal_utility_of_consumption = 1.45
        # self.pure_rate_of_social_time_preference = 0.015
        # self.inequality_aversion = 0

        self.IPCC_report_period = 2000

        # TODO APN: Change to numpy arrays
        self.global_temperature_information = []
        # Last known forecast by FaIR (every 10 years) under the form [year; mean_temperature; var_temperature]
        self.local_temperature_information = []
        # Last known downscaled forecast by FaIr (every 10 years) under the forme [year; mean_temperature; var_temperature] for each region
        self.global_distrib_flsi = [[] for i in range(Household.N_CLIMATE_BELIEFS)]
        # Estimations of future temperature elevation (ground) at global scale at future years BELIEF_YEAR_OFFSET
        self.regional_distrib_flsi = [
            [[] for i in range(Household.N_CLIMATE_BELIEFS)] for r in range(57)
        ]
        # Estimations of future temperature elevation (ground) at local scale at future years BELIEF_YEAR_OFFSET
        # TODO APN: 57 is the number of regions: use global var or get from JUSTICE model

    def step(self, timestep):

        # TODO APN: don't use timestep as it may not correspond 1::1 to a year increment
        if timestep % self.IPCC_report_period == 0:
            self.generate_information()
        self.construct_flsi(timestep)

        return

    def generate_information(self):
        """
        Take into account a JUSTICE model and future emissions control rate for all regions.
        Return the full set of data for the execution of the JUSTICE model under these conditions.
        Is used in the model a a source of information (eg. similar to an IPCC report)

        Parameters
        ----------
        justice_model : TYPE
            DESCRIPTION. A deepcopy of the current JUSTICE model
        emissions_control_rate : TYPE
            DESCRIPTION. array: region * time horizon array

        Returns
        -------
        None.

        """

        # Get a fresh model
        # information_model = JUSTICE()
        print("      -> RUNNING PROJECTION MODEL")
        information_model = JusticeProjection(self.justice_model)
        information_model.run(
            emission_control_rate=self.justice_model.emission_control_rate,
            endogenous_savings_rate=True
        )
        datasets = (
            information_model.evaluate()
        )  # Might be useful to just create an evaluation function for the relevant informations only
        self.generate_climate_information(
            datasets["global_temperature"], datasets["regional_temperature"]
        )
        print(
            "         L> PROJECTION DONE! (2300 mean temperature :",
            self.global_temperature_information[0][-1],
            "C!)",
        )

        #print(datasets["consumption_per_capita"])
        return

    def generate_climate_information(self, g_temp, l_temp):
        # g_temp An array of time horizon * number of ensembles
        # l_temp An array of size regions * time horizon * number of ensembles

        # Global Temperatures: expected means and variances
        means_global_temp = g_temp.mean(axis=1)
        # array of shape time horizon
        std_global_temp = g_temp.std(axis=1)
        # array of shape time horizon
        self.global_temperature_information = [means_global_temp, std_global_temp]
        # array of shape  2 * time_horizon

        # Local Temperatures: expected means and variances
        means_local_temp = l_temp.mean(axis=2)
        # array of shape region * time horizon
        std_local_temp = l_temp.std(axis=2)
        # array of shape region * time horizon
        self.local_temperature_information = [means_local_temp, std_local_temp]
        # array of shape  2 * region * time_horizon
        return

    def construct_flsi(self, time):
        # Global
        for i in range(Household.N_CLIMATE_BELIEFS):
            year = Household.BELIEF_YEAR_OFFSET[i]
            self.global_distrib_flsi[i] = Household.gaussian_distrib(
                g_mean=self.global_temperature_information[0][year],
                g_std=self.global_temperature_information[1][year],
            )
            norm_coeff = np.sum(self.global_distrib_flsi[i], axis=0)
            self.global_distrib_flsi[i] = self.global_distrib_flsi[i] / norm_coeff

        # Local
        for r in range(57):
            for i in range(Household.N_CLIMATE_BELIEFS):
                year = Household.BELIEF_YEAR_OFFSET[i]
                self.regional_distrib_flsi[r][i] = Household.gaussian_distrib(
                    g_mean=self.local_temperature_information[0][r][year],
                    g_std=self.local_temperature_information[1][r][year],
                )
                norm_coeff = np.sum(self.regional_distrib_flsi[r][i], axis=0)
                # TODO NOT SURE ABOUT THE COMPUTATION OF NORM COEFF HERE
                self.regional_distrib_flsi[r][i] = (
                    self.regional_distrib_flsi[r][i] / norm_coeff
                )

        return
