"""
This is the main JUSTICE model.
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
from src.model import JUSTICE
from src.exploration.twolevelsgame import TwoLevelsGame


class JusticeProjection(JUSTICE):
    """
    This is the JusticeProjection class. It redefines the JUSTICE class by overriding the init() method to initialize.
    """

    def __init__(self, justice: JUSTICE):
        """
        @param justice: The reference model
        """

        # Save the model configuration #TODO - These are not fully implemented yet
        self.economy_type = justice.economy_type
        self.damage_function_type = justice.damage_function_type
        self.abatement_type = justice.abatement_type
        self.welfare_function = justice.welfare_function

        # Load the data
        self.data_loader = copy.deepcopy(justice.data_loader)

        # Instantiate the TimeHorizon class
        # Here we don't just call from the original because we are going to run the model until end time and modify the content
        self.time_horizon = TimeHorizon(
            start_year=justice.time_horizon.start_year,
            end_year=justice.time_horizon.end_year,
            data_timestep=5,
            timestep=justice.time_horizon.timestep,
        )

        self.scenario = justice.scenario

        self.climate = copy.deepcopy(justice.climate)
        self.downscaler = justice.downscaler

        # Check if climate_ensembles is passed as a parameter
        self.no_of_ensembles = justice.no_of_ensembles

        self.region_list = self.data_loader.REGION_LIST

        # Set the savings rate and emissions control rate levers
        self.fixed_savings_rate = justice.fixed_savings_rate

        # Set the savings rate and emissions control rate levers
        self.savings_rate = justice.savings_rate

        self.emission_control_rate = justice.emission_control_rate

        # TODO: Checking the Enums in the init is sufficient as long as the name of the methods are same across all classes
        # I think it is failing because I am checking self.economy_type instead of economy_type, which is passed as a parameter
        # TODO: Incomplete Implementation
        # if self.damage_function_type == DamageFunction.KALKUHL:
        self.damage_function = copy.deepcopy(justice.damage_function)
        # TODO: Incomplete Implementation
        # if self.abatement_type == Abatement.ENERDATA:
        self.abatement = copy.deepcopy(justice.abatement)

        # TODO: Incomplete Implementation
        # if self.economy_type == Economy.NEOCLASSICAL:
        self.economy = copy.deepcopy(justice.economy)

        self.emissions = copy.deepcopy(justice.emissions)

        # Create a data dictionary to store the data
        self.data = {
            "net_economic_output": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "consumption": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "consumption_per_capita": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "emissions": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "regional_temperature": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "global_temperature": np.zeros(
                (len(self.time_horizon.model_time_horizon), self.no_of_ensembles)
            ),
            "economic_damage": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "abatement_cost": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "carbon_price": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "disentangled_utility": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "welfare_utilitarian_regional_temporal": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "welfare_utilitarian_regional": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    self.no_of_ensembles,
                )
            ),
            "welfare_utilitarian_temporal": np.zeros(
                (
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "welfare_utilitarian": np.zeros((self.no_of_ensembles,)),
        }

