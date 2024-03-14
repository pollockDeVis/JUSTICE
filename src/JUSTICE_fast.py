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


class JUSTICE_PROJECTION:
    """
    This is the JUSTICE model.
    """

    def __init__(
        self,
        justice: JUSTICE
    ):
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
        #Here we don't just call from the original because we are going to run the model until end time and modify the content
        self.time_horizon = TimeHorizon(
            start_year=justice.time_horizon.start_year, end_year=justice.time_horizon.end_year, data_timestep=5, timestep=justice.time_horizon.timestep
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


        self.emissions = justice.emissions


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
            "emission_cutting_rate":  np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "opinion_share":  np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
        }


    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)

    def run(
        self,
        emission_control_rate,
        savings_rate=None,
        endogenous_savings_rate=False,
    ):
        """
        Run the model.
        """
        if endogenous_savings_rate == True:
            self.savings_rate = self.fixed_savings_rate
        else:
            self.savings_rate = savings_rate

        # Check the shape of the emission_control_rate whether 1D or 2D
        if len(emission_control_rate.shape) == 2:
            emission_control_rate = np.tile(
                emission_control_rate[:, :, np.newaxis], (1, 1, self.no_of_ensembles)
            )

        self.emission_control_rate = emission_control_rate

        for timestep in range(len(self.time_horizon.model_time_horizon)):
            """
            Main loop of the model. This loop runs the model for each timestep.
            """

            output = self.economy.run(
                scenario=self.scenario,
                timestep=timestep,
                savings_rate=self.savings_rate[:, timestep],
            )

            emissions_array = self.emissions.run(
                timestep=timestep,
                scenario=self.scenario,
                output=output,
                emission_control_rate=self.emission_control_rate[:, timestep, :],
            )

            # Run the model for all timesteps except the last one. Damages and Abatement applies to the next timestep
            if timestep < (len(self.time_horizon.model_time_horizon) - 1):
                global_temperature = self.climate.compute_temperature_from_emission(
                    timestep, emissions_array
                )

                regional_temperature = self.downscaler.get_regional_temperature(
                    global_temperature
                )

                # Save the regional temperature
                self.data["regional_temperature"][:, timestep, :] = regional_temperature

                damage = self.damage_function.calculate_damage(
                    temperature=regional_temperature, timestep=timestep
                )

                abatement_cost = self.abatement.calculate_abatement(
                    timestep=timestep,
                    emissions=emissions_array,
                    emission_control_rate=self.emission_control_rate[:, timestep, :],
                )
                # TODO: Incomplete Implementation
                # carbon_price = self.abatement.calculate_carbon_price(
                #     timestep=timestep,
                #     emission_control_rate=self.emissions_control_rate[:, timestep],
                # )

                self.economy.apply_damage_to_output(
                    timestep=timestep + 1, damage=damage
                )
                self.economy.apply_abatement_to_output(
                    timestep=timestep + 1, abatement=abatement_cost
                )
            elif timestep == (len(self.time_horizon.model_time_horizon) - 1):
                self.data["global_temperature"][timestep, :] = (
                    self.climate.get_justice_temperature_array()
                )[timestep, :]

                regional_temperature = self.downscaler.get_regional_temperature(
                    self.data["global_temperature"][timestep, :]
                )
                # Save the regional temperature
                self.data["regional_temperature"][:, timestep, :] = regional_temperature

    def evaluate(
        self,
    ):
        """
        Evaluate the model.
        """
        # Fill the data dictionary
        self.data["net_economic_output"] = self.economy.get_net_output()
        self.data["consumption"] = self.economy.calculate_consumption(
            savings_rate=self.savings_rate
        )
        self.data["consumption_per_capita"] = self.economy.get_consumption_per_capita(
            scenario=self.scenario,
            savings_rate=self.savings_rate,
        )

        self.data["emissions"] = self.emissions.get_emissions()
        self.data["economic_damage"] = self.economy.get_damages()
        self.data["abatement_cost"] = self.economy.get_abatement()
        self.data["global_temperature"] = self.climate.get_justice_temperature_array()

        # TODO: to be implemented later. Checking the enums doesn't work well with EMA #need to make it self.welfare_function?
        # if welfare_function == WelfareFunction.UTILITARIAN:

        (
            self.data["disentangled_utility"],
            self.data["welfare_utilitarian_regional_temporal"],
            self.data["welfare_utilitarian_temporal"],
            self.data["welfare_utilitarian_regional"],
            self.data["welfare_utilitarian"],
        ) = self.welfare_function.calculate_welfare(
            consumption_per_capita=self.data["consumption_per_capita"]
        )
        return self.data

    def get_outcome_names(self):
        """
        Get the list of outcomes of the model.
        """
        return self.data.keys()
