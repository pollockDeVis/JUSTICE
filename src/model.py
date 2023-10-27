"""
This is the main JUSTICE model.
"""

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.enumerations import Economy, DamageFunction, Abatement, WelfareFunction
from src.model_time import TimeHorizon
from src.economy.neoclassical import NeoclassicalEconomyModel
from src.emissions.emission import OutputToEmissions
from src.damage.kalkuhl import DamageKalkuhl
from src.climate.coupled_fair import CoupledFAIR
from src.climate.temperature_downscaler import TemperatureDownscaler
from src.abatement.abatement_enerdata import AbatementEnerdata
from src.welfare.utilitarian import calculate_utilitarian_welfare

# from src import utils


class JUSTICE:
    """
    This is the JUSTICE model.
    """

    def __init__(
        self,
        start_year,
        end_year,
        timestep,
        scenario,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
    ):
        """
        @param start_year: The start year of the model
        @param end_year: The end year of the model
        @param timestep: The timestep of the model
        @param scenario: The scenario of the model
        """

        # Save the model configuration #TODO - These are not fully implemented yet
        self.economy_type = economy_type
        self.damage_function_type = damage_function_type
        self.abatement_type = abatement_type

        # Load the data
        self.data_loader = DataLoader()

        # Instantiate the TimeHorizon class
        self.time_horizon = TimeHorizon(
            start_year=start_year, end_year=end_year, data_timestep=5, timestep=timestep
        )

        self.scenario = scenario

        self.climate = CoupledFAIR()
        self.downscaler = TemperatureDownscaler(input_dataset=self.data_loader)

        self.no_of_ensembles = self.climate.fair_justice_run_init(
            time_horizon=self.time_horizon, scenarios=self.scenario
        )
        self.region_list = self.data_loader.REGION_LIST
        # TODO: Checking the Enums in the init is sufficient as long as the name of the methods are same across all classes
        # TODO: Incomplete Implementation
        # if self.damage_function_type == DamageFunction.KALKUHL:
        self.damage_function = DamageKalkuhl(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            climate_ensembles=self.no_of_ensembles,
        )
        # TODO: Incomplete Implementation
        # if self.abatement_type == Abatement.ENERDATA:
        self.abatement = AbatementEnerdata(
            input_dataset=self.data_loader, time_horizon=self.time_horizon
        )
        # TODO: Incomplete Implementation
        # if self.economy_type == Economy.NEOCLASSICAL:
        self.economy = NeoclassicalEconomyModel(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            climate_ensembles=self.no_of_ensembles,
        )

        self.emissions = OutputToEmissions(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            climate_ensembles=self.no_of_ensembles,
        )

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
            "welfare_utilitarian": np.zeros((self.no_of_ensembles)),
        }

    def run(self, savings_rate, emissions_control_rate):
        """
        Run the model.
        """
        # Save the savings rate and emissions control rate
        self.savings_rate = savings_rate
        self.emissions_control_rate = emissions_control_rate

        for timestep in range(len(self.time_horizon.model_time_horizon)):
            """
            Main loop of the model. This loop runs the model for each timestep.
            """

            output = self.economy.run(
                scenario=self.scenario,
                timestep=timestep,
                savings_rate=self.savings_rate[:, timestep],
            )

            self.emissions_array = self.emissions.run_emissions(
                timestep=timestep,
                scenario=self.scenario,
                output=output,
                emission_control_rate=self.emissions_control_rate[:, timestep],
            )

            # Run the model for all timesteps except the last one. Damages and Abatement applies to the next timestep
            if timestep < (len(self.time_horizon.model_time_horizon) - 1):
                global_temperature = self.climate.compute_temperature_from_emission(
                    timestep, self.emissions_array
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
                    emissions=self.emissions_array,
                    emission_control_rate=self.emissions_control_rate[:, timestep],
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

    def evaluate(
        self,
        elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference,
        inequality_aversion,
        welfare_function=WelfareFunction.UTILITARIAN,
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

        population = self.economy.get_population(scenario=self.scenario)

        if welfare_function == WelfareFunction.UTILITARIAN:
            (
                self.data["disentangled_utility"],
                self.data["welfare_utilitarian"],
            ) = calculate_utilitarian_welfare(
                time_horizon=self.time_horizon,
                region_list=self.region_list,
                scenario=self.scenario,
                # savings_rate=self.savings_rate,
                population=population,
                consumption_per_capita=self.data["consumption_per_capita"],
                elasticity_of_marginal_utility_of_consumption=elasticity_of_marginal_utility_of_consumption,
                pure_rate_of_social_time_preference=pure_rate_of_social_time_preference,
                inequality_aversion=inequality_aversion,
            )
        return self.data

    def get_outcome_names(self):
        """
        Get the list of outcomes of the model.
        """
        return self.data.keys()
