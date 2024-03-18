"""
This is the main JUSTICE model.
"""

import numpy as np
import pandas as pd
from typing import Any

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
from src.model import JUSTICE
from src.exploration.information import Information


class ABM_JUSTICE:
    """
    This is the ABM_JUSTICE model.
    """

    def __init__(
        self,
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
        self.welfare_function = social_welfare_function

        # Load the data
        print("   -> Loading data")
        self.data_loader = DataLoader()
        print("      OK")

        # Instantiate the TimeHorizon class
        self.time_horizon = TimeHorizon(
            start_year=start_year, end_year=end_year, data_timestep=5, timestep=timestep
        )

        self.scenario = scenario

        # INSTANTIATION OF POLICY MODULE
        print("   -> Instantiation of policy module")
        self.two_levels_game = TwoLevelsGame(self, timestep=timestep)
        print("      OK")

        # Instantiate FaIR
        print("   -> Setting up CoupledFaIR")
        self.climate = CoupledFAIR(ch4_method="Thornhill2021")
        self.downscaler = TemperatureDownscaler(input_dataset=self.data_loader)
        print("      OK")

        print("   -> Climate ensembles")
        # Check if climate_ensembles is passed as a parameter
        if climate_ensembles is not None:
            self.no_of_ensembles = self.climate.fair_justice_run_init(
                time_horizon=self.time_horizon,
                scenarios=self.scenario,
                climate_ensembles=climate_ensembles,
            )
        else:
            self.no_of_ensembles = self.climate.fair_justice_run_init(
                time_horizon=self.time_horizon, scenarios=self.scenario
            )
        print("      OK")

        print("   -> Loading regions")
        self.region_list = self.data_loader.REGION_LIST
        print("      OK")

        print("   -> Policy levers")
        # Set the savings rate and emissions control rate levers
        self.fixed_savings_rate = np.zeros(
            (
                len(self.data_loader.REGION_LIST),
                len(self.time_horizon.model_time_horizon),
            )
        )

        # Set the savings rate and emissions control rate levers
        self.savings_rate = np.zeros(
            (
                len(self.data_loader.REGION_LIST),
                len(self.time_horizon.model_time_horizon),
            )
        )

        self.emission_control_rate = np.zeros(
            (
                len(self.data_loader.REGION_LIST),
                len(self.time_horizon.model_time_horizon),
                self.no_of_ensembles,
            )
        )
        print("      OK")

        print("   -> Setting up the economy and damage functions")
        # TODO: Checking the Enums in the init is sufficient as long as the name of the methods are same across all classes
        # I think it is failing because I am checking self.economy_type instead of economy_type, which is passed as a parameter
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

        # Checking if the savings rate is endogenous or exogenous by checking kwargs
        if (
            "elasticity_of_marginal_utility_of_consumption" in kwargs
            and "pure_rate_of_social_time_preference" in kwargs
        ):
            elasticity_of_marginal_utility_of_consumption = kwargs[
                "elasticity_of_marginal_utility_of_consumption"
            ]
            pure_rate_of_social_time_preference = kwargs[
                "pure_rate_of_social_time_preference"
            ]

            self.fixed_savings_rate = self.economy.get_fixed_savings_rate(
                elasticity_of_marginal_utility_of_consumption=elasticity_of_marginal_utility_of_consumption,
                pure_rate_of_social_time_preference=pure_rate_of_social_time_preference,
            )

        print("      OK")

        self.emissions = OutputToEmissions(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            climate_ensembles=self.no_of_ensembles,
        )

        print("   -> Instantiation of the Welfare function")
        # TODO: Incomplete Implementation
        # if self.social_welfare_function == WelfareFunction.UTILITARIAN:
        self.welfare_function = Utilitarian(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            population=self.economy.get_population(scenario=self.scenario),
            **kwargs,
        )
        print("      OK")

        print("   -> Building data saving structure")
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
            "emission_cutting_rate": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "opinion_share": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
        }
        print("      OK")

        # INSTANTIATE INFORMATION MODULE
        print("   -> Loading information")
        self.information_model = Information(
            self,
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
        )

        self.information_model.generate_information(self, self.emission_control_rate)
        print("      OK")

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)

    def stepwise_run(
        self,
        timestep,
        savings_rate=None,
        endogenous_savings_rate=False,
    ):
        """
        Note that emission_control_rate have been removed from the args as it in endogeneous 
        
        This method is used for Reinforcement Learning (RL) applications.

        Run the model timestep by timestep and return the outcomes every timestep

        @param timestep: The timestep to run the model for 0 to model_time_horizon
        @param savings_rate: The savings rate for each timestep. So shape will be (no_of_regions,)
        @param emission_control_rate: The emissions control rate for each timestep. So shape will be (no_of_regions,)
        """

        self.information_model.step(timestep)
        self.two_levels_game.step(timestep)
        emission_control_rate = self.emission_control_rate[:, timestep]

        # Error check on the inputs
        assert timestep >= 0 and timestep <= len(
            self.time_horizon.model_time_horizon
        ), "The given timestep is out of range."

        if endogenous_savings_rate == True:
            self.savings_rate[:, timestep] = self.fixed_savings_rate[:, timestep]
        else:
            self.savings_rate[:, timestep] = savings_rate

        # Check the shape of the emission_control_rate whether it is 1D or 2D
        if len(emission_control_rate.shape) == 1:
            emission_control_rate = np.tile(
                emission_control_rate[:, np.newaxis], (1, self.no_of_ensembles)
            )

        self.emission_control_rate[:, timestep, :] = emission_control_rate

        gross_output = self.economy.run(
            scenario=self.scenario,
            timestep=timestep,
            savings_rate=self.savings_rate[:, timestep],
        )

        self.data["emissions"][:, timestep, :] = self.emissions.run(
            timestep=timestep,
            scenario=self.scenario,
            output=gross_output,
            emission_control_rate=self.emission_control_rate[:, timestep, :],
        )

        # Run the model for all timesteps except the last one. Damages and Abatement applies to the next timestep
        if timestep < (len(self.time_horizon.model_time_horizon) - 1):
            # Filling in the temperature of the first timestep from FAIR
            if timestep == 0:
                self.data["global_temperature"][
                    0, :
                ] = self.climate.get_justice_initial_temperature()

                self.data["regional_temperature"][:, 0, :] = (
                    self.downscaler.get_regional_temperature(
                        self.data["global_temperature"][0, :]
                    )
                )

            self.data["global_temperature"][(timestep + 1), :] = (
                self.climate.compute_temperature_from_emission(
                    timestep, self.data["emissions"][:, timestep, :]
                )
            )


            # Save the regional temperature
            self.data["regional_temperature"][:, (timestep + 1), :] = (
                self.downscaler.get_regional_temperature(
                    self.data["global_temperature"][(timestep + 1), :]
                )
            )

            damage = self.damage_function.calculate_damage(
                temperature=self.data["regional_temperature"][:, timestep, :],
                timestep=timestep,
            )

            abatement_cost = self.abatement.calculate_abatement(
                timestep=timestep,
                scenario=self.scenario,
                emission_control_rate=emission_control_rate,
            )
            # Apply the computed damage and abatement to the economic output for the next timestep.
            self.economy.apply_damage_to_output(timestep=timestep + 1, damage=damage)
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

        # Save the data
        self.data["net_economic_output"][:, timestep, :] = (
            self.economy.get_net_output()
        )[:, timestep, :]
        self.data["regional_temperature"][:, timestep, :] = self.data[
            "regional_temperature"
        ][:, timestep, :]
        self.data["emissions"][:, timestep, :] = (self.emissions.get_emissions())[
            :, timestep, :
        ]
        self.data["economic_damage"][:, timestep, :] = (self.economy.get_damages())[
            :, timestep, :
        ]
        self.data["abatement_cost"][:, timestep, :] = (self.economy.get_abatement())[
            :, timestep, :
        ]
        self.data["global_temperature"][timestep, :] = (
            self.climate.get_justice_temperature_array()
        )[timestep, :]
        self.data["consumption"][:, timestep, :] = (
            self.economy.calculate_consumption_per_timestep(
                self.savings_rate[:, timestep], timestep
            )
        )
        self.data["consumption_per_capita"][:, timestep, :] = (
            self.economy.get_consumption_per_capita_per_timestep(
                self.scenario, self.savings_rate[:, timestep], timestep
            )
        )

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

    def stepwise_evaluate(
        self,
        timestep=0,
    ):
        """
        Evaluate the model timestep by timestep and return the outcomes every timestep

        @param timestep: The timestep to run the model for 0 to model_time_horizon
        @param elasticity_of_marginal_utility_of_consumption: The elasticity of marginal utility of consumption
        @param pure_rate_of_social_time_preference: The pure rate of social time preference
        @param inequality_aversion: The inequality aversion
        @param welfare_function: The welfare function to use
        """

        # Error check on the inputs
        assert timestep >= 0 and timestep <= len(
            self.time_horizon.model_time_horizon
        ), "The given timestep is out of range."

        # TODO : Check the enums. To be implemented later
        # if welfare_function == WelfareFunction.UTILITARIAN:

        # TODO: Return step by step individual data/observations
        # TODO: FOr RL, we have to separate obeservation and rewards

        (
            self.data["disentangled_utility"][:, timestep, :],
            self.data["welfare_utilitarian_regional_temporal"][:, timestep, :],
            self.data["welfare_utilitarian_temporal"][timestep, :],
        ) = self.welfare_function.calculate_stepwise_welfare(
            consumption_per_capita=self.data["consumption_per_capita"][:, timestep, :],
            timestep=timestep,
        )
        return self.data

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

    def close_files(self):
        self.two_levels_game.close_files()
