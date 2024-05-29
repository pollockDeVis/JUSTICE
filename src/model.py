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
from src.welfare.social_welfare_function import SocialWelfareFunction
from config.default_parameters import SocialWelfareDefaults


class JUSTICE:
    """
    This is the JUSTICE model.
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
        ############################################################################################################################################################
        #
        #   Instatiating the JUSTICE Model
        #
        ############################################################################################################################################################
        # Set the model blocks
        self.economy_type = economy_type
        self.damage_function_type = damage_function_type
        self.abatement_type = abatement_type
        self.scenario = scenario

        # Check for Kwargs. This is implemented for EMA Workbench Support
        if "social_welfare_function_type" in kwargs:
            self.welfare_function_type = WelfareFunction.from_index(
                kwargs["social_welfare_function_type"]
            )
        else:
            self.welfare_function_type = social_welfare_function

        # Load the datasets by instantiating the DataLoader class
        self.data_loader = DataLoader()
        self.region_list = self.data_loader.REGION_LIST

        # TODO: Need to do the data slicing here for different start and end years
        # Instantiate the TimeHorizon class
        self.time_horizon = TimeHorizon(
            start_year=start_year, end_year=end_year, data_timestep=5, timestep=timestep
        )

        ############################################################################################################################################################
        #
        #   Instatiating the FAIR Climate Model
        #
        ############################################################################################################################################################
        # Instantiate the CoupledFAIR class
        self.climate = CoupledFAIR(ch4_method="Thornhill2021")
        self.downscaler = TemperatureDownscaler(input_dataset=self.data_loader)

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

        ############################################################################################################################################################
        #
        #   Initializing the Data Structures & Loading the Default Parameters
        #
        ############################################################################################################################################################

        # Set the savings rate and emissions control rate levers
        # TODO: check if we need this
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

        # Instantiate the SocialWelfareDefaults class
        social_welfare_defaults = SocialWelfareDefaults()
        # Fetch the defaults for Social Welfare Function
        welfare_defaults = social_welfare_defaults.get_defaults(
            self.welfare_function_type.name
        )
        # Assign the defaults to the class attributes
        self.elasticity_of_marginal_utility_of_consumption = welfare_defaults[
            "elasticity_of_marginal_utility_of_consumption"
        ]
        self.pure_rate_of_social_time_preference = welfare_defaults[
            "pure_rate_of_social_time_preference"
        ]
        self.inequality_aversion = welfare_defaults["inequality_aversion"]
        self.sufficiency_threshold = welfare_defaults["sufficiency_threshold"]
        self.egality_strictness = welfare_defaults["egality_strictness"]

        ############################################################################################################################################################
        #
        #   Instantiating the Model Blocks
        #
        ############################################################################################################################################################
        # TODO: Incomplete Implementation
        if self.damage_function_type == DamageFunction.KALKUHL:
            self.damage_function = DamageKalkuhl(
                input_dataset=self.data_loader,
                time_horizon=self.time_horizon,
                climate_ensembles=self.no_of_ensembles,
            )
        else:
            # Assert and raise an error if the damage function is not implemented
            assert False, "The damage function is not provided!"

        # TODO: Incomplete Implementation
        if self.abatement_type == Abatement.ENERDATA:
            self.abatement = AbatementEnerdata(
                input_dataset=self.data_loader, time_horizon=self.time_horizon
            )
        else:
            # Assert and raise an error if the abatement model is not implemented
            assert False, "The abatement model is not provided!"

        # TODO: Incomplete Implementation
        if self.economy_type == Economy.NEOCLASSICAL:
            self.economy = NeoclassicalEconomyModel(
                input_dataset=self.data_loader,
                time_horizon=self.time_horizon,
                scenario=self.scenario,
                climate_ensembles=self.no_of_ensembles,
                elasticity_of_marginal_utility_of_consumption=self.elasticity_of_marginal_utility_of_consumption,
                pure_rate_of_social_time_preference=self.pure_rate_of_social_time_preference,
            )
        else:
            # Assert and raise an error if the economy model is not implemented
            assert False, "The economy model is not provided!"

        self.emissions = OutputToEmissions(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            climate_ensembles=self.no_of_ensembles,
        )

        # Instantiate the SocialWelfareFunction class
        self.welfare_function = SocialWelfareFunction(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            population=self.economy.get_population(),  # TODO: This makes welfare function dependent on economy model
            elasticity_of_marginal_utility_of_consumption=self.elasticity_of_marginal_utility_of_consumption,
            pure_rate_of_social_time_preference=self.pure_rate_of_social_time_preference,
            inequality_aversion=self.inequality_aversion,
            sufficiency_threshold=self.sufficiency_threshold,
            egality_strictness=self.egality_strictness,
        )

        # Get the fixed savings rate for model time horizon
        self.fixed_savings_rate = self.economy.get_fixed_savings_rate(
            self.time_horizon.model_time_horizon
        )

        ############################################################################################################################################################
        #
        #   Initializing the Output Data Structures
        #
        ############################################################################################################################################################

        # Create a data dictionary to store the data
        self.data = {
            "gross_economic_output": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
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
            "damage_fraction": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
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
            "welfare_regional_temporal": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "welfare_regional": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    self.no_of_ensembles,
                )
            ),
            "welfare_temporal": np.zeros(
                (
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "welfare": np.zeros((self.no_of_ensembles,)),
        }

    ############################################################################################################################################################
    #
    #   End of JUSTICE Initialization and Instantiation
    #
    ############################################################################################################################################################

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)

    ############################################################################################################################################################
    #
    #   JUSTICE Model Stepwise Run
    #
    ############################################################################################################################################################
    def stepwise_run(
        self,
        emission_control_rate,
        timestep,
        savings_rate=None,
        endogenous_savings_rate=False,
    ):
        """
        This method is used for EMODPS & Reinforcement Learning (RL) applications.

        Run the model timestep by timestep and return the outcomes every timestep

        @param timestep: The timestep to run the model for 0 to model_time_horizon
        @param savings_rate: The savings rate for each timestep. So shape will be (no_of_regions,)
        @param emission_control_rate: The emissions control rate for each timestep. So shape will be (no_of_regions,)
        """

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
            # scenario=self.scenario,
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

            self.data["regional_temperature"][:, (timestep + 1), :] = (
                self.downscaler.get_regional_temperature(
                    self.data["global_temperature"][(timestep + 1), :]
                )
            )

            damage_fraction = self.damage_function.calculate_damage(
                temperature=self.data["regional_temperature"][:, timestep, :],
                timestep=timestep,
            )

            # Storing the damage fraction for the timestep
            self.data["damage_fraction"][:, timestep, :] = damage_fraction

            # Abatement cost is only dependent on the emission control rate
            abatement_cost = self.abatement.calculate_abatement(
                scenario=self.scenario,
                timestep=timestep,
                emission_control_rate=emission_control_rate,
            )

            # This applies damages and abatement costs and triggers the Investment & Capital Calculation
            # NOTE This is necessary to calculate the capital and investment for the next timestep
            # It closes the loop of the economy model
            self.economy.feedback_loop_for_economic_output(
                timestep=timestep,
                savings_rate=self.savings_rate[:, timestep],
                damage_fraction=damage_fraction,
                abatement=abatement_cost,
            )

            # Store the net economic output for the timestep
            self.data["net_economic_output"][:, timestep, :] = (
                self.economy.get_net_output_by_timestep(timestep)
            )
        elif timestep == (
            len(self.time_horizon.model_time_horizon) - 1
        ):  # Last timestep
            # No need to calculate temperature for the last timestep because current emissions produce temperature in the next timestep
            # Calculate damage for the last timestep
            damage_fraction = self.damage_function.calculate_damage(
                temperature=self.data["regional_temperature"][:, timestep, :],
                timestep=timestep,
            )

            # Storing the damage fraction for the timestep
            self.data["damage_fraction"][:, timestep, :] = damage_fraction

            # Calculate the abatement cost
            abatement_cost = self.abatement.calculate_abatement(
                scenario=self.scenario,
                timestep=timestep,
                emission_control_rate=emission_control_rate,
            )

            # This applies damages and abatement costs and triggers the Investment & Capital Calculation
            # NOTE This is necessary to calculate the capital and investment for the next timestep
            # It closes the loop of the economy model
            self.economy.feedback_loop_for_economic_output(
                timestep=timestep,
                savings_rate=self.savings_rate[:, timestep],
                damage_fraction=damage_fraction,
                abatement=abatement_cost,
            )

        # Save the data
        self.data["gross_economic_output"] = self.economy.get_gross_output()
        self.data["net_economic_output"] = self.economy.get_net_output()
        self.data["economic_damage"][:, timestep, :] = (self.economy.get_damages())[
            :, timestep, :
        ]
        self.data["abatement_cost"][:, timestep, :] = (self.economy.get_abatement())[
            :, timestep, :
        ]
        self.data["consumption"][:, timestep, :] = (
            self.economy.calculate_consumption_per_timestep(
                self.savings_rate[:, timestep], timestep
            )
        )
        self.data["consumption_per_capita"][:, timestep, :] = (
            self.economy.get_consumption_per_capita_per_timestep(
                self.savings_rate[:, timestep], timestep
            )
        )

    ############################################################################################################################################################
    #
    #   JUSTICE Model Run for the Entire Time Horizon
    #
    ############################################################################################################################################################

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
            gross_output = self.economy.run(
                timestep=timestep,
                savings_rate=self.savings_rate[:, timestep],
            )

            self.data["emissions"][:, timestep, :] = self.emissions.run(
                timestep=timestep,
                scenario=self.scenario,
                output=gross_output,
                emission_control_rate=self.emission_control_rate[:, timestep, :],
            )

            # Run the model for all timesteps except the last one
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

                # Emission in the current timestep produces temperature in the next timestep
                self.data["global_temperature"][(timestep + 1), :] = (
                    self.climate.compute_temperature_from_emission(
                        timestep, self.data["emissions"][:, timestep, :]
                    )
                )

                self.data["regional_temperature"][:, (timestep + 1), :] = (
                    self.downscaler.get_regional_temperature(
                        self.data["global_temperature"][(timestep + 1), :]
                    )
                )

                # Damages is calculated based on current temperature
                damage_fraction = self.damage_function.calculate_damage(
                    temperature=self.data["regional_temperature"][:, timestep, :],
                    timestep=timestep,
                )

                # Storing the damage fraction for the timestep
                self.data["damage_fraction"][:, timestep, :] = damage_fraction

                # TODO: Incomplete Implementation
                # carbon_price = self.abatement.calculate_carbon_price(
                #     timestep=timestep,
                #     emission_control_rate=self.emissions_control_rate[:, timestep],
                # )

                # Abatement cost is only dependent on the emission control rate
                abatement_cost = self.abatement.calculate_abatement(
                    scenario=self.scenario,
                    timestep=timestep,
                    emission_control_rate=self.emission_control_rate[:, timestep, :],
                )

                # This applies damages and abatement costs and triggers the Investment & Capital Calculation
                # NOTE This is necessary to calculate the capital and investment for the next timestep
                # It closes the loop of the economy model
                self.economy.feedback_loop_for_economic_output(
                    timestep=timestep,
                    savings_rate=self.savings_rate[:, timestep],
                    damage_fraction=damage_fraction,
                    abatement=abatement_cost,
                )

                # Store the net economic output for the timestep
                self.data["net_economic_output"][:, timestep, :] = (
                    self.economy.get_net_output_by_timestep(timestep)
                )
            elif timestep == (
                len(self.time_horizon.model_time_horizon) - 1
            ):  # Last timestep
                # No need to calculate temperature for the last timestep because current emissions produce temperature in the next timestep
                # Calculate damage for the last timestep
                damage_fraction = self.damage_function.calculate_damage(
                    temperature=self.data["regional_temperature"][:, timestep, :],
                    timestep=timestep,
                )

                # Storing the damage fraction for the timestep
                self.data["damage_fraction"][:, timestep, :] = damage_fraction

                # Calculate the abatement cost
                abatement_cost = self.abatement.calculate_abatement(
                    scenario=self.scenario,
                    timestep=timestep,
                    emission_control_rate=self.emission_control_rate[:, timestep, :],
                )

                # This applies damages and abatement costs and triggers the Investment & Capital Calculation
                # NOTE This is necessary to calculate the capital and investment for the next timestep
                # It closes the loop of the economy model
                self.economy.feedback_loop_for_economic_output(
                    timestep=timestep,
                    savings_rate=self.savings_rate[:, timestep],
                    damage_fraction=damage_fraction,
                    abatement=abatement_cost,
                )
        # Loading the consumption and consumption per capita from the economy model
        self.data["gross_economic_output"] = self.economy.get_gross_output()
        self.data["net_economic_output"] = self.economy.get_net_output()
        self.data["consumption"] = self.economy.calculate_consumption(
            savings_rate=self.savings_rate
        )
        self.data["consumption_per_capita"] = self.economy.get_consumption_per_capita(
            savings_rate=self.savings_rate,
        )

        self.data["economic_damage"] = self.economy.get_damages()
        self.data["abatement_cost"] = self.economy.get_abatement()

    ############################################################################################################################################################
    #
    #   JUSTICE Model Stepwise Evaluation
    #
    ############################################################################################################################################################
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

        # TODO: Return step by step individual data/observations
        # TODO: FOr RL, we have to separate obeservation and rewards

        (
            self.data["disentangled_utility"][:, timestep, :],
            self.data["welfare_regional_temporal"][:, timestep, :],
            self.data["welfare_temporal"][timestep, :],
        ) = self.welfare_function.calculate_stepwise_welfare(
            consumption_per_capita=self.data["consumption_per_capita"][:, timestep, :],
            timestep=timestep,
        )

        # Last timestep. Welfare_utilitarian_regional and welfare_utilitarian are calculated only for the last timestep
        if timestep == (len(self.time_horizon.model_time_horizon) - 1):
            (
                self.data["disentangled_utility"],
                self.data["welfare_regional_temporal"],
                self.data["welfare_temporal"],
                self.data["welfare_regional"],
                self.data["welfare"],
            ) = self.welfare_function.calculate_welfare(
                consumption_per_capita=self.data["consumption_per_capita"]
            )
        return self.data

    ############################################################################################################################################################
    #
    #   JUSTICE Model Evaluation for entire Time Horizon
    #
    ############################################################################################################################################################
    def evaluate(
        self,
    ):
        """
        Evaluate the model.
        """

        (
            self.data["disentangled_utility"],
            self.data["welfare_regional_temporal"],
            self.data["welfare_temporal"],
            self.data["welfare_regional"],
            self.data["welfare"],
        ) = self.welfare_function.calculate_welfare(
            consumption_per_capita=self.data["consumption_per_capita"]
        )
        return self.data

    def get_outcome_names(self):
        """
        Get the list of outcomes of the model.
        """
        return self.data.keys()

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
