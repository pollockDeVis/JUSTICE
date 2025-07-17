"""
This is the main JUSTICE model.
"""

import numpy as np
import json
from typing import Any

from justice.util.data_loader import DataLoader
from justice.util.enumerations import (
    Economy,
    DamageFunction,
    Abatement,
    WelfareFunction,
)
from justice.util.model_time import TimeHorizon
from justice.economy.neoclassical import NeoclassicalEconomyModel
from justice.economy.green import GreenEconomyModel
from justice.emissions.emission import OutputToEmissions
from justice.damage.kalkuhl import DamageKalkuhl
from justice.climate.coupled_fair import CoupledFAIR
from justice.climate.temperature_downscaler import TemperatureDownscaler
from justice.abatement.abatement_enerdata import AbatementEnerdata
from justice.welfare.social_welfare_function import SocialWelfareFunction
from config.default_parameters import SocialWelfareDefaults


class JUSTICE:
    """
    This is the JUSTICE model.

    This class implements the integrated assessment model JUSTICE.
    It performs extensive initialization by loading datasets, configuring climate components, setting up
    economic and damage sub-models, and initializing output data structures. The model is designed to run
    simulations either stepwise or over an entire time horizon and provides evaluation methods to assess
    welfare metrics and economic indicators.

    Initialization Parameters:
        start_year (int): The beginning year for the simulation (default is 2015).
        end_year (int): The ending year for the simulation (default is 2300).
        timestep (int): The simulation timestep increment (default is 1).
        scenario (int): Identifier for the simulation scenario.
        climate_ensembles (int or list, optional): Specifies one or more climate ensemble indices used
            to initialize the climate model.
        economy_type: Determines the economy model to use (default: Economy.NEOCLASSICAL).
        damage_function_type: Specifies the type of damage function to apply (default: DamageFunction.KALKUHL).
        abatement_type: Specifies the abatement model (default: Abatement.ENERDATA).
        social_welfare_function: Sets the welfare function (default: WelfareFunction.UTILITARIAN).
        **kwargs: Additional keyword arguments. For example, 'social_welfare_function_type' can be provided
            to override the default welfare function type.

    Key Methods:
        stepwise_run(emission_control_rate, timestep, savings_rate=None, endogenous_savings_rate=False):
            Executes a single simulation step by updating emissions, temperature (both global and regional),
            damages, abatement costs, and economic outputs while closing the feedback loop in the economy model.

        run(emission_control_rate, savings_rate=None, endogenous_savings_rate=False):
            Runs the simulation over the entire time horizon by iteratively executing the model steps and
            updating all relevant outputs.

        stepwise_evaluate(timestep=0):
            Evaluates the model at a given timestep by calculating stepwise welfare and reward signals,
            with additional comprehensive welfare evaluation for the final timestep.

        evaluate():
            Performs an overall evaluation of the model, aggregating consumption, welfare, and damage data
            over the entire simulation period.

        reset() / reset_model():
            Resets the model’s output data structures and state to allow for a fresh simulation run.

        get_outcome_names():
            Returns a list of keys representing all outcome metrics stored in the model’s data dictionary.

    Usage Notes:
        - The heavy initialization performed in the constructor (__init__) should only be executed once per instance.
        - Ensure that any modifications to the model state are followed by a subsequent run or evaluation if needed.
        - The model relies on external components (e.g., data loaders, climate models, economic models) which
          must be correctly configured and available.

    """

    _instance = None  # class-level instance for caching heavy initialization

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(JUSTICE, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        start_year=2015,  # Model is only tested for start year 2015
        end_year=2300,  # Model is only tested for end year 2300
        timestep=1,  # Model is only tested for timestep 1
        scenario=0,
        climate_ensembles=None,
        stochastic_run=True,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=WelfareFunction.UTILITARIAN,
        clustering=False,
        cluster_level=None,
        **kwargs,
    ):
        # If already initialized, do not repeat heavy initialization.
        if hasattr(self, "_already_initialized") and self._already_initialized:
            return

        ############################################################################################################################################################
        #
        #   Instantiating the JUSTICE Model (Heavy Initialization)
        #
        ############################################################################################################################################################
        self.economy_type = economy_type
        self.damage_function_type = damage_function_type
        self.abatement_type = abatement_type
        self.scenario = scenario
        self.clustering = clustering
        self.cluster_level = cluster_level

        if "social_welfare_function_type" in kwargs:
            self.welfare_function_type = WelfareFunction.from_index(
                kwargs["social_welfare_function_type"]
            )
        else:
            self.welfare_function_type = social_welfare_function

        # Instantiate the TimeHorizon class # TODO: Need to do the data slicing here for different start and end years
        self.time_horizon = TimeHorizon(
            start_year=start_year, end_year=end_year, data_timestep=5, timestep=timestep
        )

        # Load the datasets by instantiating the DataLoader class
        self.data_loader = DataLoader(self.time_horizon)
        self.region_list = self.data_loader.REGION_LIST

        if self.clustering:
            if self.cluster_level == 12:
                with open("data/input/rice_12_regions_dict.json") as f:
                    rice_json = json.load(f)
            elif self.cluster_level == 5:
                with open("data/input/5_regions.json") as f:
                    rice_json = json.load(f)
            else:
                raise ValueError("Cluster level not supported")

            with open("data/input/rice50_regions_dict.json") as f:
                rice_50_json = json.load(f)

            self.clusters = list(rice_json.keys())

            region_list = self.region_list.tolist()

            region_to_index = {region: idx for idx, region in enumerate(region_list)}
            cluster_to_index = {
                cluster: idx for idx, cluster in enumerate(self.clusters)
            }

            # create a mapping from region to cluster
            self.country_to_cluster = {}
            for region, country_codes in rice_50_json.items():
                region_index = region_to_index.get(region)
                for code in country_codes:
                    for cluster, cluster_codes in rice_json.items():
                        if code in cluster_codes:
                            cluster_index = cluster_to_index[cluster]
                            self.country_to_cluster[region_index] = cluster_index
                            break  # Break the loop as soon as we find a match for efficiency

            # create an inverse mapping from cluster to region
            self.cluster_to_country = {}
            for region, cluster in self.country_to_cluster.items():
                if cluster not in self.cluster_to_country:
                    self.cluster_to_country[cluster] = []
                self.cluster_to_country[cluster].append(region)

        ############################################################################################################################################################
        #
        #   Instatiating the FAIR Climate Model
        #
        ############################################################################################################################################################
        self.climate = CoupledFAIR(ch4_method="Thornhill2021")
        self.downscaler = TemperatureDownscaler(input_dataset=self.data_loader)

        if climate_ensembles is not None:
            if isinstance(climate_ensembles, int):
                self.no_of_ensembles = self.climate.fair_justice_run_init(
                    time_horizon=self.time_horizon,
                    scenarios=self.scenario,
                    climate_ensembles=[climate_ensembles],
                    stochastic_run=stochastic_run,
                )
            else:
                self.no_of_ensembles = self.climate.fair_justice_run_init(
                    time_horizon=self.time_horizon,
                    scenarios=self.scenario,
                    climate_ensembles=climate_ensembles,
                    stochastic_run=stochastic_run,
                )
        else:
            self.no_of_ensembles = self.climate.fair_justice_run_init(
                time_horizon=self.time_horizon,
                scenarios=self.scenario,
                stochastic_run=stochastic_run,
            )

        ############################################################################################################################################################
        #
        #   Initializing the Data Structures & Loading the Default Parameters
        #
        ############################################################################################################################################################
        self.fixed_savings_rate = np.zeros(
            (
                len(self.data_loader.REGION_LIST),
                len(self.time_horizon.model_time_horizon),
            )
        )
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

        social_welfare_defaults = SocialWelfareDefaults()
        welfare_defaults = social_welfare_defaults.get_defaults(
            self.welfare_function_type.name
        )
        self.elasticity_of_marginal_utility_of_consumption = welfare_defaults[
            "elasticity_of_marginal_utility_of_consumption"
        ]
        self.pure_rate_of_social_time_preference = welfare_defaults[
            "pure_rate_of_social_time_preference"
        ]
        self.inequality_aversion = welfare_defaults["inequality_aversion"]
        self.sufficiency_threshold = welfare_defaults["sufficiency_threshold"]
        self.egality_strictness = welfare_defaults["egality_strictness"]
        self.risk_aversion = welfare_defaults["risk_aversion"]
        self.limitarian_threshold_emissions = welfare_defaults[
            "limitarian_threshold_emissions"
        ]
        self.limitarian_start_year_of_remaining_budget = welfare_defaults[
            "limitarian_start_year_of_remaining_budget"
        ]

        ############################################################################################################################################################
        #
        #   Instantiating the Model Blocks
        #
        ############################################################################################################################################################
        if self.damage_function_type == DamageFunction.KALKUHL:
            self.damage_function = DamageKalkuhl(
                input_dataset=self.data_loader,
                time_horizon=self.time_horizon,
                climate_ensembles=self.no_of_ensembles,
            )
        else:
            assert False, "The damage function is not provided!"

        if self.abatement_type == Abatement.ENERDATA:
            self.abatement = AbatementEnerdata(
                input_dataset=self.data_loader,
                time_horizon=self.time_horizon,
                scenario=self.scenario,
            )
        else:
            assert False, "The abatement model is not provided!"

        if self.economy_type == Economy.NEOCLASSICAL:
            self.economy = NeoclassicalEconomyModel(
                input_dataset=self.data_loader,
                time_horizon=self.time_horizon,
                scenario=self.scenario,
                climate_ensembles=self.no_of_ensembles,
                elasticity_of_marginal_utility_of_consumption=self.elasticity_of_marginal_utility_of_consumption,
                pure_rate_of_social_time_preference=self.pure_rate_of_social_time_preference,
            )
        elif self.economy_type == Economy.GREEN:
            self.economy = GreenEconomyModel(
                input_dataset=self.data_loader,
                time_horizon=self.time_horizon,
                scenario=self.scenario,
                climate_ensembles=self.no_of_ensembles,
                elasticity_of_marginal_utility_of_consumption=self.elasticity_of_marginal_utility_of_consumption,
                pure_rate_of_social_time_preference=self.pure_rate_of_social_time_preference,
            )
        else:
            assert False, "The economy model is not provided!"

        self.emissions = OutputToEmissions(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            scenario=self.scenario,
            climate_ensembles=self.no_of_ensembles,
        )

        self.welfare_function = SocialWelfareFunction(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            climate_ensembles=self.no_of_ensembles,
            population=self.economy.get_population(),  # FIXME: This makes welfare function dependent on economy model
            risk_aversion=self.risk_aversion,
            elasticity_of_marginal_utility_of_consumption=self.elasticity_of_marginal_utility_of_consumption,
            pure_rate_of_social_time_preference=self.pure_rate_of_social_time_preference,
            inequality_aversion=self.inequality_aversion,
            sufficiency_threshold=self.sufficiency_threshold,
            egality_strictness=self.egality_strictness,
            limitarian_threshold_emissions=self.limitarian_threshold_emissions,
            limitarian_start_year_of_remaining_budget=self.limitarian_start_year_of_remaining_budget,
        )

        self.fixed_savings_rate = self.economy.get_fixed_savings_rate(
            self.time_horizon.model_time_horizon
        )

        ############################################################################################################################################################
        #
        #   Initializing the Output Data Structures
        #
        ############################################################################################################################################################
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
            "damage_cost_per_capita": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "abatement_cost_per_capita": np.zeros(
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
            "states_aggregated_consumption_per_capita": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                )
            ),
            "spatially_aggregated_welfare": np.zeros(
                (len(self.time_horizon.model_time_horizon),)
            ),
            "stepwise_marl_reward": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                )
            ),
            "temporally_disaggregated_welfare": np.zeros(
                (len(self.time_horizon.model_time_horizon),)
            ),
            "welfare": np.zeros((1,)),
        }

        # Flag heavy initialization as complete.
        self._already_initialized = True

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

        if endogenous_savings_rate:
            self.savings_rate[:, timestep] = self.fixed_savings_rate[:, timestep]
        elif not self.clustering:
            self.savings_rate[:, timestep] = savings_rate
        else:
            for (
                key,
                value,
            ) in self.country_to_cluster.items():  # [('region', 'cluster')]
                self.savings_rate[key, timestep] = savings_rate[value]
            savings_rate = self.savings_rate[:, timestep]

        # Check the shape of the emission_control_rate whether it is 1D or 2D
        if len(emission_control_rate.shape) == 1:
            emission_control_rate = np.tile(
                emission_control_rate[:, np.newaxis], (1, self.no_of_ensembles)
            )

        if not self.clustering:
            self.emission_control_rate[:, timestep, :] = emission_control_rate

        else:
            for (
                key,
                value,
            ) in self.country_to_cluster.items():  # [('region', 'cluster')]
                self.emission_control_rate[key, timestep, :] = emission_control_rate[
                    value
                ]
            emission_control_rate = self.emission_control_rate[:, timestep, :]

        self.emission_control_rate[:, timestep, :] = emission_control_rate

        gross_output = self.economy.run(
            # scenario=self.scenario,
            timestep=timestep,
            savings_rate=self.savings_rate[:, timestep],
        )

        self.data["emissions"][:, timestep, :] = self.emissions.run(
            timestep=timestep,
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
        self.data["damage_cost_per_capita"][:, timestep, :] = (
            self.economy.get_damage_cost_per_capita_per_timestep(timestep)
        )
        self.data["abatement_cost_per_capita"][:, timestep, :] = (
            self.economy.get_abatement_cost_per_capita_per_timestep(timestep)
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
        if endogenous_savings_rate:
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
        self.data["damage_cost_per_capita"] = self.economy.get_damage_cost_per_capita()
        self.data["abatement_cost_per_capita"] = (
            self.economy.get_abatement_cost_per_capita()
        )

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
        # TODO: For RL, we have to separate obeservation and rewards

        (
            self.data["spatially_aggregated_welfare"][timestep],
            self.data["stepwise_marl_reward"][:, timestep],
        ) = self.welfare_function.calculate_stepwise_welfare(
            consumption_per_capita=self.data["consumption_per_capita"][:, timestep, :],
            timestep=timestep,
        )

        # Last timestep. Welfare_utilitarian_regional and welfare_utilitarian are calculated only for the last timestep
        if timestep == (len(self.time_horizon.model_time_horizon) - 1):

            (
                self.data["states_aggregated_consumption_per_capita"],
                self.data["spatially_aggregated_welfare"],
                self.data["temporally_disaggregated_welfare"],
                self.data["welfare"],
            ) = self.welfare_function.calculate_welfare(
                consumption_per_capita=self.data["consumption_per_capita"],
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
            self.data["states_aggregated_consumption_per_capita"],
            self.data["spatially_aggregated_welfare"],
            self.data["temporally_disaggregated_welfare"],
            self.data["welfare"],
        ) = self.welfare_function.calculate_welfare(
            consumption_per_capita=self.data["consumption_per_capita"],
            emissions=self.data["emissions"],
        )
        return self.data

    def reset(self):
        """
        Reset the model to the initial state by setting the data dictionary to zero.
        """

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
            "damage_cost_per_capita": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                    self.no_of_ensembles,
                )
            ),
            "abatement_cost_per_capita": np.zeros(
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
            "states_aggregated_consumption_per_capita": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                )
            ),
            "spatially_aggregated_welfare": np.zeros(
                (len(self.time_horizon.model_time_horizon),)
            ),
            "stepwise_marl_reward": np.zeros(
                (
                    len(self.data_loader.REGION_LIST),
                    len(self.time_horizon.model_time_horizon),
                )
            ),
            "temporally_disaggregated_welfare": np.zeros(
                (len(self.time_horizon.model_time_horizon),)
            ),
            "welfare": np.zeros((1,)),
        }

        self.climate.reset()
        self.economy.reset()
        self.emissions.reset()
        self.damage_function.reset()

    # Added for EMA Workbench Support
    def reset_model(self):
        self.reset()

    @classmethod
    def hard_reset(cls):
        """
        Delete the cached singleton instance so next instantiation creates a clean new instance.
        """
        if cls._instance is not None:
            # Optionally call any cleanup on the existing instance before deleting
            # e.g. cls._instance.cleanup()
            cls._instance = None

    def get_outcome_names(self):
        """
        Get the list of outcomes of the model.
        """
        return self.data.keys()
