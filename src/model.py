"""
This is the main JUSTICE model.
"""

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.model_time import TimeHorizon
from src.economy.neoclassical import NeoclassicalEconomyModel
from src.emissions.emission import OutputToEmissions
from src.damage.kalkuhl import DamageKalkuhl
from src.climate.coupled_fair import CoupledFAIR
from src.climate.temperature_downscaler import TemperatureDownscaler
from src.abatement.abatement_enerdata import AbatementEnerdata

# from src import utils


class JUSTICE:
    def __init__(self, start_year, end_year, timestep, scenario):
        """
        Initialize the model.
        """

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

        self.damage_function = DamageKalkuhl(
            input_dataset=self.data_loader,
            time_horizon=self.time_horizon,
            climate_ensembles=self.no_of_ensembles,
        )
        self.abatement = AbatementEnerdata(
            input_dataset=self.data_loader, time_horizon=self.time_horizon
        )

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

    def run(self, savings_rate, emissions_control_rate):
        """
        Run the model.
        """
        for timestep in range(len(self.time_horizon.model_time_horizon)):
            output = self.economy.run(
                scenario=self.scenario,
                timestep=timestep,
                savings_rate=savings_rate[:, timestep],
            )

            self.emissions_array = self.emissions.run_emissions(
                timestep=timestep,
                scenario=self.scenario,
                output=output,
                emission_control_rate=emissions_control_rate[:, timestep],
            )

            # Run the model for all timesteps except the last one. Damages and Abatement applies to the next timestep
            if timestep < (len(self.time_horizon.model_time_horizon) - 1):
                global_temperature = self.climate.compute_temperature_from_emission(
                    timestep, self.emissions_array
                )

                regional_temperature = self.downscaler.get_regional_temperature(
                    global_temperature
                )

                damage = self.damage_function.calculate_damage(
                    temperature=regional_temperature, timestep=timestep
                )

                abatement_cost = self.abatement.calculate_abatement(
                    timestep=timestep,
                    emissions=self.emissions_array,
                    emission_control_rate=emissions_control_rate[:, timestep],
                )

                # carbon_price = self.abatement.calculate_carbon_price(
                #     timestep=timestep,
                #     emission_control_rate=emissions_control_rate[:, timestep],
                # )

                self.economy.apply_damage_to_output(
                    timestep=timestep + 1, damage=damage
                )
                self.economy.apply_abatement_to_output(
                    timestep=timestep + 1, abatement=abatement_cost
                )
