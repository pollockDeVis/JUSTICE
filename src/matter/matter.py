"""
This file contains the matter-use part of the JUSTICE model. 
this module is inspired in the DEFINE-MATTER (Dafermos, 2021) set of equations
"""

from typing import Any
from scipy.interpolate import interp1d
import numpy as np
import copy
from config.default_parameters import EconomicSubModules, EmissionsAvoidedDefaults, RecyclingCostsIncomeLevels
from src.util.enumerations import get_economic_scenario


class MatterUse:
    """
    This class describes the matter-use dynamics in the JUSTICE model.
    """

    def __init__(
        self,
        input_dataset,
        time_horizon,
        climate_ensembles,
        scenario,
    ):

        # Load the defaults 
        matter_defaults = EconomicSubModules().get_defaults("MATTER")
        self.emissions_defaults = EmissionsAvoidedDefaults().get_defaults()
        self.recycling_defaults = RecyclingCostsIncomeLevels().get_defaults()

        # Parameters
        self.physical_use_ratio = matter_defaults["physical_use_ratio"]
        self.discard_rate = matter_defaults["discard_rate"]
        self.conversion_rate_material_reserves = matter_defaults[
            "conversion_rate_material_reserves"
        ]
        self.recycling_rate = matter_defaults["recycling_rate"]

        # Saving the climate ensembles adn scenario
        self.NUM_OF_ENSEMBLES = climate_ensembles
        self.scenario = get_economic_scenario(scenario)

        #Input data
        self.region_list = input_dataset.REGION_LIST
        self.material_intensity_array = copy.deepcopy(
            input_dataset.MATERIAL_INTENSITY_ARRAY
        )
        self.income_level = input_dataset.INCOME_LEVEL_ARRAY

        #Time horizon
        #Time horizon
        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Selecting only the required scenario
        self.material_intensity_array = self.material_intensity_array[
            :, :, self.scenario
        ]

        if self.timestep != self.data_timestep:
            # Interpolate Material Intensity Dictionary
            self._interpolate_material_intensity()

        # Intializing the material intensity array DMC/GDP
        self.material_intensity = self.material_intensity_array

        # Intializing initial variables values (57,1)
        self.in_use_stock_init = input_dataset.IN_USE_STOCK_INIT_ARRAY
        self.material_reserves_init = input_dataset.MATERIAL_RESERVES_INIT_ARRAY
        self.material_resources_init = input_dataset.MATERIAL_RESOURCES_INIT_ARRAY

        # Initialize MATTER variables
        self.material_consumption = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.in_use_stock = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.discarded_material = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.recycled_material = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.waste = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.extracted_matter = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.converted_material_reserves = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.material_reserves = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.material_resources = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.depletion_ratio = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.emissions_avoided = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        # Initializing recycling costs array
        self.recycling_cost = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

    def stepwise_run(self, timestep, output, recycling_rate):
        """
        Run the matter-use calculations for a given timestep,
        output shape (57, 1001)
        """
        if len(recycling_rate.shape) == 1:
            recycling_rate = recycling_rate.reshape(-1, 1)

        material_consumption = (
            (self.material_intensity[:, timestep]).reshape(-1, 1) * output
        )
        
        # Calculate the part of material consumption that goes to in-use stock
        material_to_in_use_stock = material_consumption * self.physical_use_ratio

        if timestep == 0:
            self.in_use_stock[:, timestep, :] = self.in_use_stock_init  # [:, 0]
            in_use_stock = self.in_use_stock[:, timestep, :]
            discarded_material = self.discard_rate * in_use_stock
            recycled_material= recycling_rate * discarded_material
            waste = discarded_material - recycled_material
            extracted_matter = material_consumption - recycled_material
            self.material_resources[:,timestep,:] = self.material_resources_init
            material_resources = self.material_resources[:, timestep, :]
            converted_material_reserves = self.conversion_rate_material_reserves * material_resources
            self.material_reserves[:,timestep,:] = self.material_reserves_init
            material_reserves = self.material_reserves[:, timestep, :]
            depletion_ratio = extracted_matter/ (material_reserves + 1e-10) #avoid division to zero
            emissions_avoided = self.get_emissions_avoided(recycled_material)
            recycling_cost = self.calculate_recycling_cost()[:, timestep, :]
            self.recycling_cost[:, timestep, :] = recycling_cost * recycled_material
            total_costs= self.recycling_cost[:, timestep, :]

        else:# If timestep > 0
            in_use_stock = self.in_use_stock[:, timestep - 1, :]
            discarded_material = self.discard_rate * in_use_stock
            self.in_use_stock[:, timestep, :] = (
                in_use_stock + material_to_in_use_stock - discarded_material
            )
            in_use_stock = self.in_use_stock[:, timestep, :]
            recycled_material= recycling_rate * discarded_material
            waste = discarded_material - recycled_material
            extracted_matter = material_consumption - recycled_material
            material_resources = self.material_resources[:, timestep - 1, :]
            converted_material_reserves = (
                self.conversion_rate_material_reserves * material_resources
            )
            self.material_resources[:,timestep,:] = (
                material_resources - converted_material_reserves
            )
            material_resources = self.material_resources[:,timestep,:]
            material_reserves = self.material_reserves[:,timestep - 1,:]
            depletion_ratio = extracted_matter/ (material_reserves + 1e-10)
            self.material_reserves[:,timestep,:] =(
                material_reserves + converted_material_reserves - extracted_matter
            )
            material_reserves = self.material_reserves[:,timestep,:]
            emissions_avoided = self.get_emissions_avoided(recycled_material)
            recycling_cost = self.calculate_recycling_cost()[:, timestep, :]
            self.recycling_cost[:, timestep, :] = recycling_cost * recycled_material
            total_costs= self.recycling_cost[:, timestep, :]

        #Save data
        self.depletion_ratio[:, timestep, :] = depletion_ratio
        self.emissions_avoided[:, timestep, :] = emissions_avoided
        self.recycled_material[:, timestep, :] = recycled_material
        self.material_consumption[:, timestep, :] = material_consumption
        self.discarded_material[:, timestep, :] = discarded_material
        self.extracted_matter[:, timestep, :] = extracted_matter
        self.waste[:, timestep, :] = waste
        self.material_reserves[:,timestep,:] = material_reserves
        self.material_resources[:,timestep,:] = material_resources
        self.recycling_cost[:,timestep,:] = total_costs


        return (depletion_ratio, emissions_avoided, material_reserves,
            recycled_material, material_consumption, discarded_material,
            extracted_matter, waste, material_resources, total_costs
        )
    def run(self, output, recycling_rate):
        """
        Run the matter-use calculations for the entire time horizon.
        """
        for timestep in range(len(self.model_time_horizon)):
            depletion_ratio, emissions_avoided_timestep, recycling_cost = self.stepwise_run(
                timestep, output, recycling_rate
            )
            self.depletion_ratio[:, timestep, :] = depletion_ratio
            self.emissions_avoided[:, timestep, :] = emissions_avoided_timestep
            self.recycling_cost[:, timestep, :] = recycling_cost
        return depletion_ratio, emissions_avoided_timestep, recycling_cost

    
    ########################################################################################
    # Emissions avoided through recycling of paper and plastics
    ########################################################################################

    def get_emissions_avoided(
        self, recycled_material
    ):  # Extract recycled material for the current timestep
        recycled_material_timestep = recycled_material[:, :]

        # Calculate proportions of recycled materials in gigatons (Gt)
        recycled_paper = (
            recycled_material_timestep * self.emissions_defaults["PROPORTION_PAPER"]
        )
        recycled_plastic = (
            recycled_material_timestep * self.emissions_defaults["PROPORTION_PLASTIC"]
        )
        
        # Calculate GHG emissions avoided
        em_ghg_avoided = self.calculate_ghg_avoided(recycled_paper, recycled_plastic)

        # Calculate energy savings
        e_total_saved = self.calculate_energy_saved(recycled_paper, recycled_plastic)
    
        # Calculate fuel saved and CO2 emissions avoided
        em_co2_avoided = self.calculate_co2_avoided(e_total_saved)

        # Total emissions avoided
        em_total = (
            (em_ghg_avoided + em_co2_avoided) * 365
        ) / 1e12  # Convert kg to Gt per year
        return em_total  # Gt per year

    def calculate_ghg_avoided(self, recycled_paper, recycled_plastic):
        # Calculate GHG emissions avoided for paper
        em_ghg_vg_paper = self.emissions_defaults["EFACTOR_VG_PAPER"] * recycled_paper
        em_ghg_rec_paper = self.emissions_defaults["EFACTOR_REC_PAPER"] * recycled_paper
        # Calculate GHG emissions avoided for plastic
        em_ghg_vg_plastic = (
            self.emissions_defaults["EFACTOR_VG_PLASTIC"] * recycled_plastic
        )
        em_ghg_rec_plastic = (
            self.emissions_defaults["EFACTOR_REC_PLASTIC"] * recycled_plastic
        )
        # Total GHG emissions avoided
        em_ghg_avoided_paper = em_ghg_vg_paper - em_ghg_rec_paper
        em_ghg_avoided_plastic = em_ghg_vg_plastic - em_ghg_rec_plastic

        return em_ghg_avoided_paper + em_ghg_avoided_plastic

    def calculate_energy_saved(self, recycled_paper, recycled_plastic):
        # Calculate energy savings for paper
        e_vg_paper = recycled_paper * self.emissions_defaults["ENERGY_FACTOR_VG_PAPER"]
        e_rec_paper = (
            recycled_paper * self.emissions_defaults["ENERGY_FACTOR_REC_PAPER"]
        )
        # Calculate energy savings for plastic
        e_vg_plastic = (
            recycled_plastic * self.emissions_defaults["ENERGY_FACTOR_VG_PLASTIC"]
        )
        e_rec_plastic = (
            recycled_plastic * self.emissions_defaults["ENERGY_FACTOR_REC_PLASTIC"]
        )
        # Total energy saved
        e_total_saved_paper = e_vg_paper - e_rec_paper
        e_total_saved_plastic = e_vg_plastic - e_rec_plastic

        return (e_total_saved_paper + e_total_saved_plastic) / self.emissions_defaults[
            "CONVERSION_FACTOR_GJ_TON"
        ]

    def calculate_co2_avoided(self, e_total_saved):
        fuel_saved = (e_total_saved) / (
            self.emissions_defaults["GENERATOR_EFFICIENCY"]
            * self.emissions_defaults["LOWER_HEATING_VALUE"]
        )
        return fuel_saved * self.emissions_defaults["EMISSION_FACTOR_DIESEL"]

    ##################################################################################
    # Recycling cost based on income level calculation
    ##################################################################################
    def calculate_recycling_cost(self):
        # Convert income levels from bytes to strings
        income_levels = self.income_level.flatten().astype(str)  # (57,)
        
        # Normalize income level strings to match keys in recycling_defaults
        income_level_mapping = {
            'Low income': 'LOW_INCOME',
            'Lower middle income': 'LOWER_MIDDLE_INCOME',
            'Upper middle income': 'UPPER_MIDDLE_INCOME',
            'High income': 'HIGH_INCOME'
        }
        max_costs = np.zeros_like(income_levels, dtype=float)
        min_costs = np.zeros_like(income_levels, dtype=float)

        # Map income levels to costs
        for i, income_level in enumerate(income_levels):
            normalized_level = income_level_mapping[income_level]
            min_costs[i], max_costs[i] = self.recycling_defaults[normalized_level]

        # Calculate linearly decreasing costs
        num_timesteps = len(self.model_time_horizon)
        timesteps = np.arange(num_timesteps)

        # Reshape for broadcasting
        max_costs = max_costs[:, np.newaxis]
        min_costs = min_costs[:, np.newaxis]

        # Linearly interpolate costs over time
        cost_slope = (max_costs - min_costs) / (num_timesteps - 1)
        recycling_costs = max_costs - cost_slope * timesteps

        # Apply the costs across the ensembles
        recycling_costs = recycling_costs[:, :, np.newaxis]

        return recycling_costs

    def _interpolate_material_intensity(self):
        interp_data = np.zeros(
            (
                self.material_intensity_array.shape[0],
                len(self.model_time_horizon),
                # self.gdp_array.shape[2],
            )
        )
        for i in range(self.material_intensity_array.shape[0]):

            if len(self.data_time_horizon) != self.material_intensity_array.shape[1]:
                raise ValueError(
                    f"Mismatch between data time horizon length ({len(self.data_time_horizon)}) and material intensity array length ({self.material_intensity_array.shape[1]})."
                )

            f = interp1d(
                self.data_time_horizon,
                self.material_intensity_array[i, :],
                kind="linear",
                fill_value="extrapolate",
            )
            interp_data[i, :] = f(self.model_time_horizon)  # , j

        self.material_intensity_array = interp_data

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
