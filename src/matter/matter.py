"""
This file contains the matter-use part of the JUSTICE model. 
this module is inspired in the DEFINE-MATTER (Dafermos, 2021) set of equations
"""

from typing import Any
from scipy.interpolate import interp1d
import numpy as np
import copy
from config.default_parameters import EconomicSubModules, EmissionsAvoidedDefaults
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

    

        # Parameters
        self.physical_use_ratio = matter_defaults["physical_use_ratio"]
        self.discard_rate = matter_defaults["discard_rate"]
        self.conversion_rate_material_reserves = matter_defaults[
            "conversion_rate_material_reserves"
        ]
        self.recycling_rate = matter_defaults["recycling_rate"]

        # Saving the climate ensembles and scenario
        self.NUM_OF_ENSEMBLES = climate_ensembles
        self.scenario = get_economic_scenario(scenario)

        #Load input data
        self.region_list = input_dataset.REGION_LIST
        self.material_intensity_array = copy.deepcopy(
            input_dataset.MATERIAL_INTENSITY_ARRAY
        )
        self.income_level = input_dataset.INCOME_LEVEL_ARRAY

        #Time horizon
        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Selecting only the required scenario
        self.material_intensity = self.material_intensity_array[
            :, :, self.scenario
        ]

        if self.timestep != self.data_timestep:
            # Interpolate Material Intensity Dictionary
            self._interpolate_material_intensity()


        # Intializing the initial values (57,1)
        self.in_use_stock_init = input_dataset.IN_USE_STOCK_INIT_ARRAY
        self.material_reserves_init = input_dataset.MATERIAL_RESERVES_INIT_ARRAY
        self.material_resources_init = input_dataset.MATERIAL_RESOURCES_INIT_ARRAY

        # Initialize matter variables 
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
        self.recycling_costs = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
    #  Angela - if you are taking timestep as an argument, change the name to stepwise_run. You can create a run method that will just call the stepwise_run for the entire time horizon

    def stepwise_run(self, timestep, output, recycling_rate):
        """
        Run the matter-use calculations for a given timestep,
        output shape (57, 1001)
        """
        if len(recycling_rate.shape) == 1:
            recycling_rate = recycling_rate.reshape(-1, 1)

        material_consumption = ((self.material_intensity[:, timestep]).reshape(-1, 1) * output)
        print(f"Material Consumption shape: {material_consumption.shape}")

        if timestep == 0:
            # in_use_stock = self.in_use_stock_init
            self.in_use_stock[:, timestep, :] = self.in_use_stock_init  # [:, 0]
            in_use_stock = self.in_use_stock[:, timestep, :]

            self.material_reserves[:, timestep, :] = (
                self.material_reserves_init
            )  # [:, 0]
            material_reserves = self.material_reserves[:, timestep, :]

            self.material_resources[:, timestep, :] = (
                self.material_resources_init
            )  # [:, 0]
            material_resources = self.material_resources[:, timestep, :]

            # Convert to 1D array
            # in_use_stock = in_use_stock.flatten()
            print(f"In-use stock shape at t=0: {in_use_stock.shape}")

            # in_use_stock = np.repeat(
            #     self.in_use_stock_init, output.shape[1], axis=1
            # )  # self.in_use_stock[:, timestep, :]

            print(f"In-use stock shape after repeat: {in_use_stock.shape}")
            # material_reserves = np.repeat(
            #     self.material_reserves_init, output.shape[1], axis=1
            # )
            # material_resources = np.repeat(
            #     self.material_resources_init, output.shape[1], axis=1
            # )
            print(f"In-use stock shape: {in_use_stock.shape}")
        else:  # If timestep > 0
            in_use_stock = self.in_use_stock[:, timestep - 1, :]
            material_resources = self.material_resources[:, timestep - 1, :]
            material_reserves = self.material_reserves[:, timestep - 1, :]
            print(f"In-use stock shape: {in_use_stock.shape}")

        discarded_material = self.discard_rate * in_use_stock
        print(f"Discarded material shape: {discarded_material.shape}")

        # Calculate in-use stock for the current timestep
        self.in_use_stock[:, timestep, :] = (
            in_use_stock + material_consumption - discarded_material
        )
        in_use_stock = self.in_use_stock[:, timestep, :]

        print(f"In-use stock (updated) shape: {in_use_stock.shape}")

        recycled_material = recycling_rate * discarded_material
        print(f"Recycled material shape: {recycled_material.shape}")

        waste = discarded_material - recycled_material
        print(f"Waste shape: {waste.shape}")

        extracted_matter = material_consumption - recycled_material
        print(f"Extracted matter shape: {extracted_matter.shape}")

        converted_material_reserves = (
            self.conversion_rate_material_reserves * material_resources
        )
        print(f"Converted material reserves shape: {converted_material_reserves.shape}")

        material_reserves = (
            material_reserves + converted_material_reserves - extracted_matter
        )
        print(f"Material reserves shape: {material_reserves.shape}")

        material_resources = material_resources - converted_material_reserves
        print(f"Material resources shape: {material_resources.shape}")

        depletion_ratio = extracted_matter / (material_resources+ 1e-10)
        print(f"Depletion ratio shape: {depletion_ratio.shape}")

        # Emissions avoided by the amount of recycled material
        emissions_avoided = self.get_emissions_avoided(timestep, recycled_material)

        # Storage data
        self.depletion_ratio[:, timestep, :] = depletion_ratio
        self.emissions_avoided[:, timestep, :] = emissions_avoided
        self.recycled_material[:,timestep,:] = recycled_material
        self.material_consumption[:,timestep,:] = material_consumption
        self.discarded_material[:,timestep,:]= discarded_material
        self.extracted_matter[:,timestep,:] = extracted_matter
        self.waste[:,timestep,:]= waste

        return depletion_ratio, emissions_avoided, recycled_material, material_consumption, discarded_material, extracted_matter, waste

        """

        # Calculate discarded material using previous in-use stock
        if timestep == 0:
            in_use_stock = np.repeat(
                self.in_use_stock_init[:, np.newaxis])
        else:
            in_use_stock = self.in_use_stock[:, timestep - 1, :]
        discarded_material = self.discard_rate * in_use_stock
        print(f"Discarded material shape: {discarded_material.shape}")
    
        # Calculate discarded material using previous in-use stock
        if timestep == 0:
            in_use_stock = self.in_use_stock_init[:,timestep]
        else:
            in_use_stock = self.in_use_stock[:, timestep - 1, :]
        
        discarded_material = self.discard_rate * in_use_stock
        print(f"Discarded material shape: {discarded_material.shape}")

        # Calculate in-use stock for the current timestep
        self.in_use_stock[:, timestep, :] = (
            in_use_stock +
            material_consumption -
            discarded_material
        )
        in_use_stock = self.in_use_stock[:, timestep, :]
        print(f"in use stock shape: {in_use_stock.shape}")# NOTE: testing point

        recycled_material = recycling_rate * discarded_material[:,timestep,:]
        print(f"recycled material shape: {recycled_material.shape}")# NOTE: testing point

        waste = discarded_material- recycled_material
        print(f"waste shape: {waste.shape}")# NOTE: testing point

        extracted_matter = self.get_extracted_matter(
            material_consumption, recycled_material, timestep)

        converted_material_reserves = self.get_converted_material_reserves(
            self.material_resources, timestep)

        material_reserves = self.get_material_reserves(
            extracted_matter, converted_material_reserves, timestep)

        material_resources = self.get_material_resources(
            converted_material_reserves, timestep)

        depletion_ratio = self.get_depletion_ratio(
            extracted_matter, material_resources, timestep) 

        
        # Emissions avoided by the amount of recycled material
        emissions_avoided = self.get_emissions_avoided(timestep, recycled_material)
        # Store the results
        self.depletion_ratio[:, timestep, :] = depletion_ratio
        self.emissions_avoided[:, timestep, :] = emissions_avoided

        print(f"Depletion Ratio shape: {depletion_ratio.shape}") #NOTE Testing point
        print(f"Emissions Avoided shape: {emissions_avoided.shape}")
        # Calculate recycling costs
        #recycling_costs = self.recycling_cost(recycled_material)

        return depletion_ratio, emissions_avoided  # , recycling_costs
        """

    def run(self, output, recycling_rate):
        """
        Run the matter-use calculations for the entire time horizon.
        """
        depletion_ratios = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        emissions_avoided = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        # recycling_costs = np.zeros(
        #    (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        # )
        for timestep in range(len(self.model_time_horizon)):
            depletion_ratio, emissions_avoided_timestep = self.stepwise_run(
                timestep, output, recycling_rate
            )
            depletion_ratios[:, timestep, :] = depletion_ratio
            emissions_avoided[:, timestep, :] = emissions_avoided_timestep
            # recycling_costs[:, timestep, :] = recycling_costs_timestep
        return depletion_ratios, emissions_avoided  # recycling_costs

    ############################################################################################################

    # Matter-use variable calculations functions

    ############################################################################################################

    # NOTE: if the following functions are only specific to this class, and not used anywhere else, you can use the decorator @classmethod to make them private
    def get_in_use_stock(self, material_consumption, discarded_material, timestep):
        if timestep == 0:
            # Initialize the in-use stock for the first time step
            self.in_use_stock[:, timestep, :] = self.in_use_stock_init
        else:
            # Update the in-use stock for subsequent time steps
            self.in_use_stock[:, timestep, :] = (
                self.in_use_stock[:, timestep - 1, :]
                + material_consumption[:, timestep, :]
                - discarded_material[:, timestep, :]
            )

        return self.in_use_stock

    def get_discarded_material(self, in_use_stock, timestep):
        self.discarded_material = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.discarded_material[:, timestep, :] = (
            self.discard_rate * in_use_stock[:, timestep - 1, :]
        )

        return self.discarded_material

    def get_recycled_material(self, discarded_material, recycling_rate, timestep):
        recycled_material = np.zeros_like(discarded_material)
        recycled_material = recycling_rate * discarded_material
        return recycled_material

    def get_waste(self, discarded_material, recycled_material, timestep):
        self.waste = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.waste[:, timestep, :] = (
            discarded_material[:, timestep, :] - recycled_material[:, timestep, :]
        )
        return self.waste

    def get_extracted_matter(self, material_consumption, recycled_material, timestep):
        self.extracted_matter = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.extracted_matter[:, timestep, :] = (
            material_consumption[:, timestep, :] - recycled_material[:, timestep, :]
        )
        return self.extracted_matter

    def get_converted_material_reserves(self, material_resources, timestep):
        self.converted_material_reserves = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        self.converted_material_reserves[:, timestep, :] = (
            self.conversion_rate_material_reserves - material_resources[:, timestep, :]
        )
        return self.converted_material_reserves

    def get_material_reserves(
        self, extracted_matter, converted_material_reserves, timestep
    ):
        if timestep == 0:
            # Initialize the in-use stock for the first time step
            self.material_reserves[:, timestep, :] = self.material_reserves_init
        else:
            # Update the in-use stock for subsequent time steps
            self.material_reserves[:, timestep, :] = (
                self.material_reserves[:, timestep - 1, :]
                + converted_material_reserves[:, timestep, :]
                - extracted_matter[:, timestep, :]
            )
        return self.material_reserves

    def get_material_resources(self, converted_material_reserves, timestep):
        if timestep == 0:
            # Initialize the in-use stock for the first time step
            self.material_resources[:, timestep, :] = self.material_resources_init
        else:
            # Update the in-use stock for subsequent time steps
            self.material_resources[:, timestep, :] = (
                self.material_resources[:, timestep - 1, :]
                - converted_material_reserves[:, timestep, :]
            )
        return self.material_resources

    def get_depletion_ratio(self, extracted_matter, material_resources, timestep):
        self.depletion_ratio[:, timestep, :] = (
            extracted_matter[:, timestep, :] / material_resources[:, timestep, :]
        )
        return self.depletion_ratio

    ########################################################################################
    # Emissions avoided through recycling of paper and plastics
    ########################################################################################

    def get_emissions_avoided(
        self, timestep, recycled_material
    ):  # Extract recycled material for the current timestep
        recycled_material_timestep = recycled_material[:, :]

        # Calculate proportions of recycled materials in gigatons (Gt)
        recycled_paper = (
            recycled_material_timestep * self.emissions_defaults["PROPORTION_PAPER"]
        )
        recycled_plastic = (
            recycled_material_timestep * self.emissions_defaults["PROPORTION_PLASTIC"]
        )
        print(f"Recycled Paper shape: {recycled_paper.shape}")
        print(f"Recycled Plastic shape: {recycled_plastic.shape}")

        # Calculate GHG emissions avoided
        em_ghg_avoided = self.calculate_ghg_avoided(recycled_paper, recycled_plastic)
        print(f"GHG Avoided shape: {em_ghg_avoided.shape}")

        # Calculate energy savings
        e_total_saved = self.calculate_energy_saved(recycled_paper, recycled_plastic)
        print(f"Total Energy Saved shape: {e_total_saved.shape}")

        # Calculate fuel saved and CO2 emissions avoided
        em_co2_avoided = self.calculate_co2_avoided(e_total_saved)
        print(f"CO2 Avoided shape: {em_co2_avoided.shape}")

        # Total emissions avoided
        em_total = (
            (em_ghg_avoided + em_co2_avoided) * 365
        ) / 1e12  # Convert kg to Gt per year
        print(f"Total Emissions Avoided shape: {em_total.shape}")

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
    """
    def linear_decrease_cost(self, min_cost, max_cost, timestep):
       
        start_year = self.model_time_horizon[0]
        end_year = self.model_time_horizon[-1]
        year = self.model_time_horizon[timestep]
        slope = (min_cost - max_cost) / (end_year - start_year)
        return max_cost + slope * (year - start_year)

    def recycling_cost(self, recycled_material):
        # TODO: Need to move this to default_parameters.py
        cost_ranges = {
            "Low income": (0 * 1e-3, 25 * 1e-3),
            "Lower middle income": (5 * 1e-3, 30 * 1e-3),
            "Upper middle income": (5 * 1e-3, 50 * 1e-3),
            "High income": (30 * 1e-3, 80 * 1e-3),
        }
        costs = np.zeros_like(recycled_material)
        for i in range(recycled_material.shape[0]):  # Loop over regions
            for j in range(recycled_material.shape[1]):  # Loop over timesteps
                for k in range(recycled_material.shape[2]):  # Loop over scenarios
                    income_level = self.income_level[i]
                    min_cost, max_cost = cost_ranges[income_level]
                    average_cost = self.linear_decrease_cost(min_cost, max_cost, j)
                    costs[i, j, k] = (
                        average_cost * recycled_material[i, j, k]
                    )  # Assuming recycled_material is in Gt
        return costs """

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
