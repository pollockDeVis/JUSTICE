"""
This file contains the matter-use part of the JUSTICE model. 
this module is inspired in the DEFINE-MATTER (Dafermos, 2021) set of equations
"""

from typing import Any
from scipy.interpolate import interp1d
import numpy as np
import copy
from config.default_parameters import EconomicSubModules, EmissionsAvoidedDefaults


class MatterUse:
    """
    This class describes the matter-use dynamics in the JUSTICE model.
    """

    def __init__(
        self,
        input_dataset,
        time_horizon,
        climate_ensembles,
        economy,
    ):

        # Load the defaults #TODO Angela - you can implement this
        matter_defaults = EconomicSubModules().get_defaults("MATTER")
        # Load the emissions avoided defaults
        emissions_defaults = EmissionsAvoidedDefaults().get_defaults()
        # Emissions defaults
        self.emissions_defaults = emissions_defaults

        # Load the instantiated economy model and set it as an attribute
        self.economy = economy

        # Parameters
        self.physical_use_ratio = matter_defaults["physical_use_ratio"]
        self.discard_rate = matter_defaults["discard_rate"]
        self.conversion_rate_material_reserves = matter_defaults[
            "conversion_rate_material_reserves"
        ]
        self.recycling_rate = matter_defaults["recycling_rate"]

        # Saving the climate ensembles ?
        self.NUM_OF_ENSEMBLES = climate_ensembles

        # Saving the scenario
        self.scenario = self.economy.scenario
        # self.scenario = get_economic_scenario(scenario)

        self.region_list = input_dataset.REGION_LIST
        self.material_intensity_array = copy.deepcopy(
            input_dataset.MATERIAL_INTENSITY_ARRAY
        )

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

        """
        Initialize matter-use variables arrays
        """

        # TODO: put the units of each variable in default_parameters.py
        # Intializing the material intensity array Unit: kg/USD per year
        self.material_intensity = self.material_intensity_array

        # Intializing the material intensity array Unit: Gt per year
        self.material_consumption = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the in-use stock array Unit: Gt per year
        self.in_use_stock = copy.deepcopy(input_dataset.IN_USE_STOCK_INIT_ARRAY)

        # Intializing the discarded material array Unit: Gt per year
        self.discarded_material = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the recycled material array Unit: Gt per year
        self.recycled_material = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the waste array Unit: Gt per year
        self.waste = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the extracted matter array Unit: Gt per year
        self.extracted_matter = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the material reserves array Unit: Gt per year
        self.material_reserves = copy.deepcopy(
            input_dataset.MATERIAL_RESERVES_INIT_ARRAY
        )

        # Intializing the converted material reserves array Unit: Gt per year
        self.converted_material_reserves = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the material resources array Unit: Gt per year
        self.material_resources = copy.deepcopy(
            input_dataset.MATERIAL_RESOURCES_INIT_ARRAY
        )

        # Intializing the depletion ratio
        self.depletion_ratio = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        # Initializing emissions avoided array
        self.emmissions_avoided = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

    #  Angela - if you are taking timestep as an argument, change the name to stepwise_run. You can create a run method that will just call the stepwise_run for the entire time horizon
    # TODO Palok - please review the new methods, is this what you meant here?
    def stepwise_run(self, timestep, output, recycling_rate):
        """
        Run the matter-use calculations for a given timestep.
        """
        if len(recycling_rate.shape) == 1:
            recycling_rate = recycling_rate.reshape(-1, 1)

        material_consumption = (
            self.material_intensity[:, timestep, :]
            * output[:, timestep, :]
            * 1000  # Output in trillions USD
        ) / 1_000_000_000  # Convert to Gt

        in_use_stock = self.get_in_use_stock(material_consumption, timestep)
        discarded_material = self.get_discarded_material(in_use_stock, timestep)
        recycled_material = self.get_recycled_material(discarded_material, recycling_rate
        )
        waste = self.get_waste(discarded_material, recycled_material)
        extracted_matter = self.get_extracted_matter(
            material_consumption, recycled_material
        )
        converted_material_reserves = self.get_converted_material_reserves(timestep)
        material_reserves = self.get_material_reserves(
            extracted_matter, converted_material_reserves, timestep
        )
        material_resources = self.get_material_resources(
            converted_material_reserves, timestep
        )
        depletion_ratio = self.get_depletion_ratio(
            extracted_matter, material_resources, timestep
        )

        # Emissions avoided by the amount of recycled material
        emissions_avoided = self.get_emissions_avoided(timestep, recycled_material)

        return depletion_ratio[:, timestep, :], emissions_avoided[:, timestep, :]

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
        for timestep in range(len(self.model_time_horizon)):
            depletion_ratio, emissions_avoided_timestep = self.stepwise_run(timestep, output, recycling_rate)
            depletion_ratios[:, timestep, :] = depletion_ratio
            emissions_avoided[:, timestep, :] = emissions_avoided_timestep
        return depletion_ratios, emissions_avoided
    
    ############################################################################################################

    # Matter-use variable calculations functions

    ############################################################################################################

    # NOTE: if the following functions are only specific to this class, and not used anywhere else, you can use the decorator @classmethod to make them private
    # Palok I have never use decorators, so do you think are necessary? also I changed the methods and don't know if I should use timestep here ?
    def get_in_use_stock(self, material_consumption, timestep):
        if timestep == 0:
            return self.in_use_stock[:, timestep, :]
        else:
            return (
                self.in_use_stock[:, timestep - 1, :]
                + material_consumption * self.physical_use_ratio
                - self.discarded_material[:, timestep, :]
            )

    def get_discarded_material(self, in_use_stock):
        return self.discard_rate * in_use_stock

    def get_recycled_material(self, discarded_material, recycling_rate=None):
        if recycling_rate is None:
            recycling_rate = self.recycling_rate
        return recycling_rate * discarded_material

    def get_waste(self, discarded_material, recycled_material):
        return discarded_material - recycled_material

    def get_extracted_matter(self, material_consumption, recycled_material):
        return material_consumption - recycled_material

    def get_converted_material_reserves(self, timestep):
        return (
            self.conversion_rate_material_reserves
            * self.material_resources[:, timestep - 1, :]
        )

    def get_material_reserves(
        self, extracted_matter, converted_material_reserves, timestep
    ):
        if timestep == 0:
            return self.material_reserves[:, timestep, :]
        else:
            return (
                self.material_reserves[:, timestep - 1, :]
                + converted_material_reserves
                - extracted_matter
            )

    def get_material_resources(self, converted_material_reserves, timestep):
        if timestep == 0:
            return self.material_resources[:, timestep, :]
        else:
            return (
                self.material_resources[:, timestep - 1, :]
                - converted_material_reserves
            )

    def get_depletion_ratio(self, extracted_matter, material_resources, timestep):
        return extracted_matter / material_resources

    ########################################################################################
    # Emissions avoided through recycling of paper and plastics
    ########################################################################################

    def get_emissions_avoided(self, timestep, recycled_material):
        # Calculate proportions of recycled materials in gigatons (Gt)
        recycled_paper = recycled_material * self.emissions_defaults["PROPORTION_PAPER"]
        recycled_plastic = (
            recycled_material * self.emissions_defaults["PROPORTION_PLASTIC"]
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

    def _interpolate_material_intensity(self):
        interp_data = np.zeros(
            (
                self.material_intensity_array.shape[0],
                len(self.model_time_horizon),
                # self.gdp_array.shape[2],
            )
        )
        for i in range(self.material_intensity_array.shape[0]):
            f = interp1d(
                self.data_time_horizon,
                self.material_intensity_array[i, :],
                kind="linear",  # , j
            )
            interp_data[i, :] = f(self.model_time_horizon)  # , j

        self.material_intensity_array = interp_data

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
