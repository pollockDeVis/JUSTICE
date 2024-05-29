"""
This file contains the DataLoader class for the JUSTICE model.
"""

import os
import pandas as pd
import numpy as np
import h5py


class DataLoader:
    """
    This class loads all the input data for the JUSTICE model.

    """

    # TODO: Future Optimization - Use scenario to instantiate the DataLoader and load only the required data
    # TODO: Bring the interpolation here from economy. Instantiation should take time horizon - Can also select specific years

    def __init__(self):
        """
        This method initializes the DataLoader class.

        """
        # Get the current working directory
        current_directory = os.path.dirname(os.path.realpath(__file__))

        # Go up two levels to the root directory of the project
        root_directory = os.path.dirname(os.path.dirname(current_directory))

        # Create the data file path
        data_file_path = os.path.join(root_directory, "data/input")

        ###############################################################################
        # Load the Economic data in hdf5 format
        ###############################################################################

        # Load GDP
        with h5py.File(os.path.join(data_file_path, "gdp_array.hdf5"), "r") as f:
            self.GDP_ARRAY = f["gdp"][:]

        # Load the population
        with h5py.File(os.path.join(data_file_path, "population_array.hdf5"), "r") as f:
            self.POPULATION_ARRAY = f["population"][:]

        # Load the emissions dictionary
        with h5py.File(os.path.join(data_file_path, "emissions_array.hdf5"), "r") as f:
            self.EMISSIONS_ARRAY = f["emissions"][:]

        # Load the capital stock initial values
        with h5py.File(os.path.join(data_file_path, "capital_init.hdf5"), "r") as f:
            self.CAPITAL_INIT_ARRAY = f["capital_init"][:]

        # Load the Saving Rate initial values
        with h5py.File(
            os.path.join(data_file_path, "savings_rate_init.hdf5"), "r"
        ) as f:
            self.SAVING_RATE_INIT_ARRAY = f["savings_rate_init"][:]

        # Load the region list
        with h5py.File(os.path.join(data_file_path, "region_list.hdf5"), "r") as f:
            regions_df = pd.DataFrame(f["region_list"][:])
            regions_str = regions_df[0].apply(lambda x: x.decode("utf-8")).values
            self.REGION_LIST = regions_str

        # Load PPP2MER conversion factor
        with h5py.File(os.path.join(data_file_path, "ppp2mer.hdf5"), "r") as f:
            self.PPP_TO_MER_CONVERSION_FACTOR = f["ppp2mer"][:]

        # Load the Temperature Downscaler Coefficients Alpha
        with h5py.File(
            os.path.join(data_file_path, "alpha_downscaler_coefficient.hdf5"), "r"
        ) as f:
            self.TEMPERATURE_DOWNSCALER_COEFFICIENT_ALPHA = f[
                "alpha_downscaler_coefficient"
            ][:]

        # Load the Temperature Downscaler Coefficients Beta
        with h5py.File(
            os.path.join(data_file_path, "beta_downscaler_coefficient.hdf5"), "r"
        ) as f:
            self.TEMPERATURE_DOWNSCALER_COEFFICIENT_BETA = f[
                "beta_downscaler_coefficient"
            ][:]

        # Load Abatement Coefficients
        with h5py.File(
            os.path.join(data_file_path, "abatement_coefficient_a.hdf5"), "r"
        ) as f:
            self.ABATEMENT_COEFFICIENT_A = f["abatement_coefficient_a"][:]

        with h5py.File(
            os.path.join(data_file_path, "abatement_coefficient_b.hdf5"), "r"
        ) as f:
            self.ABATEMENT_COEFFICIENT_B = f["abatement_coefficient_b"][:]
        
        # Load Material Intensity
        with h5py.File(os.path.join(data_file_path, "material_intensity_array.hdf5"), "r") as f:
            self.MATERIAL_INTENSITY_ARRAY = f["material_intensity"][:]
         
        #In use stock initial values
        with h5py.File(os.path.join(data_file_path, "in_use_init.hdf5"), "r") as f:
            self.IN_USE_STOCK_INIT_ARRAY = f["in_use_init"][:]
        
        #In use stock initial values
        with h5py.File(os.path.join(data_file_path, "material_reserves_init.hdf5"), "r") as f:
            self.MATERIAL_RESERVES_INIT_ARRAY = f["material_reserves_init"][:]
        
        #In use stock initial values
        with h5py.File(os.path.join(data_file_path, "material_resources_init.hdf5"), "r") as f:
            self.MATERIAL_RESOURCES_INIT_ARRAY = f["material_resources_init"][:]
