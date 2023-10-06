"""
This file contains the DataLoader class for the JUSTICE model.
"""

import os
import pandas as pd
import numpy as np


class DataLoader:

    """
    This class loads all the input data for the JUSTICE model.

    """

    def __init__(self):
        """
        This method initializes the DataLoader class.

        """
        # Get the current working directory
        current_directory = os.path.dirname(os.path.realpath(__file__))
        print(current_directory)
        # Go up to the root directory of the project (two levels up)

        root_directory = os.path.dirname(current_directory)
        print(root_directory)

        # Create the data file path
        data_file_path = os.path.join(root_directory, "data")
        print(data_file_path)

        ###############################################################################
        # Load the Economic data in pickle format
        ###############################################################################

        # Load GDP
        with open(os.path.join(data_file_path, "gdp_dict.pickle"), "rb") as f:
            self.GDP_DICT = pd.read_pickle(f)

        # Load the population
        with open(os.path.join(data_file_path, "population_dict.pickle"), "rb") as f:
            self.POPULATION_DICT = pd.read_pickle(f)

        # Load the capital stock initial values
        with open(os.path.join(data_file_path, "capital_init_arr.pickle"), "rb") as f:
            self.CAPITAL_INIT_ARRAY = pd.read_pickle(f)

        # Load the Saving Rate initial values
        with open(
            os.path.join(data_file_path, "savings_rate_init_arr.pickle"), "rb"
        ) as f:
            self.SAVING_RATE_INIT_ARRAY = pd.read_pickle(f)

        # Load the region list
        with open(os.path.join(data_file_path, "regions_list.pkl"), "rb") as f:
            self.REGION_LIST = pd.read_pickle(f)
