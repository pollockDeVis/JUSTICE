"""
This is the Temperature Downscaler module that converts the global temperature
to regional temperature.

A calibrated Statistical Downscaler, converting Global average temperature to heterogenous warming across the regions
Used CMIP5 database by Taylor et al., 2011 (https://journals.ametsoc.org/view/journals/bams/93/4/bams-d-11-00094.1.xml)
Data contains historical projections of temperature and precipitation on a 0.5 degree grid
Values aggregated using population weights using population data from the year 2000 and kept fixed over time #Assumption
Mean model ensemble from all RCPs used to link global mean temperature increase to the country level average temperature increase
Coefficients p and q determine the local temperature T. T = p + q*Global_Temperature
"""

import numpy as np


class TemperatureDownscaler:
    """
    This class converts the global temperature to regional temperature.
    """

    def __init__(self, input_dataset):
        """
        This method initializes the TemperatureDownscaler class.
        """
        # Get the Downscaling Coefficients
        self.alpha = input_dataset.TEMPERATURE_DOWNSCALER_COEFFICIENT_ALPHA
        self.beta = input_dataset.TEMPERATURE_DOWNSCALER_COEFFICIENT_BETA

    def get_regional_temperature(self, global_temperature):
        """
        This method returns the regional temperature.
        """

        # Reshape the global temperature to (1, climate_ensembles)
        global_temperature = global_temperature.reshape(1, global_temperature.shape[0])

        # Calculate the regional temperature
        regional_temperature = self.alpha + self.beta * global_temperature

        # Return the regional temperature
        return regional_temperature
