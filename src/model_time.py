"""
This file contains the Time class for the JUSTICE model which determines the time horizon and timesteps of the model.
"""


import numpy as np


class TimeHorizon:

    """
    This class controls the time horizon and time steps of the JUSTICE model.

    """

    def __init__(
        self,
        start_year,
        end_year,
        data_timestep,  # the time step of the data
        timestep,  # the time step of the model
    ):
        # Assign values to instance variables
        self.start_year = start_year

        self.end_year = end_year

        self.timestep = timestep

        self.data_timestep = data_timestep

    def get_data_time_horizon(self):
        """
        This method returns the time horizon of the data.
        """
        # Calculate the time horizon of the data
        self.data_time_horizon = len(
            np.arange(
                self.start_year,
                (self.end_year + self.data_timestep),
                self.data_timestep,
            )
        )  # Time horizon

        return self.data_time_horizon

    def get_model_time_horizon(self):
        """
        This method returns the time horizon of the model.
        """
        # Calculate the time horizon of the model
        self.model_time_horizon = len(
            np.arange(self.start_year, (self.end_year + self.timestep), self.timestep)
        )

        return self.model_time_horizon
