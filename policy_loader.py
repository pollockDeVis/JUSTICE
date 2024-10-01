
import json
import h5py
import numpy as np


class PolicyLoader:
    """
    This class loads all the input policy data for the JUSTICE model.
    """
    def __init__(self):
        # Create data file path for recycling data 
        recycling_file_path = "data/input/recycling/"
        emission_control_rate_file_path = "data/input/emissions_control_rate/constrained_emission_control_rate.npz"

        ###############################################################################
        # Load the Recycling data in hdf5 format
        ###############################################################################

        # In use stock initial values
        with h5py.File(f"{recycling_file_path}recycling_rate_linear_proyection.hdf5", "r") as f:
            self.RECYCLING_RATE_LINEAR_PROYECTION = f["recycling_rate_linear_proyection"][:]

        # In use stock initial values
        with h5py.File(f"{recycling_file_path}recycling_rate_2050_target.hdf5", "r") as f:
            self.RECYCLING_RATE_2050_TARGET = f["recycling_rate_2050_target"][:]

        ###############################################################################
        # Load the Constrained Emission Control Rate data in NPZ format
        ###############################################################################

        self.constrained_emission_control_rate = self.load_npz_file(emission_control_rate_file_path)
        print(f"Loaded constrained_emission_control_rate with scenarios: {list(self.constrained_emission_control_rate.keys())}")

    def load_npz_file(self, file_path):
        data = np.load(file_path)
        constrained_emission_control_rate = {key: data[key] for key in data}
        return constrained_emission_control_rate

if __name__ == "__main__":
    policy_loader = PolicyLoader()
    print(f"RECYCLING_RATE_LINEAR_PROYECTION shape: {policy_loader.RECYCLING_RATE_LINEAR_PROYECTION.shape}")
    print(f"RECYCLING_RATE_2050_TARGET shape: {policy_loader.RECYCLING_RATE_2050_TARGET.shape}")
    for scenario, array in policy_loader.constrained_emission_control_rate.items():
        print(f"{scenario} shape: {array.shape}")