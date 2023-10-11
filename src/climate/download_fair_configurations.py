"""
This helper script downloads the FAIR configuration files and Exogenous Emissions Data
NOTE: This script is not used in the model. It is only used to download the data. The URLs might change in the future.

"""

import os
import pooch


def download_fair_config(url, name, hash):
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Go up to the root directory of the project (two levels up)
    root_directory = os.path.dirname(os.path.dirname(current_directory))

    # Create the data file path
    data_file_path = os.path.join(root_directory, "data")

    print(data_file_path)

    # Create a Pooch instance to manage the data
    pooch.retrieve(url=url, path=data_file_path, fname=name, known_hash=hash)


fair_configs = [
    {
        "url": "https://zenodo.org/record/7694879/files/calibrated_constrained_parameters.csv",
        "name": "calibrated_constrained_parameters.csv",
        "hash": "md5:9f236c43dd18a36b7b63b94e05f3caab",
    },
    {
        # WARNING: This URL might change in the future
        "url": "https://raw.githubusercontent.com/OMS-NetZero/FAIR/03aac4fba28bb3c9bf8cf10898df7b7fbeea1359/examples/data/species_configs_properties_calibration1.2.0.csv",
        "name": "species_configs_properties_calibration.csv",
        "hash": "md5:92ed36d299e9b48c7a16acc8fd0f973a",
    },
    {
        "url": "doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
        "name": "rcmip_emissions_annual.csv",
        "hash": "md5:4044106f55ca65b094670e7577eaf9b3",
    },
    {
        "url": "doi:10.5281/zenodo.4589756/rcmip-concentrations-annual-means-v5-1-0.csv",
        "name": "rcmip_concentrations_annual.csv",
        "hash": "md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
    },
    {
        "url": "doi:10.5281/zenodo.4589756/rcmip-radiative-forcing-annual-means-v5-1-0.csv",
        "name": "rcmip_forcing_annual.csv",
        "hash": "md5:87ef6cd4e12ae0b331f516ea7f82ccba",
    },
]

if __name__ == "__main__":
    for fair_config in fair_configs:
        download_fair_config(
            fair_config["url"], fair_config["name"], fair_config["hash"]
        )
