# JUSTICE
# Set this path to the src folder
# export PYTHONPATH=$PYTHONPATH:/Users/palokbiswas/Desktop/pollockdevis_git/JUSTICE/src

import pandas as pd
import os
from justice.util.data_loader import DataLoader
from scipy.interpolate import interp1d
from justice.util.model_time import TimeHorizon
import numpy as np
from justice.util.regional_configuration import justice_region_aggregator
import json


def data_interpolator(
    input_filepath,
    output_filepath,
    filename,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
    aggregate_regions=True,
    region_mapping_json_path="data/input/rice_12_regions_dict.json",
):
    RICE50_data = pd.read_excel(os.path.join(input_filepath, filename))

    # Remove the first column
    RICE50_data = RICE50_data.iloc[:, 1:]
    # Print shape of the dataframe
    print(RICE50_data.shape)
    # Convert to numpy array
    RICE50_data = RICE50_data.to_numpy()

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(start_year, end_year, data_timestep, timestep)
    data_loader = DataLoader()
    region_list = data_loader.REGION_LIST
    print("Original regions: ", region_list)
    interp_data = np.zeros(
        (
            RICE50_data.shape[0],
            len(time_horizon.model_time_horizon),
        )
    )

    for i in range(RICE50_data.shape[0]):
        f = interp1d(
            time_horizon.data_time_horizon,
            RICE50_data[i, :],
            kind="linear",
        )
        interp_data[i, :] = f(time_horizon.model_time_horizon)

    if aggregate_regions:
        # Add a third dimension to the data by broadcasting it along the third axis. So that the shape changes from (57, 286) to (57, 286, 1)
        interp_data = np.expand_dims(interp_data, axis=2)

        # Load the dictionary from the json file
        with open(region_mapping_json_path, "r") as f:
            region_mapping_json = json.load(f)

        aggregated_region_list, interp_data = justice_region_aggregator(
            data_loader=data_loader, region_config=region_mapping_json, data=interp_data
        )

        # Drop the third dimension
        interp_data = np.squeeze(interp_data, axis=2)
        print("Aggregated regions: ", aggregated_region_list)

        interpolated_df = pd.DataFrame(
            interp_data,
            index=aggregated_region_list,
            columns=time_horizon.model_time_horizon,
        )

        interpolated_filename = (
            os.path.splitext(filename)[0] + "_aggregated_interpolated.xlsx"
        )
        interpolated_df.to_excel(os.path.join(output_filepath, interpolated_filename))
    else:
        interpolated_df = pd.DataFrame(
            interp_data,
            index=region_list,
            columns=time_horizon.model_time_horizon,
        )

        interpolated_filename = os.path.splitext(filename)[0] + "_interpolated.xlsx"

        interpolated_df.to_excel(os.path.join(output_filepath, interpolated_filename))


def temperature_interpolator(
    input_filepath,
    output_filepath,
    filename,
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
):
    RICE50_data = pd.read_excel(os.path.join(input_filepath, filename))

    # Only Keep the 'level' column
    RICE50_data = RICE50_data["level"]

    # Convert to numpy array
    RICE50_data = RICE50_data.to_numpy()

    # Transpose the numpy array
    RICE50_data = RICE50_data.reshape(1, -1)

    # Print shape of the dataframe
    print(RICE50_data.shape)

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(start_year, end_year, data_timestep, timestep)

    interp_data = np.zeros(
        (
            RICE50_data.shape[0],
            len(time_horizon.model_time_horizon),
        )
    )

    for i in range(RICE50_data.shape[0]):
        f = interp1d(
            time_horizon.data_time_horizon,
            RICE50_data[i, :],
            kind="linear",
            fill_value="extrapolate",
        )
        interp_data[i, :] = f(time_horizon.model_time_horizon)

    interpolated_df = pd.DataFrame(
        interp_data,
        columns=time_horizon.model_time_horizon,
    )

    interpolated_filename = os.path.splitext(filename)[0] + "_interpolated.xlsx"
    interpolated_df.to_excel(os.path.join(output_filepath, interpolated_filename))


def RICE50_regional_SSP_data_converter(
    input_filepath="raw_data/",
    output_filepath="verification_data/",
    filename=None,
    sheet_name=None,
    data_label="Value",
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
):
    """
    Convert regional data for each SSP scenario from RICE50 to a format that can be used in the model
    """
    # TODO: Not tested

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(start_year, end_year, data_timestep, timestep)

    RICE50_data = pd.read_excel(input_filepath + filename, sheet_name=sheet_name)

    RICE50_data.columns = ["SSP", "Timestep", "Regions", data_label]

    # Filling in the NaNs in the SSP column that was not present due to merged cells in the original excel file
    RICE50_data["SSP"] = RICE50_data["SSP"].fillna(method="ffill")

    # Filling in the NaN values in the Timestep column with the previous value
    RICE50_data["Timestep"] = RICE50_data["Timestep"].fillna(method="ffill")

    # Separating out the different SSP scenarios

    # SSP1
    SSP1_data = RICE50_data[RICE50_data["SSP"] == "SSP1"]

    # SSP2
    SSP2_data = RICE50_data[RICE50_data["SSP"] == "SSP2"]

    # SSP3
    SSP3_data = RICE50_data[RICE50_data["SSP"] == "SSP3"]

    # SSP4
    SSP4_data = RICE50_data[RICE50_data["SSP"] == "SSP4"]

    # SSP5
    SSP5_data = RICE50_data[RICE50_data["SSP"] == "SSP5"]

    # Convert the timestep to year
    SSP1_data["Year"] = time_horizon.timestep_to_year(
        (SSP1_data["Timestep"] - 1), data_timestep
    )
    SSP2_data["Year"] = time_horizon.timestep_to_year(
        (SSP2_data["Timestep"] - 1), data_timestep
    )
    SSP3_data["Year"] = time_horizon.timestep_to_year(
        (SSP3_data["Timestep"] - 1), data_timestep
    )
    SSP4_data["Year"] = time_horizon.timestep_to_year(
        (SSP4_data["Timestep"] - 1), data_timestep
    )
    SSP5_data["Year"] = time_horizon.timestep_to_year(
        (SSP5_data["Timestep"] - 1), data_timestep
    )

    # Drop the Timestep and SSP column
    SSP1_data = SSP1_data.drop(columns=["Timestep", "SSP"])
    SSP2_data = SSP2_data.drop(columns=["Timestep", "SSP"])
    SSP3_data = SSP3_data.drop(columns=["Timestep", "SSP"])
    SSP4_data = SSP4_data.drop(columns=["Timestep", "SSP"])
    SSP5_data = SSP5_data.drop(columns=["Timestep", "SSP"])

    # Pivot the dataframe
    SSP1_data = SSP1_data.pivot_table(
        index="Regions", columns="Year", values=data_label
    )
    SSP2_data = SSP2_data.pivot_table(
        index="Regions", columns="Year", values=data_label
    )
    SSP3_data = SSP3_data.pivot_table(
        index="Regions", columns="Year", values=data_label
    )
    SSP4_data = SSP4_data.pivot_table(
        index="Regions", columns="Year", values=data_label
    )
    SSP5_data = SSP5_data.pivot_table(
        index="Regions", columns="Year", values=data_label
    )

    # Now save these different SSP pivot dataframes in single excel file with different sheets
    with pd.ExcelWriter(
        output_filepath + "RICE50_regional_SSP_" + data_label + "_data.xlsx"
    ) as writer:
        SSP1_data.to_excel(writer, sheet_name="SSP1")
        SSP2_data.to_excel(writer, sheet_name="SSP2")
        SSP3_data.to_excel(writer, sheet_name="SSP3")
        SSP4_data.to_excel(writer, sheet_name="SSP4")
        SSP5_data.to_excel(writer, sheet_name="SSP5")


# Write a same method but without the SSP column
def RICE50_regional_data_converter(
    input_filepath="raw_data/",
    output_filepath="verification_data/",
    filename=None,
    variable_name="DAMFRAC",
    sheet_name=None,
    cutout_columns=3,
    column_names=["Timestep", "Regions", "data_label"],
    start_year=2015,
    end_year=2300,
    data_timestep=5,
    timestep=1,
):
    """
    Convert regional data for each SSP scenario from RICE50 to a format that can be used in the model
    """

    # Instantiate the TimeHorizon class
    time_horizon = TimeHorizon(start_year, end_year, data_timestep, timestep)

    RICE50_data = pd.read_excel(input_filepath + filename, sheet_name=sheet_name)

    # Convert RICE50 data into a dataframe
    RICE50_data = RICE50_data[variable_name]

    # Only keep the first few columns according to the cutout_columns
    RICE50_data = RICE50_data.iloc[:, :cutout_columns]

    # Rename the columns
    RICE50_data.columns = column_names  # ["Timestep", "Regions", data_label]

    timestep_label = column_names[0]
    region_label = column_names[1]
    data_label = column_names[2]
    # Filling in the NaN values in the Timestep column with the previous value
    RICE50_data[timestep_label] = RICE50_data[timestep_label].fillna(method="ffill")

    # Convert the timestep to year
    RICE50_data["Year"] = time_horizon.timestep_to_year(
        (RICE50_data[timestep_label] - 1), data_timestep  # RICE50 data starts from 1
    )

    # Drop the Timestep column
    RICE50_data = RICE50_data.drop(columns=[timestep_label])

    # Pivot the dataframe
    RICE50_data = RICE50_data.pivot_table(
        index=region_label, columns="Year", values=data_label
    )

    # Now save these different SSP pivot dataframes in single excel file with different sheets
    with pd.ExcelWriter(
        output_filepath + "RICE50_regional_" + variable_name + "_data.xlsx"
    ) as writer:
        RICE50_data.to_excel(writer, sheet_name="RICE50")
    print(filename + " has been converted successfully")
    # Print the output filepath
    current_dir = os.getcwd()
    print(
        current_dir
        + "/"
        + output_filepath
        + "RICE50_regional_"
        + data_label
        + "_data.xlsx"
    )


if __name__ == "__main__":

    # Print the working directory
    print(os.getcwd())

    # Convert regional data for each SSP scenario from RICE50 to a format that can be used in the model
    # RICE50_regional_data_converter(
    #     input_filepath="tests/raw_data/",
    #     output_filepath="tests/verification_data/",
    #     filename="results_ssp2_cba_coop_flex_sr_ecr10_2015_export.xlsx",
    #     variable_name="TATM",
    #     sheet_name=None,
    #     cutout_columns=3,
    #     column_names=["Timestep", "Regions", "data_label"],
    #     start_year=2015,
    #     end_year=2300,
    #     data_timestep=5,
    #     timestep=1,
    # )

    data_interpolator(
        input_filepath="tests/verification_data/",
        output_filepath="tests/verification_data/",
        filename="RICE50_regional_MIU_data.xlsx",
        start_year=2015,
        end_year=2300,
        data_timestep=5,
        timestep=1,
        aggregate_regions=False,
        region_mapping_json_path="data/input/rice_12_regions_dict.json",
    )

    # temperature_interpolator(
    #     input_filepath="tests/verification_data/",
    #     output_filepath="tests/verification_data/",
    #     filename="RICE50_regional_TATM_data.xlsx",
    #     start_year=2015,
    #     end_year=2300,
    #     data_timestep=5,
    #     timestep=1,
    # )
