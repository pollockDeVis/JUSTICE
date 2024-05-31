# JUSTICE
# Set this path to the src folder
# export PYTHONPATH=$PYTHONPATH:/Users/palokbiswas/Desktop/pollockdevis_git/JUSTICE/src

import pandas as pd

from src.util.data_loader import DataLoader

from src.util.model_time import TimeHorizon


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
        output_filepath + "RICE50_regional_" + data_label + "_data.xlsx"
    ) as writer:
        RICE50_data.to_excel(writer, sheet_name="RICE50")
    print(filename + " has been converted successfully")


if __name__ == "__main__":

    # Convert regional data for each SSP scenario from RICE50 to a format that can be used in the model
    RICE50_regional_data_converter(
        input_filepath="tests/raw_data/",
        output_filepath="tests/verification_data/",
        filename="damfrac_ssp1_cba_coop.xlsx",
        variable_name="DAMFRAC",
        sheet_name=None,
        cutout_columns=3,
        column_names=["Timestep", "Regions", "data_label"],
        start_year=2015,
        end_year=2300,
        data_timestep=5,
        timestep=1,
    )
