import numpy as np
import pytest
from src.climate.coupled_fair import CoupledFAIR
from src.util.model_time import TimeHorizon


def test_coupled_fair_historical_temperature():
    # Arrange: Define the time horizon and scenario
    time_horizon = TimeHorizon(
        start_year=2015, end_year=2300, data_timestep=5, timestep=1
    )
    scenario = 2

    # Instantiate the model with the desired CH4 method.
    climate = CoupledFAIR(ch4_method="Thornhill2021")

    # Act: Run the historical simulation.
    no_of_ensembles = climate.fair_justice_run_init(
        time_horizon=time_horizon,
        scenarios=scenario,
        baseline_run="default",
    )

    # Get the output cummins_state_array.
    cummins_state_array = climate.cummins_state_array

    # Load the validation data (make sure the file is at the correct path)
    validation_file = "tests/verification_data/fair/cummins_state_array_default.npy"
    cummins_state_array_default = np.load(validation_file)

    # Assert: Compare the resulting state array to the validation data.
    np.testing.assert_allclose(
        cummins_state_array, cummins_state_array_default, rtol=1e-5, atol=1e-5
    )


def test_coupled_fair_historical_temperature_with_purge():
    # Arrange: define the time horizon and scenario
    time_horizon = TimeHorizon(
        start_year=2015, end_year=2300, data_timestep=5, timestep=1
    )
    scenario = 2

    # Instantiate the CoupledFAIR model with the desired CH4 method
    climate = CoupledFAIR(ch4_method="Thornhill2021")

    # Act: Run the model with the 'purge' option for baseline_run.
    no_of_ensembles = climate.fair_justice_run_init(
        time_horizon=time_horizon, scenarios=scenario, baseline_run="purge"
    )

    # Obtain the resulting cummins_state_array which holds the historical state including temperature.
    cummins_state_array = climate.cummins_state_array

    # Load the expected validation data from file.
    validation_file = "tests/verification_data/fair/cummins_state_array_purge.npy"
    cummins_state_array_purge = np.load(validation_file)

    # Assert: Compare the computed array to the expected validation data.
    np.testing.assert_allclose(
        cummins_state_array, cummins_state_array_purge, rtol=1e-5, atol=1e-5
    )


def test_coupled_fair_historical_temperature_with_purge_with_reset():
    # Arrange: define the time horizon and scenario
    start_year = 2015
    end_year = 2300
    data_timestep = 5
    timestep = 1
    scenario = 2

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )

    # Instantiate the CoupledFAIR model with the desired CH4 method
    climate = CoupledFAIR(ch4_method="Thornhill2021")

    # Act: Run the model with the 'purge' option for baseline_run.
    no_of_ensembles = climate.fair_justice_run_init(
        time_horizon=time_horizon, scenarios=scenario, baseline_run="purge"
    )

    # Fair Reset
    climate.reset()

    # Create a numpy array of zeros to hold global temperature data of shape ((len(time_horizon.model_time_horizon), no_of_ensembles))
    global_temperature = np.zeros(
        (len(time_horizon.model_time_horizon), no_of_ensembles)
    )

    # Load the emissions data
    emissions_data = np.load(
        "tests/verification_data/regional_emissions_justice_linear_ecr.npy",
        allow_pickle=True,
    )

    # Load the expected validation data from file.
    global_temperature_justice_linear_ecr = np.load(
        "tests/verification_data/global_temperature_justice_linear_ecr.npy"
    )

    global_temperature[0, :] = climate.get_justice_initial_temperature()
    for timestep in range((len(time_horizon.model_time_horizon)) - 1):
        global_temperature[(timestep + 1), :] = (
            climate.compute_temperature_from_emission(
                timestep, emissions_data[:, timestep, :]
            )
        )

    # Assert: Compare the computed array to the expected validation data.
    np.testing.assert_allclose(
        global_temperature, global_temperature_justice_linear_ecr, rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
    test_coupled_fair_historical_temperature_with_purge_with_reset()
