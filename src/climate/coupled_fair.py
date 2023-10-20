"""
This is the script that inherits original FaIR model and modifies it to be used in the JUSTICE model.
Original FaIR model can be found here: https://github.com/OMS-NetZero/FAIR
"""

# FaIR Imports
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
from fair.forcing.ghg import etminan2016, leach2021ghg, meinshausen2020, myhre1998

from fair.forcing.aerosol.erfaci import logsum
from fair.forcing.aerosol.erfari import calculate_erfari_forcing
from fair.forcing.minor import calculate_linear_forcing
from fair.forcing.ozone import thornhill2021

from fair.energy_balance_model import (
    calculate_toa_imbalance_postrun,
    multi_ebm,
    step_temperature,
)

from fair.earth_params import (
    earth_radius,
    mass_atmosphere,
    molecular_weight_air,
    seconds_per_year,
)

from fair.structure.units import (
    compound_convert,
    desired_concentration_units,
    desired_emissions_units,
    mixing_ratio_convert,
    prefix_convert,
    time_convert,
)

from fair.constants import SPECIES_AXIS, TIME_AXIS

from fair.gas_cycle import calculate_alpha
from fair.gas_cycle.ch4_lifetime import calculate_alpha_ch4
from fair.gas_cycle.eesc import calculate_eesc
from fair.gas_cycle.forward import step_concentration
from fair.gas_cycle.inverse import unstep_concentration

# Other Imports
import numpy as np
import pandas as pd
import warnings
import os
import copy
from scipy.interpolate import interp1d
from typing import Any


# FaIR Model Constants
fair_start_year = 1750
suppress_warnings = True

# Path to the data directory
current_directory = os.path.dirname(os.path.realpath(__file__))

# Go up to the root directory of the project (two levels up)
root_directory = os.path.dirname(os.path.dirname(current_directory))

# Create the data file path
data_file_path = os.path.join(root_directory, "data/input")

# Modified version of FAIR where class is inherited from FAIR and new methods are added


class CoupledFAIR(FAIR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)

    def fair_justice_run_init(self, time_horizon, scenarios):
        """Setup the stepwise run of the FaIR model with the JUSTICE IAM.

        Parameters
        ----------
        suppress_warnings : bool
            Hide warnings relating to covariance in energy balance matrix.
        """
        self.start_year_fair = fair_start_year
        self.start_year_justice = time_horizon.start_year
        self.end_year_fair = time_horizon.end_year
        self.timestep_justice = time_horizon.timestep

        scenarios = [scenarios]  # Converting into a list

        self.fair_fill_data(scenarios)

        # End of filling in configs
        self._check_properties()
        self._make_indices()
        if self._routine_flags["temperature"]:
            with warnings.catch_warnings():
                if suppress_warnings:
                    warnings.filterwarnings(
                        "ignore", message="covariance is not positive-semidefinite."
                    )
                self._make_ebms()

        # part of pre-run: TODO move to a new method
        if (
            self._co2_indices.sum()
            + self._co2_ffi_indices.sum()
            + self._co2_afolu_indices.sum()
            == 3
        ):
            self.emissions[..., self._co2_indices] = (
                self.emissions[..., self._co2_ffi_indices].data
                + self.emissions[..., self._co2_afolu_indices].data
            )

        self.cumulative_emissions[1:, ...] = (
            self.emissions.cumsum(axis=0, skipna=False) * self.timestep
            + self.cumulative_emissions[0, ...]
        )

        # create numpy arrays
        self.alpha_lifetime_array = self.alpha_lifetime.data
        self.airborne_emissions_array = self.airborne_emissions.data
        self.baseline_concentration_array = self.species_configs[
            "baseline_concentration"
        ].data
        self.baseline_emissions_array = self.species_configs["baseline_emissions"].data
        self.br_atoms_array = self.species_configs["br_atoms"].data
        self.ch4_lifetime_chemical_sensitivity_array = self.species_configs[
            "ch4_lifetime_chemical_sensitivity"
        ].data
        self.lifetime_temperature_sensitivity_array = self.species_configs[
            "lifetime_temperature_sensitivity"
        ].data
        self.cl_atoms_array = self.species_configs["cl_atoms"].data
        self.concentration_array = self.concentration.data
        self.concentration_per_emission_array = self.species_configs[
            "concentration_per_emission"
        ].data
        self.contrails_radiative_efficiency_array = self.species_configs[
            "contrails_radiative_efficiency"
        ].data
        self.cummins_state_array = (
            np.ones(
                (
                    self._n_timebounds,
                    self._n_scenarios,
                    self._n_configs,
                    self._n_layers + 1,
                )
            )
            * np.nan
        )
        self.cumulative_emissions_array = self.cumulative_emissions.data
        self.deep_ocean_efficacy_array = self.climate_configs[
            "deep_ocean_efficacy"
        ].data
        self.emissions_array = self.emissions.data

        self.erfari_radiative_efficiency_array = self.species_configs[
            "erfari_radiative_efficiency"
        ].data
        self.erfaci_scale_array = self.species_configs["aci_scale"].data
        self.erfaci_shape_array = self.species_configs["aci_shape"].data
        self.forcing_array = self.forcing.data
        self.forcing_scale_array = self.species_configs["forcing_scale"].data * (
            1 + self.species_configs["tropospheric_adjustment"].data
        )
        self.forcing_efficacy_array = self.species_configs["forcing_efficacy"].data
        self.forcing_efficacy_sum_array = (
            np.ones((self._n_timebounds, self._n_scenarios, self._n_configs)) * np.nan
        )
        self.forcing_reference_concentration_array = self.species_configs[
            "forcing_reference_concentration"
        ].data
        self.forcing_sum_array = self.forcing_sum.data
        self.forcing_temperature_feedback_array = self.species_configs[
            "forcing_temperature_feedback"
        ].data
        self.fractional_release_array = self.species_configs["fractional_release"].data
        self.g0_array = self.species_configs["g0"].data
        self.g1_array = self.species_configs["g1"].data
        self.gas_partitions_array = self.gas_partitions.data
        self.greenhouse_gas_radiative_efficiency_array = self.species_configs[
            "greenhouse_gas_radiative_efficiency"
        ].data
        self.h2o_stratospheric_factor_array = self.species_configs[
            "h2o_stratospheric_factor"
        ].data
        self.iirf_0_array = self.species_configs["iirf_0"].data
        self.iirf_airborne_array = self.species_configs["iirf_airborne"].data
        self.iirf_temperature_array = self.species_configs["iirf_temperature"].data
        self.iirf_uptake_array = self.species_configs["iirf_uptake"].data
        self.land_use_cumulative_emissions_to_forcing_array = self.species_configs[
            "land_use_cumulative_emissions_to_forcing"
        ].data
        self.lapsi_radiative_efficiency_array = self.species_configs[
            "lapsi_radiative_efficiency"
        ].data
        self.ocean_heat_transfer_array = self.climate_configs[
            "ocean_heat_transfer"
        ].data
        self.ozone_radiative_efficiency_array = self.species_configs[
            "ozone_radiative_efficiency"
        ].data
        self.partition_fraction_array = self.species_configs["partition_fraction"].data
        self.unperturbed_lifetime_array = self.species_configs[
            "unperturbed_lifetime"
        ].data

        if self._routine_flags["temperature"]:
            self.eb_matrix_d_array = self.ebms["eb_matrix_d"].data
            self.forcing_vector_d_array = self.ebms["forcing_vector_d"].data
            self.stochastic_d_array = self.ebms["stochastic_d"].data

        # forcing should be initialised so this should not be nan. We could check, or
        # allow silent fail as some species don't take forcings and would correctly be
        # nan.
        self.forcing_sum_array[0:1, ...] = np.nansum(
            self.forcing_array[0:1, ...], axis=SPECIES_AXIS
        )

        # this is the most important state vector
        self.cummins_state_array[0, ..., 0] = self.forcing_sum_array[0, ...]
        self.cummins_state_array[..., 1:] = self.temperature.data

        # non-linear forcing relationships need an offset. To save calculating
        # them every timestep, we'll pre-determine the forcing to use as the
        # baseline values.
        # GHGs forcing under Meinshausen2020
        # This check, and others, need to come earlier.
        if self._routine_flags["ghg"] and self.ghg_method == "meinshausen2020":
            if (
                np.sum(
                    np.isnan(
                        self.forcing_reference_concentration_array[:, self._ghg_indices]
                    )
                )
                > 0
            ):
                raise ValueError(
                    "There are NaNs in "
                    "FAIR.species_configs['forcing_reference_concentration'] which "
                    "means that I can't calculate greenhouse gas forcing."
                )
            self.ghg_forcing_offset = meinshausen2020(
                self.baseline_concentration_array[None, None, ...],
                self.forcing_reference_concentration_array[None, None, ...],
                self.forcing_scale_array[None, None, ...],
                self.greenhouse_gas_radiative_efficiency_array[None, None, ...],
                self._co2_indices,
                self._ch4_indices,
                self._n2o_indices,
                self._minor_ghg_indices,
            )

        # purge emissions

        self.purge_emissions(scenarios)

        # Run the historical temperature computation

        self.run_historical_temperature_calculation()

    def compute_temperature_from_emission(self, timestep, emissions_data):
        """
        Fill emissions for a given timestep with new emissions and computes the temperature rise in celcius.
        Args:
        timestep (int): The timestep to fill emissions for.
        emissions_data (numpy.ndarray): The new emissions data. Its shape should be (1001, ).
        """

        # Verify shape of emissions data #TODO: Later
        # expected_shape = ( self.number_of_ensembles,)
        # assert (
        #     emissions_data.shape == expected_shape
        # ), f"Emissions data shape: {emissions_data.shape}, expected shape {expected_shape}"

        # Sum all the regions
        emissions_data = np.sum(emissions_data, axis=0)

        fill_index = timestep + self.justice_start_index

        # Replace the respective timestep with the emissions data
        self.emissions_purge_array[fill_index, 0, :] = emissions_data

        # Fill the emissions array with the new emissions data
        fill(self.emissions, self.emissions_purge_array, specie="CO2 FFI")

        self.stepwise_run(fill_index)
        global_temperature = self.get_temperature_array(timestep)
        # Shape [timestep, scenario, ensemble, box/layer=0] # Layer 0 is used in FAIR example. The current code works only with one SSP-RCP scenario
        # global_temperature = global_temperature[timestep, 0, :, 0]
        return global_temperature

    def purge_emissions(self, scenario):
        """
        This function purges the emissions after the justice start year.
        This is because FAIR starts from 1750 and we want calculate all the historical temperatures
        up until the start year of JUSTICE model.
        JUSTICE uses the historical temperature and emissions and builds on top of it.
        Only CO2 FFI is purged because  the other emissions are exogenous
        emissions_purge_array shape: (550, 1, 1001)
        """
        # Select data for "CO2 FFI" and scenario
        rcmip_emission_array = self.emissions.sel(specie="CO2 FFI", scenario=scenario)
        # Calculate justice start index
        self.justice_start_index = self.start_year_justice - self.start_year_fair
        # Create array with rcmip emissions before justice_start_index and zeros after

        self.emissions_purge_array = np.concatenate(
            [
                rcmip_emission_array[0 : self.justice_start_index].values,
                np.full(
                    (self.end_year_fair - self.start_year_justice,)
                    + rcmip_emission_array.shape[1:],
                    np.nan,
                ),
            ],
            axis=0,
        )

        fill(self.emissions, self.emissions_purge_array, specie="CO2 FFI")

    def run_historical_temperature_calculation(self):
        """
        This function calculates the historical temperature from 1750 to the start year of JUSTICE model.
        The historical temperature is required for the FAIR model to project temperatures in the future using JUSTICE model.
        """
        fair_historical_years = np.arange(
            fair_start_year, self.start_year_justice, self.timestep_justice
        )
        # justice_run_years = np.arange(
        #     self.start_year_justice,
        #     (self.end_year_fair + self.timestep_justice),
        #     self.timestep_justice,
        # )
        self.fair_historical_timestep_run_count = len(fair_historical_years)

        for i in range(0, len(fair_historical_years)):
            self.stepwise_run(i)

    # #range(self._n_timepoints) 0 - 549 1750 - 2300 , we gotta run the loop from 1750 - JUSTICE start time first, and then do step by step/ just call this function within a loop
    def stepwise_run(self, i_timepoint):
        if self._routine_flags["ghg"]:
            # 1. alpha scaling
            self.alpha_lifetime_array[
                i_timepoint : i_timepoint + 1, ..., self._ghg_indices
            ] = calculate_alpha(  # this timepoint
                self.airborne_emissions_array[
                    i_timepoint : i_timepoint + 1, ..., self._ghg_indices
                ],  # last timebound
                self.cumulative_emissions_array[
                    i_timepoint : i_timepoint + 1, ..., self._ghg_indices
                ],  # last timebound
                self.g0_array[None, None, ..., self._ghg_indices],
                self.g1_array[None, None, ..., self._ghg_indices],
                self.iirf_0_array[None, None, ..., self._ghg_indices],
                self.iirf_airborne_array[None, None, ..., self._ghg_indices],
                self.iirf_temperature_array[None, None, ..., self._ghg_indices],
                self.iirf_uptake_array[None, None, ..., self._ghg_indices],
                self.cummins_state_array[i_timepoint : i_timepoint + 1, ..., 1:2],
                self.iirf_max,
            )

        # 2. multi-species methane lifetime if desired; update GHG concentration
        # for CH4
        # needs previous timebound but this is no different to the generic
        if self.ch4_method == "thornhill2021":
            self.alpha_lifetime_array[
                i_timepoint : i_timepoint + 1, ..., self._ch4_indices
            ] = calculate_alpha_ch4(
                self.emissions_array[i_timepoint : i_timepoint + 1, ...],
                self.concentration_array[i_timepoint : i_timepoint + 1, ...],
                self.cummins_state_array[i_timepoint : i_timepoint + 1, ..., 1:2],
                self.baseline_emissions_array[None, None, ...],
                self.baseline_concentration_array[None, None, ...],
                self.ch4_lifetime_chemical_sensitivity_array[None, None, ...],
                self.lifetime_temperature_sensitivity_array[None, None, :, None],
                self._aerosol_chemistry_from_emissions_indices,
                self._aerosol_chemistry_from_concentration_indices,
            )

        # 3. greenhouse emissions to concentrations; include methane from IIRF
        (
            self.concentration_array[
                i_timepoint + 1 : i_timepoint + 2,
                ...,
                self._ghg_forward_indices,
            ],
            self.gas_partitions_array[..., self._ghg_forward_indices, :],
            self.airborne_emissions_array[
                i_timepoint + 1 : i_timepoint + 2,
                ...,
                self._ghg_forward_indices,
            ],
        ) = step_concentration(
            self.emissions_array[
                i_timepoint : i_timepoint + 1,
                ...,
                self._ghg_forward_indices,
                None,
            ],  # this timepoint
            self.gas_partitions_array[
                ..., self._ghg_forward_indices, :
            ],  # last timebound
            self.airborne_emissions_array[
                i_timepoint + 1 : i_timepoint + 2,
                ...,
                self._ghg_forward_indices,
                None,
            ],  # last timebound
            self.alpha_lifetime_array[
                i_timepoint : i_timepoint + 1,
                ...,
                self._ghg_forward_indices,
                None,
            ],
            self.baseline_concentration_array[
                None, None, ..., self._ghg_forward_indices
            ],
            self.baseline_emissions_array[
                None, None, ..., self._ghg_forward_indices, None
            ],
            self.concentration_per_emission_array[
                None, None, ..., self._ghg_forward_indices
            ],
            self.unperturbed_lifetime_array[
                None, None, ..., self._ghg_forward_indices, :
            ],
            #        oxidation_matrix,
            self.partition_fraction_array[
                None, None, ..., self._ghg_forward_indices, :
            ],
            self.timestep,
        )

        # 4. greenhouse gas concentrations to emissions
        (
            self.emissions_array[
                i_timepoint : i_timepoint + 1, ..., self._ghg_inverse_indices
            ],
            self.gas_partitions_array[..., self._ghg_inverse_indices, :],
            self.airborne_emissions_array[
                i_timepoint + 1 : i_timepoint + 2,
                ...,
                self._ghg_inverse_indices,
            ],
        ) = unstep_concentration(
            self.concentration_array[
                i_timepoint + 1 : i_timepoint + 2,
                ...,
                self._ghg_inverse_indices,
            ],  # this timepoint
            self.gas_partitions_array[
                None, ..., self._ghg_inverse_indices, :
            ],  # last timebound
            self.airborne_emissions_array[
                i_timepoint : i_timepoint + 1,
                ...,
                self._ghg_inverse_indices,
                None,
            ],  # last timebound
            self.alpha_lifetime_array[
                i_timepoint : i_timepoint + 1,
                ...,
                self._ghg_inverse_indices,
                None,
            ],
            self.baseline_concentration_array[
                None, None, ..., self._ghg_inverse_indices
            ],
            self.baseline_emissions_array[None, None, ..., self._ghg_inverse_indices],
            self.concentration_per_emission_array[
                None, None, ..., self._ghg_inverse_indices
            ],
            self.unperturbed_lifetime_array[
                None, None, ..., self._ghg_inverse_indices, :
            ],
            #        oxidation_matrix,
            self.partition_fraction_array[
                None, None, ..., self._ghg_inverse_indices, :
            ],
            self.timestep,
        )
        self.cumulative_emissions_array[
            i_timepoint + 1, ..., self._ghg_inverse_indices
        ] = (
            self.cumulative_emissions_array[i_timepoint, ..., self._ghg_inverse_indices]
            + self.emissions_array[i_timepoint, ..., self._ghg_inverse_indices]
            * self.timestep
        )

        # 5. greenhouse gas concentrations to forcing
        if self.ghg_method == "leach2021":
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
            ] = leach2021ghg(
                self.concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                self.baseline_concentration_array[None, None, ...]
                * np.ones((1, self._n_scenarios, self._n_configs, self._n_species)),
                self.forcing_scale_array[None, None, ...],
                self.greenhouse_gas_radiative_efficiency_array[None, None, ...],
                self._co2_indices,
                self._ch4_indices,
                self._n2o_indices,
                self._minor_ghg_indices,
            )[
                0:1, ..., self._ghg_indices
            ]
        if self.ghg_method == "meinshausen2020":
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
            ] = meinshausen2020(
                self.concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                self.forcing_reference_concentration_array[None, None, ...]
                * np.ones((1, self._n_scenarios, self._n_configs, self._n_species)),
                self.forcing_scale_array[None, None, ...],
                self.greenhouse_gas_radiative_efficiency_array[None, None, ...],
                self._co2_indices,
                self._ch4_indices,
                self._n2o_indices,
                self._minor_ghg_indices,
            )[
                0:1, ..., self._ghg_indices
            ]
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
            ] = (
                self.forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
                ]
                - self.ghg_forcing_offset[..., self._ghg_indices]
            )
        elif self.ghg_method == "etminan2016":
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
            ] = etminan2016(
                self.concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                self.baseline_concentration_array[None, None, ...]
                * np.ones((1, self._n_scenarios, self._n_configs, self._n_species)),
                self.forcing_scale_array[None, None, ...],
                self.greenhouse_gas_radiative_efficiency_array[None, None, ...],
                self._co2_indices,
                self._ch4_indices,
                self._n2o_indices,
                self._minor_ghg_indices,
            )[
                0:1, ..., self._ghg_indices
            ]
        elif self.ghg_method == "myhre1998":
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._ghg_indices
            ] = myhre1998(
                self.concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                self.baseline_concentration_array[None, None, ...]
                * np.ones((1, self._n_scenarios, self._n_configs, self._n_species)),
                self.forcing_scale_array[None, None, ...],
                self.greenhouse_gas_radiative_efficiency_array[None, None, ...],
                self._co2_indices,
                self._ch4_indices,
                self._n2o_indices,
                self._minor_ghg_indices,
            )[
                0:1, ..., self._ghg_indices
            ]

        # 6. aerosol direct forcing
        if self._routine_flags["ari"]:
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._ari_indices
            ] = calculate_erfari_forcing(
                self.emissions_array[i_timepoint : i_timepoint + 1, ...],
                self.concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                self.baseline_emissions_array[None, None, ...],
                self.baseline_concentration_array[None, None, ...],
                self.forcing_scale_array[None, None, ...],
                self.erfari_radiative_efficiency_array[None, None, ...],
                self._aerosol_chemistry_from_emissions_indices,
                self._aerosol_chemistry_from_concentration_indices,
            )

        # 7. aerosol indirect forcing
        if self._routine_flags["aci"]:
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._aci_indices
            ] = logsum(
                self.emissions_array[i_timepoint : i_timepoint + 1, ...],
                self.concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                self.baseline_emissions_array[None, None, ...],
                self.baseline_concentration_array[None, None, ...],
                self.forcing_scale_array[None, None, ..., self._aci_indices],
                self.erfaci_scale_array[None, None, :],
                self.erfaci_shape_array[None, None, ...],
                self._aerosol_chemistry_from_emissions_indices,
                self._aerosol_chemistry_from_concentration_indices,
            )

        # 8. calculate EESC this timestep for ozone forcing (and use it for
        # methane lifetime in the following timestep)
        if self._routine_flags["eesc"]:
            self.concentration_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._eesc_indices
            ] = calculate_eesc(
                self.concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                self.fractional_release_array[None, None, ...],
                self.cl_atoms_array[None, None, ...],
                self.br_atoms_array[None, None, ...],
                self._cfc11_indices,
                self._halogen_indices,
                self.br_cl_ods_potential,
            )

        # 9. ozone emissions & concentrations to forcing
        if self._routine_flags["ozone"]:
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._ozone_indices
            ] = thornhill2021(
                self.emissions_array[i_timepoint : i_timepoint + 1, ...],
                self.concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                self.baseline_emissions_array[None, None, ...],
                self.baseline_concentration_array[None, None, ...],
                self.forcing_scale_array[None, None, ..., self._ozone_indices],
                self.ozone_radiative_efficiency_array[None, None, ...],
                self._aerosol_chemistry_from_emissions_indices,
                self._aerosol_chemistry_from_concentration_indices,
            )

        # 10. contrails forcing from NOx emissions
        if self._routine_flags["contrails"]:
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._contrails_indices
            ] = calculate_linear_forcing(
                self.emissions_array[i_timepoint : i_timepoint + 1, ...],
                0,
                self.forcing_scale_array[None, None, ..., self._contrails_indices],
                self.contrails_radiative_efficiency_array[None, None, ...],
            )

        # 11. LAPSI forcing from BC and OC emissions
        if self._routine_flags["lapsi"]:
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._lapsi_indices
            ] = calculate_linear_forcing(
                self.emissions_array[i_timepoint : i_timepoint + 1, ...],
                self.baseline_emissions_array[None, None, ...],
                self.forcing_scale_array[None, None, ..., self._lapsi_indices],
                self.lapsi_radiative_efficiency_array[None, None, ...],
            )

        # 12. CH4 forcing to stratospheric water vapour forcing
        if self._routine_flags["h2o stratospheric"]:
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._h2ostrat_indices
            ] = calculate_linear_forcing(
                self.forcing_array[i_timepoint + 1 : i_timepoint + 2, ...],
                0,
                self.forcing_scale_array[None, None, ..., self._h2ostrat_indices],
                self.h2o_stratospheric_factor_array[None, None, ...],
            )

        # 13. CO2 cumulative emissions to land use change forcing
        if self._routine_flags["land use"]:
            self.forcing_array[
                i_timepoint + 1 : i_timepoint + 2, ..., self._landuse_indices
            ] = calculate_linear_forcing(
                self.cumulative_emissions_array[i_timepoint + 1 : i_timepoint + 2, ...],
                0,
                self.forcing_scale_array[None, None, ..., self._landuse_indices],
                self.land_use_cumulative_emissions_to_forcing_array[None, None, ...],
            )

        # 14. apply temperature-forcing feedback here.
        self.forcing_array[i_timepoint + 1 : i_timepoint + 2, ...] = (
            self.forcing_array[i_timepoint + 1 : i_timepoint + 2, ...]
            + self.cummins_state_array[i_timepoint : i_timepoint + 1, ..., 1:2]
            * self.forcing_temperature_feedback_array[None, None, ...]
        )

        # 15. sum forcings
        self.forcing_sum_array[i_timepoint + 1 : i_timepoint + 2, ...] = np.nansum(
            self.forcing_array[i_timepoint + 1 : i_timepoint + 2, ...],
            axis=SPECIES_AXIS,
        )
        self.forcing_efficacy_sum_array[
            i_timepoint + 1 : i_timepoint + 2, ...
        ] = np.nansum(
            self.forcing_array[i_timepoint + 1 : i_timepoint + 2, ...]
            * self.forcing_efficacy_array[None, None, ...],
            axis=SPECIES_AXIS,
        )

        # 16. forcing to temperature
        if self._routine_flags["temperature"]:
            self.cummins_state_array[
                i_timepoint + 1 : i_timepoint + 2, ...
            ] = step_temperature(
                self.cummins_state_array[i_timepoint : i_timepoint + 1, ...],
                self.eb_matrix_d_array[None, None, ...],
                self.forcing_vector_d_array[None, None, ...],
                self.stochastic_d_array[i_timepoint + 1 : i_timepoint + 2, None, ...],
                self.forcing_efficacy_sum_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., None
                ],
            )

    def get_exogenous_land_use_emissions(self, fair_scenario):
        """Get the exogenous land use emissions for a given scenario.

        Parameters
        ----------
        scenario : str
            The name of the scenario to get the emissions for.

        Returns
        -------
        land_use_emissions : np.ndarray
            The land use emissions for the given scenario for the whole world.
        """

        land_use_emissions = self.emissions.sel(
            specie="CO2 AFOLU", scenario=[fair_scenario]
        )

        land_use_emissions = (land_use_emissions.values)[
            (self.justice_start_index - self.timestep_justice) :, 0, :
        ] / 1e3  # Converting to GtCO2 from MtCO2

        return land_use_emissions

    def calculate_toa_ocean_airborne_fraction(self):
        """Calculate the fraction of airborne emissions that are taken up by the
        ocean.

        Returns
        -------
        toa_ocean_airborne_fraction : np.ndarray
            The fraction of airborne emissions that are taken up by the ocean.
        """
        self.toa_imbalance_array = calculate_toa_imbalance_postrun(
            self.cummins_state_array,
            self.forcing_sum_array,  # [..., None],
            self.ocean_heat_transfer_array,
            self.deep_ocean_efficacy_array,
        )

        # 18. Ocean heat content change
        self.ocean_heat_content_change_array = (
            np.cumsum(self.toa_imbalance_array * self.timestep, axis=TIME_AXIS)
            * earth_radius**2
            * 4
            * np.pi
            * seconds_per_year
        )

        # 19. calculate airborne fraction - we have NaNs and zeros we know about, and we
        # don't mind
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.airborne_fraction_array = (
                self.airborne_emissions_array / self.cumulative_emissions_array
            )

    def prepare_output_xarrays(self):
        # 20. (Re)allocate to xarray
        self.temperature.data = self.cummins_state_array[..., 1:]
        self.concentration.data = self.concentration_array
        self.emissions.data = self.emissions_array
        self.forcing.data = self.forcing_array
        self.forcing_sum.data = self.forcing_sum_array
        self.cumulative_emissions.data = self.cumulative_emissions_array
        self.airborne_emissions.data = self.airborne_emissions_array
        self.airborne_fraction.data = self.airborne_fraction_array
        self.gas_partitions.data = self.gas_partitions_array
        self.ocean_heat_content_change.data = self.ocean_heat_content_change_array
        self.toa_imbalance.data = self.toa_imbalance_array
        self.stochastic_forcing.data = self.cummins_state_array[..., 0]

    def get_temperature_array(self, timestep):
        # Shape [timestep, scenario, ensemble, box/layer=0] # Layer 0 is used in FAIR example. The current code works only with one SSP-RCP scenario
        temperature_array = self.cummins_state_array[..., 1:]
        temperature_array = temperature_array[timestep, 0, :, 0]
        return temperature_array

    def fill_emissions_from_economy_submodel(self, emissions_economy_submodule):
        """Fill emissions from the economy submodule.
        #TODO: Need to test this function. Maybe need to add a scenario argument?
        """
        fill(
            self.emissions,
            emissions_economy_submodule[
                :, None, None
            ],  # essentially reshaping the emissions into 3D shape used by FAIR
            specie="CO2 FFI",
        )  # scenario=scenario

    def fill_from_rcmip_local(self, data_file_path):
        """Fill emissions, concentrations and/or forcing from RCMIP scenarios."""
        # lookup converting FaIR default names to RCMIP names
        species_to_rcmip = {specie: specie.replace("-", "") for specie in self.species}
        species_to_rcmip["CO2 FFI"] = "CO2|MAGICC Fossil and Industrial"
        species_to_rcmip["CO2 AFOLU"] = "CO2|MAGICC AFOLU"
        species_to_rcmip["NOx aviation"] = "NOx|MAGICC Fossil and Industrial|Aircraft"
        species_to_rcmip[
            "Aerosol-radiation interactions"
        ] = "Aerosols-radiation interactions"
        species_to_rcmip[
            "Aerosol-cloud interactions"
        ] = "Aerosols-radiation interactions"
        species_to_rcmip["Contrails"] = "Contrails and Contrail-induced Cirrus"
        species_to_rcmip["Light absorbing particles on snow and ice"] = "BC on Snow"
        species_to_rcmip[
            "Stratospheric water vapour"
        ] = "CH4 Oxidation Stratospheric H2O"
        species_to_rcmip["Land use"] = "Albedo Change"

        species_to_rcmip_copy = copy.deepcopy(species_to_rcmip)

        for specie in species_to_rcmip_copy:
            if specie not in self.species:
                del species_to_rcmip[specie]

        df_emis = pd.read_csv(data_file_path + "/rcmip_emissions_annual.csv")
        df_conc = pd.read_csv(data_file_path + "/rcmip_concentrations_annual.csv")
        df_forc = pd.read_csv(data_file_path + "/rcmip_forcing_annual.csv")

        for scenario in self.scenarios:
            for specie, specie_rcmip_name in species_to_rcmip.items():
                if self.properties_df.loc[specie, "input_mode"] == "emissions":
                    # Grab raw emissions from dataframe
                    emis_in = (
                        df_emis.loc[
                            (df_emis["Scenario"] == scenario)
                            & (
                                df_emis["Variable"].str.endswith(
                                    "|" + specie_rcmip_name
                                )
                            )
                            & (df_emis["Region"] == "World"),
                            "1750":"2500",
                        ]
                        .interpolate(axis=1)
                        .values.squeeze()
                    )

                    # throw error if data missing
                    if emis_in.shape[0] == 0:
                        raise ValueError(
                            f"I can't find a value for scenario={scenario}, variable "
                            f"name ending with {specie_rcmip_name} in the RCMIP "
                            f"emissions database."
                        )

                    # avoid NaNs from outside the interpolation range being mixed into
                    # the results
                    notnan = np.nonzero(~np.isnan(emis_in))

                    # RCMIP are "annual averages"; for emissions this is basically
                    # the emissions over the year, for concentrations and forcing
                    # it would be midyear values. In every case, we can assume
                    # midyear values and interpolate to our time grid.
                    rcmip_index = np.arange(1750.5, 2501.5)
                    interpolator = interp1d(
                        rcmip_index[notnan],
                        emis_in[notnan],
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    emis = interpolator(self.timepoints)

                    # We won't throw an error if the time is out of range for RCMIP,
                    # but we will fill with NaN to allow a user to manually specify
                    # pre- and post- emissions.
                    emis[self.timepoints < 1750] = np.nan
                    emis[self.timepoints > 2501] = np.nan

                    # Parse and possibly convert unit in input file to what FaIR wants
                    unit = df_emis.loc[
                        (df_emis["Scenario"] == scenario)
                        & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_emis["Region"] == "World"),
                        "Unit",
                    ].values[0]
                    emis = emis * (
                        prefix_convert[unit.split()[0]][
                            desired_emissions_units[specie].split()[0]
                        ]
                        * compound_convert[unit.split()[1].split("/")[0]][
                            desired_emissions_units[specie].split()[1].split("/")[0]
                        ]
                        * time_convert[unit.split()[1].split("/")[1]][
                            desired_emissions_units[specie].split()[1].split("/")[1]
                        ]
                    )  # * self.timestep

                    # fill FaIR xarray
                    fill(
                        self.emissions, emis[:, None], specie=specie, scenario=scenario
                    )

                if self.properties_df.loc[specie, "input_mode"] == "concentration":
                    # Grab raw concentration from dataframe
                    conc_in = (
                        df_conc.loc[
                            (df_conc["Scenario"] == scenario)
                            & (
                                df_conc["Variable"].str.endswith(
                                    "|" + specie_rcmip_name
                                )
                            )
                            & (df_conc["Region"] == "World"),
                            "1700":"2500",
                        ]
                        .interpolate(axis=1)
                        .values.squeeze()
                    )

                    # throw error if data missing
                    if conc_in.shape[0] == 0:
                        raise ValueError(
                            f"I can't find a value for scenario={scenario}, variable "
                            f"name ending with {specie_rcmip_name} in the RCMIP "
                            f"concentration database."
                        )

                    # avoid NaNs from outside the interpolation range being mixed into
                    # the results
                    notnan = np.nonzero(~np.isnan(conc_in))

                    # interpolate: this time to timebounds
                    rcmip_index = np.arange(1700.5, 2501.5)
                    interpolator = interp1d(
                        rcmip_index[notnan],
                        conc_in[notnan],
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    conc = interpolator(self.timebounds)

                    # strip out pre- and post-
                    conc[self.timebounds < 1700] = np.nan
                    conc[self.timebounds > 2501] = np.nan

                    # Parse and possibly convert unit in input file to what FaIR wants
                    unit = df_conc.loc[
                        (df_conc["Scenario"] == scenario)
                        & (df_conc["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_conc["Region"] == "World"),
                        "Unit",
                    ].values[0]
                    conc = conc * (
                        mixing_ratio_convert[unit][desired_concentration_units[specie]]
                    )

                    # fill FaIR xarray
                    fill(
                        self.concentration,
                        conc[:, None],
                        specie=specie,
                        scenario=scenario,
                    )

                if self.properties_df.loc[specie, "input_mode"] == "forcing":
                    # Grab raw concentration from dataframe
                    forc_in = (
                        df_forc.loc[
                            (df_forc["Scenario"] == scenario)
                            & (
                                df_forc["Variable"].str.endswith(
                                    "|" + specie_rcmip_name
                                )
                            )
                            & (df_forc["Region"] == "World"),
                            "1750":"2500",
                        ]
                        .interpolate(axis=1)
                        .values.squeeze()
                    )

                    # throw error if data missing
                    if forc_in.shape[0] == 0:
                        raise ValueError(
                            f"I can't find a value for scenario={scenario}, variable "
                            f"name ending with {specie_rcmip_name} in the RCMIP "
                            f"radiative forcing database."
                        )

                    # avoid NaNs from outside the interpolation range being mixed into
                    # the results
                    notnan = np.nonzero(~np.isnan(forc_in))

                    # interpolate: this time to timebounds
                    rcmip_index = np.arange(1750.5, 2501.5)
                    interpolator = interp1d(
                        rcmip_index[notnan],
                        forc_in[notnan],
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    forc = interpolator(self.timebounds)

                    # strip out pre- and post-
                    forc[self.timebounds < 1750] = np.nan
                    forc[self.timebounds > 2501] = np.nan

                    # Forcing so far is always W m-2, but perhaps this will change.

                    # fill FaIR xarray
                    fill(self.forcing, forc[:, None], specie=specie, scenario=scenario)

    def fair_fill_data(self, scenarios):
        self.define_time(
            self.start_year_fair, self.end_year_fair, self.timestep_justice
        )

        # Set up scenarios and configs
        self.define_scenarios(scenarios)

        df_configs = pd.read_csv(
            data_file_path + "/calibrated_constrained_parameters.csv", index_col=0
        )

        # take only the first row from df_configs #this is to select one scenario
        # df_configs = df_configs.iloc[0:1]

        self.define_configs(df_configs.index)
        self.number_of_ensembles = len(df_configs.index)

        species, properties = read_properties(
            filename=data_file_path + "/species_configs_properties_calibration.csv"
        )

        self.define_species(species, properties)

        self.allocate()

        self.fill_from_rcmip_local(data_file_path)

        df_emis = pd.read_csv(data_file_path + "/rcmip_emissions_annual.csv")

        gfed_sectors = [
            "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
            "Emissions|NOx|MAGICC AFOLU|Forest Burning",
            "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
            "Emissions|NOx|MAGICC AFOLU|Peat Burning",
        ]
        for scenario in scenarios:
            self.emissions.loc[dict(specie="NOx", scenario=scenario)] = (
                df_emis.loc[
                    (df_emis["Scenario"] == scenario)
                    & (df_emis["Region"] == "World")
                    & (df_emis["Variable"].isin(gfed_sectors)),
                    str(self.start_year_fair) : str(self.end_year_fair),
                ]
                .interpolate(axis=1)
                .values.squeeze()
                .sum(axis=0)
                * 46.006
                / 30.006
                + df_emis.loc[
                    (df_emis["Scenario"] == scenario)
                    & (df_emis["Region"] == "World")
                    & (df_emis["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
                    str(self.start_year_fair) : str(self.end_year_fair),
                ]
                .interpolate(axis=1)
                .values.squeeze()
                + df_emis.loc[
                    (df_emis["Scenario"] == scenario)
                    & (df_emis["Region"] == "World")
                    & (
                        df_emis["Variable"]
                        == "Emissions|NOx|MAGICC Fossil and Industrial"
                    ),
                    str(self.start_year_fair) : str(self.end_year_fair),
                ]
                .interpolate(axis=1)
                .values.squeeze()
            )[: self.emissions.shape[0], None]

        fill(
            self.climate_configs["ocean_heat_capacity"],
            df_configs.loc[:, "c1":"c3"].values,
        )
        fill(
            self.climate_configs["ocean_heat_transfer"],
            df_configs.loc[:, "kappa1":"kappa3"].values,
        )
        fill(
            self.climate_configs["deep_ocean_efficacy"],
            df_configs["epsilon"].values.squeeze(),
        )
        fill(
            self.climate_configs["gamma_autocorrelation"],
            df_configs["gamma"].values.squeeze(),
        )
        fill(
            self.climate_configs["sigma_eta"], df_configs["sigma_eta"].values.squeeze()
        )
        fill(self.climate_configs["sigma_xi"], df_configs["sigma_xi"].values.squeeze())
        fill(self.climate_configs["seed"], df_configs["seed"])
        fill(self.climate_configs["stochastic_run"], True)
        fill(self.climate_configs["use_seed"], True)
        fill(self.climate_configs["forcing_4co2"], df_configs["F_4xCO2"])

        self.fill_species_configs(
            filename=data_file_path + "/species_configs_properties_calibration.csv"
        )

        # carbon cycle
        fill(
            self.species_configs["iirf_0"],
            df_configs["r0"].values.squeeze(),
            specie="CO2",
        )
        fill(
            self.species_configs["iirf_airborne"],
            df_configs["rA"].values.squeeze(),
            specie="CO2",
        )
        fill(
            self.species_configs["iirf_uptake"],
            df_configs["rU"].values.squeeze(),
            specie="CO2",
        )
        fill(
            self.species_configs["iirf_temperature"],
            df_configs["rT"].values.squeeze(),
            specie="CO2",
        )

        # aerosol indirect
        fill(self.species_configs["aci_scale"], df_configs["beta"].values.squeeze())
        fill(
            self.species_configs["aci_shape"],
            df_configs["shape Sulfur"].values.squeeze(),
            specie="Sulfur",
        )
        fill(
            self.species_configs["aci_shape"],
            df_configs["shape BC"].values.squeeze(),
            specie="BC",
        )
        fill(
            self.species_configs["aci_shape"],
            df_configs["shape OC"].values.squeeze(),
            specie="OC",
        )

        # aerosol direct
        for specie in [
            "BC",
            "CH4",
            "N2O",
            "NH3",
            "NOx",
            "OC",
            "Sulfur",
            "VOC",
            "Equivalent effective stratospheric chlorine",
        ]:
            fill(
                self.species_configs["erfari_radiative_efficiency"],
                df_configs[f"ari {specie}"],
                specie=specie,
            )

        # forcing scaling
        for specie in [
            "CO2",
            "CH4",
            "N2O",
            "Stratospheric water vapour",
            "Contrails",
            "Light absorbing particles on snow and ice",
            "Land use",
        ]:
            fill(
                self.species_configs["forcing_scale"],
                df_configs[f"scale {specie}"].values.squeeze(),
                specie=specie,
            )
        # the halogenated gases all take the same scale factor
        for specie in [
            "CFC-11",
            "CFC-12",
            "CFC-113",
            "CFC-114",
            "CFC-115",
            "HCFC-22",
            "HCFC-141b",
            "HCFC-142b",
            "CCl4",
            "CHCl3",
            "CH2Cl2",
            "CH3Cl",
            "CH3CCl3",
            "CH3Br",
            "Halon-1211",
            "Halon-1301",
            "Halon-2402",
            "CF4",
            "C2F6",
            "C3F8",
            "c-C4F8",
            "C4F10",
            "C5F12",
            "C6F14",
            "C7F16",
            "C8F18",
            "NF3",
            "SF6",
            "SO2F2",
            "HFC-125",
            "HFC-134a",
            "HFC-143a",
            "HFC-152a",
            "HFC-227ea",
            "HFC-23",
            "HFC-236fa",
            "HFC-245fa",
            "HFC-32",
            "HFC-365mfc",
            "HFC-4310mee",
        ]:
            fill(
                self.species_configs["forcing_scale"],
                df_configs["scale minorGHG"].values.squeeze(),
                specie=specie,
            )

        # ozone
        for specie in [
            "CH4",
            "N2O",
            "Equivalent effective stratospheric chlorine",
            "CO",
            "VOC",
            "NOx",
        ]:
            fill(
                self.species_configs["ozone_radiative_efficiency"],
                df_configs[f"o3 {specie}"],
                specie=specie,
            )

        # initial value of CO2 concentration (but not baseline for forcing calculations)
        fill(
            self.species_configs["baseline_concentration"],
            df_configs["co2_concentration_1750"].values.squeeze(),
            specie="CO2",
        )

        initialise(self.concentration, self.species_configs["baseline_concentration"])
        initialise(self.forcing, 0)
        initialise(self.temperature, 0)
        initialise(self.cumulative_emissions, 0)
        initialise(self.airborne_emissions, 0)
