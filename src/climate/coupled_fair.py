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
