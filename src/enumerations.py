"""
This file contains all custom-made enumerations for JUSTICE model.
"""

from enum import Enum


class SSP(Enum):
    SSP1 = "SSP1"
    SSP2 = "SSP2"
    SSP3 = "SSP3"
    SSP4 = "SSP4"
    SSP5 = "SSP5"


class RCP(Enum):
    RCP19 = "ssp119"  # SSP1-1.9
    RCP26 = "ssp126"  # SSP1-2.6
    RCP45 = "ssp245"  # SSP2-4.5
    RCP70 = "ssp370"  # SSP3-7.0
    RCP34 = "ssp434"  # SSP4-3.4
    RCP60 = "ssp460"  # SSP4-6.0
    RCP34_OVER = "ssp534-over"  # SSP5-3.4-overshoot
    RCP85 = "ssp585"  # SSP5-8.5


# Mapping Dictionary for SSP to RCP

ssp_to_rcp = {
    SSP.SSP1: RCP.RCP19,
    SSP.SSP2: RCP.RCP26,
    SSP.SSP3: RCP.RCP70,
    SSP.SSP4: RCP.RCP34,
    SSP.SSP4: RCP.RCP60,
    SSP.SSP5: RCP.RCP34_OVER,
    SSP.SSP5: RCP.RCP85,
}


class ModelRunSpec(Enum):
    """
    Model Specifications
    """

    PROBABILISTIC = 1  # With EMA Workbench
    DETERMINISTIC = 2  # Without EMA Workbench. Can be used for Validation purposes


class DamageFunction(Enum):
    """
    Damage Functions
    """

    NORDHAUS = 0
    KALKUHL = 1
    BURKE = 2


class Economy(Enum):
    """
    Economy Types
    """

    NEOCLASSICAL = 0
    POST_KEYNESIAN = 1


class WelfareFunction(Enum):
    """
    Social Welfare Functions
    """

    UTILITARIAN = 0
    PRIORITARIAN = 1
    SUFFICIENTARIAN = 2
    EGALITARIAN = 3
