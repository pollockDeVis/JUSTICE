"""
This file contains all custom-made enumerations for JUSTICE model.
"""

from enum import Enum, IntEnum


class WelfareFunction(Enum):
    """
    Social Welfare Functions
    Tuple: (index, string)
    """

    UTILITARIAN = (0, "UTILITARIAN")
    PRIORITARIAN = (1, "PRIORITARIAN")
    SUFFICIENTARIAN = (2, "SUFFICIENTARIAN")
    EGALITARIAN = (3, "EGALITARIAN")

    @staticmethod
    def from_index(index):
        for enum in WelfareFunction:
            if enum.value[0] == index:
                return enum
        return None


class SSP(IntEnum):
    SSP1 = 0
    SSP2 = 1
    SSP3 = 2
    SSP4 = 3
    SSP5 = 4


# TODO: Add the pretty strings in the tuple like (0, SSP.SSP1, "ssp119", "SSP1-RCP1.9")
class Scenario(Enum):
    SSP119 = (0, SSP.SSP1, "ssp119")  # SSP1-RCP1.9
    SSP126 = (1, SSP.SSP1, "ssp126")  # SSP1-RCP2.6
    SSP245 = (2, SSP.SSP2, "ssp245")  # SSP2-RCP4.5
    SSP370 = (3, SSP.SSP3, "ssp370")  # SSP3-RCP7.0
    SSP434 = (4, SSP.SSP4, "ssp434")  # SSP4-RCP3.4
    SSP460 = (5, SSP.SSP4, "ssp460")  # SSP4-RCP6.0
    SSP534 = (6, SSP.SSP5, "ssp534-over")  # SSP5-RCP3.4-overshoot
    SSP585 = (7, SSP.SSP5, "ssp585")  # SSP5-RCP8.5


def get_climate_scenario(index):
    scenarios = list(Scenario)
    if index < 0 or index >= len(scenarios):
        raise ValueError(
            "Index out of range. It should be between 0 and " + str(len(scenarios) - 1)
        )
    return scenarios[index].value[2]


def get_economic_scenario(index):
    scenarios = list(Scenario)
    if index < 0 or index >= len(scenarios):
        raise ValueError(
            "Index out of range. It should be between 0 and " + str(len(scenarios) - 1)
        )
    return scenarios[index].value[1].value


def get_welfare_function_name(index):
    welfare_functions = list(WelfareFunction)
    if index < 0 or index >= len(welfare_functions):
        raise ValueError(
            "Index out of range. It should be between 0 and "
            + str(len(welfare_functions) - 1)
        )
    return welfare_functions[index].value[1]


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


class Abatement(Enum):
    """
    Abatement Types
    """

    ENERDATA = 0
    DICE = 1
