"""
This file contains all custom-made enumerations for JUSTICE model.
"""

from enum import Enum


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
