"""
This file contains the neoclassical economic part of the JUSTICE model.
"""


from default_parameters import EconomyDefaults
from enumerations import Economy


class NeoclassicalEconomyModel:

    """
    This module describes the neoclassical economic part of the JUSTICE model.
    """

    def __init__(self):
        # Create an instance of EconomyDefaults
        econ_defaults = EconomyDefaults()

        # Fetch the defaults for neoclassical submodule
        econ_neoclassical_defaults = econ_defaults.get_defaults(
            Economy.NEOCLASSICAL.name
        )

        # Assigning the defaults
        self.capital_elasticity_in_production_function = econ_neoclassical_defaults[
            "capital_elasticity_in_production_function"
        ]
