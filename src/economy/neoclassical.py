"""
This file contains the neoclassical economic part of the JUSTICE model.
"""


from src.default_parameters import EconomyDefaults
from src.enumerations import Economy


class NeoclassicalEconomyModel:

    """
    This module describes the neoclassical economic part of the JUSTICE model.
    """

    def __init__(
        self,
        input_dataset,
    ):
        # Create an instance of EconomyDefaults
        econ_defaults = EconomyDefaults()

        # Fetch the defaults for neoclassical submodule
        econ_neoclassical_defaults = econ_defaults.get_defaults(
            Economy.NEOCLASSICAL.name
        )

        # Assign retrieved values to instance variables
        self.capital_elasticity_in_production_function = econ_neoclassical_defaults[
            "capital_elasticity_in_production_function"
        ]
        self.depreciation_rate_capital = econ_neoclassical_defaults[
            "depreciation_rate_capital"
        ]
        self.elasticity_of_marginal_utility_of_consumption = econ_neoclassical_defaults[
            "elasticity_of_marginal_utility_of_consumption"
        ]
        self.pure_rate_of_social_time_preference = econ_neoclassical_defaults[
            "pure_rate_of_social_time_preference"
        ]
        self.elasticity_of_output_to_capital = econ_neoclassical_defaults[
            "elasticity_of_output_to_capital"
        ]

        # test
        print(self.capital_elasticity_in_production_function)
