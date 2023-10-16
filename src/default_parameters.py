"""
This file contains all default parameters for JUSTICE model.
"""


class EconomyDefaults:
    """
    Contains default neoclassical economy-related parameters.
    """

    def __init__(self):
        # Experimental - Call specific defaults based on the Enumeration type
        self.defaults = {
            "NEOCLASSICAL": {
                # Defines the capital elasticity in the production function. #also named gamma and alpha
                "capital_elasticity_in_production_function": 0.3,
                # The annual depreciation rate on capital. #abbreviated to dk
                "depreciation_rate_capital": 0.1,
                # Specifies the elasticity of the marginal utility of consumption. #elasmu in the code or eta in the paper
                "elasticity_of_marginal_utility_of_consumption": 1.45,
                # The discount rate, a.k.a the initial rate of social time preference. #prstp in code
                "pure_rate_of_social_time_preference": 0.015,
                # The Elasticity of Output with respect to Capital. #Zeta in paper, not named in code (hardcoded)
                "elasticity_of_output_to_capital": 0.004,
            },
            "POST_KEYNESIAN": {"capital_elasticity_in_production_function": 0.1},
        }

    def get_defaults(self, type):
        """
        Returns the default economy-related parameters as per the specified type.
        """
        return self.defaults[type]


class DamageDefaults:
    """
    Contains default damage-related parameters.
    """

    def __init__(self):
        self.defaults = {
            "KALKUHL": {
                # Short run temperature change coefficient (originally kw_DT in GAMS)
                "short_run_temp_change_coefficient": 0.00641,
                # Lagged short run temperature change coefficient (originally kw_DT_lag in GAMS)
                "lagged_short_run_temp_change_coefficient": 0.00345,
                # Interaction term temperature change coefficient (originally kw_TDT in GAMS)
                "interaction_term_temp_change_coefficient": -0.00105,
                # Lagged interaction term temperature change coefficient (originally kw_TDT_lag in GAMS)
                "lagged_interaction_term_temp_change_coefficient": -0.000718,
                # Temperature dependent coefficient (originally kw_T in GAMS)
                "temperature_dependent_coefficient": -0.00675,
                # Damage Window - Buffer to hold older temperature and new temperature
                "damage_window": 2,
            },
            "DICE": {},
            "BURKE": {},
        }

    def get_defaults(self, type):
        """
        Returns the default damage-related parameters as per the specified type.
        """
        return self.defaults[type]
