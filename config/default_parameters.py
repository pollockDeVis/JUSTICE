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
                # The annual depreciation rate on capital. This is yearly, however in RICE50 it is raised to the power of 5 #abbreviated to dk
                "depreciation_rate_capital": 0.1,
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
                # threshold_d in RICE50  # % of GDP for damage when a temperature threshold is reached
                "damage_gdp_ratio_with_threshold": 0.2,
                # threshold_temp in RICE50 # The temperature threshold after which damage occurs
                "temperature_threshold_for_damage": 3.0,
                # threshold_sigma in RICE50 # Variation for the temperature threshold
                "temperature_threshold_variation": 0.05,
                # gradient_d in RICE50 # % of GDP for damage based on temperature change rate
                "damage_gdp_ratio_with_gradient": 0.01,
                # Hardcoded in RICE50 # Essentially a scaling factor for the temperature difference
                "temperature_difference_scaling_factor": 0.35,
                # Hardcoded in RICE50 # Essentially an exponent that acts as the Growth rate of damage
                "damage_growth_rate": 4,
            },
            "DICE": {},
            "BURKE": {},
        }

    def get_defaults(self, type):
        """
        Returns the default damage-related parameters as per the specified type.
        """
        return self.defaults[type]


class SocialWelfareDefaults:
    """
    Contains default social welfare-related parameters.
    """

    def __init__(self):
        self.defaults = {
            "UTILITARIAN": {
                # Normative Parameters
                "risk_aversion": 0,  # Risk aversion parameter.
                # Specifies the elasticity of the marginal utility of consumption. #elasmu in the code or eta in the paper (Stern range: 1-2)
                # Dietz, S., & Stern, N. (2008). Why economic analysis supports strong action on climate change: a response to the Stern Review's critics. Review of Environmental Economics and Policy.
                "elasticity_of_marginal_utility_of_consumption": 1.45,
                # The discount rate, a.k.a the initial rate of social time preference. #prstp in code
                "pure_rate_of_social_time_preference": 0.015,
                # Inequality aversion parameter. #labelled gamma #Range: [0,1.5]; good options: | 0 | 0.5 | 1.45 | 2 |
                "inequality_aversion": 0.0,
                "sufficiency_threshold": 0.0,
                "egality_strictness": 0.0,  # Range: [0,1]
                "limitarian_threshold_emissions": 0.0,
                "limitarian_start_year_of_remaining_budget": 0,
            },
            "PRIORITARIAN": {
                "risk_aversion": 0,  # Risk aversion parameter.
                "elasticity_of_marginal_utility_of_consumption": 1.45,
                "pure_rate_of_social_time_preference": 0.0,
                "inequality_aversion": 2.0,
                "sufficiency_threshold": 0.0,
                "egality_strictness": 0.0,  # Range: [0,1]
                "limitarian_threshold_emissions": 0.0,
                "limitarian_start_year_of_remaining_budget": 0,
            },
            "SUFFICIENTARIAN": {  # Sufficientarian can be either Utilitarian above threshold or Prioritarian below threshold
                "risk_aversion": 0,  # Risk aversion parameter.
                "elasticity_of_marginal_utility_of_consumption": 1.45,
                "pure_rate_of_social_time_preference": 0.015,
                "inequality_aversion": 0.0,
                "sufficiency_threshold": (
                    (1.25 * 365.25) / 1e3
                ),  # World bank stipulated the poverty line of US$1.25 for 2005 USD PPP.
                # Consumption in JUSTICE is yearly (in thousands $2005 PPP), hence we simply
                # multiply this poverty line rate with average days in a year and hence  in JUSTICE will be 1.25 * 365.25/1000
                "egality_strictness": 0.0,  # Range: [0,1]
                "limitarian_threshold_emissions": 0.0,
                "limitarian_start_year_of_remaining_budget": 0,
            },
            "EGALITARIAN": {
                "risk_aversion": 0,  # Risk aversion parameter.
                "elasticity_of_marginal_utility_of_consumption": 1.45,
                "pure_rate_of_social_time_preference": 0.0,
                "inequality_aversion": 0.5,  # 2.0, #NOTE: Keeping it 0.5 to make it distinct from prioritarian
                "sufficiency_threshold": 0.0,
                "egality_strictness": 1.0,  # Range: [0,1]
                "limitarian_threshold_emissions": 0.0,
                "limitarian_start_year_of_remaining_budget": 0,
            },
            # Cox, P. M., Williamson, M. S., Friedlingstein, P., Jones, C. D., Raoult, N., Rogelj, J., & Varney, R. M. (2024).
            # Emergent constraints on carbon budgets as a function of global warming. Nature Communications, 15(1), 1885.
            "LIMITARIAN_UTILITARIAN": {
                "risk_aversion": 0,  # Risk aversion parameter.
                "elasticity_of_marginal_utility_of_consumption": 1.45,
                "pure_rate_of_social_time_preference": 0.015,
                "inequality_aversion": 0.0,
                "sufficiency_threshold": 0.0,
                "egality_strictness": 0.0,  # Range: [0,1]
                "limitarian_threshold_emissions": 422.0,  # Emergent constraints on cumulative carbon budgets, and remaining carbon budgets from the beginning of 2020 for 2°C. Confidence Intervals: [258, 586] PgC or GtC
                "limitarian_start_year_of_remaining_budget": 2020,
            },
            "LIMITARIAN_PRIORITARIAN": {
                "risk_aversion": 0,  # Risk aversion parameter.
                "elasticity_of_marginal_utility_of_consumption": 1.45,
                "pure_rate_of_social_time_preference": 0.0,
                "inequality_aversion": 2.0,
                "sufficiency_threshold": 0.0,
                "egality_strictness": 0.0,  # Range: [0,1]
                "limitarian_threshold_emissions": 422.0,  # Emergent constraints on cumulative carbon budgets, and remaining carbon budgets from the beginning of 2020 for 2°C. Confidence Intervals: [258, 586] PgC or GtC
                "limitarian_start_year_of_remaining_budget": 2020,
            },
        }

    def get_defaults(self, type):
        """
        Returns the default social welfare-related parameters as per the specified type.
        """
        return self.defaults[type]


class AbatementDefaults:
    """
    Contains default abatement-related parameters.
    """

    def __init__(self):
        self.defaults = {
            "ENERDATA": {
                # MxKali #calibrated correction multiplier starting value
                "calibrated_correction_multiplier_starting_value": 0.492373,
                # pback #Cost of backstop 2010$ per tCO2 in 2015 #DICE2013: 344     #DICE2016: 550 #DICE2023: 515 # This is dependent on the SSP scenarios
                "backstop_cost": 550,  # 515, #TODO using 550 temporarily
                # gback #Initial cost decline backstop cost per period #DICE2013: 0.05    #DICE2016: 0.025 #DICE2023: 0.01 (2020-2050) 0.001 (2050 onwards)
                "backstop_cost_decline_rate_per_5_year": 0.025,
                # tstart_pbtransition #first timestep without Enerdata projections
                "transition_year_start": 2045,
                # tend_pbtransition #time of full-convergence to backstop curve
                "transition_year_end": 2125,
                # klogistic
                "logistic_transition_speed_per_5_year": 0.25,
                # expcost2 #Exponent of control cost function
                "exponential_control_cost_function": 2.8,
            },
            "DICE": {},
        }

    def get_defaults(self, type):
        """
        Returns the default abatement-related parameters as per the specified type.
        """
        return self.defaults[type]
