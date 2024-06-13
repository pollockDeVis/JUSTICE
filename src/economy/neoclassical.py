"""
This file contains the neoclassical economic part of the JUSTICE model.
"""

from typing import Any
from scipy.interpolate import interp1d
import numpy as np
import copy


from config.default_parameters import EconomyDefaults
from src.util.enumerations import Economy, get_economic_scenario


class NeoclassicalEconomyModel:
    """
    This class describes the neoclassical economic part of the JUSTICE model.
    """

    def __init__(
        self,
        input_dataset,
        time_horizon,
        scenario,
        climate_ensembles,
        elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference,
        **kwargs,  # variable keyword argument so that we can analyze uncertainty range of any parameters
    ):

        # Create an instance of EconomyDefaults
        econ_defaults = EconomyDefaults()

        # Saving the climate ensembles
        self.NUM_OF_ENSEMBLES = climate_ensembles

        # Saving the scenario
        self.scenario = get_economic_scenario(scenario)

        # Fetch the defaults for neoclassical submodule
        econ_neoclassical_defaults = econ_defaults.get_defaults(
            Economy.NEOCLASSICAL.name
        )

        # Assign retrieved values to instance variables if kwargs is empty
        self.capital_elasticity_in_production_function = kwargs.get(
            "capital_elasticity_in_production_function",
            econ_neoclassical_defaults["capital_elasticity_in_production_function"],
        )
        # NOTE: Depreciation rate is for every year
        self.depreciation_rate_capital = kwargs.get(
            "depreciation_rate_capital",
            econ_neoclassical_defaults["depreciation_rate_capital"],
        )
        self.elasticity_of_output_to_capital = kwargs.get(
            "elasticity_of_output_to_capital",
            econ_neoclassical_defaults["elasticity_of_output_to_capital"],
        )

        self.elasticity_of_marginal_utility_of_consumption = (
            elasticity_of_marginal_utility_of_consumption
        )
        self.pure_rate_of_social_time_preference = pure_rate_of_social_time_preference

        self.region_list = input_dataset.REGION_LIST
        self.gdp_array = copy.deepcopy(input_dataset.GDP_ARRAY)
        self.population_array = copy.deepcopy(input_dataset.POPULATION_ARRAY)

        # Assert that the number of scenarios in GDP and Population are the same.
        assert (
            self.gdp_array.shape[2] == self.population_array.shape[2]
        ), "Number of scenarios in GDP and Population are not the same."

        # Selecting only the required scenario
        self.gdp_array = self.gdp_array[:, :, self.scenario]
        self.population_array = self.population_array[:, :, self.scenario]

        self.capital_init_arr = input_dataset.CAPITAL_INIT_ARRAY
        self.savings_rate_init_arr = (
            input_dataset.SAVING_RATE_INIT_ARRAY
        )  # Probably won't be used. Can be passed while setting up the Savings Rate Lever

        # Load PPP2MER conversion factor. Conversion factor for Purchasing Power Parity (PPP) to Market Exchange Rate (MER)
        # PPP is widely used to calculate international ineqquality (Milanovic, 2005)
        # When using PPP rather than market exchange rate (MER),
        # poor countries get a income boost and difference between RICH and POOR incomes is LESS than MER.

        self.ppp_to_mer = input_dataset.PPP_TO_MER_CONVERSION_FACTOR
        self.mer_to_ppp = 1 / self.ppp_to_mer  # Conversion of MER to PPP
        # mer_to_ppp is a 2D array. Need to convert it to (regions, 1)
        self.mer_to_ppp = self.mer_to_ppp[:, 0:1]

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Initializing the capital and TFP array This will be of the same shape as data with 5 year timestep
        # TODO check if this is needed
        self.capital_tfp_data = np.zeros(
            (len(self.region_list), len(self.data_time_horizon))
        )

        # Calculate the baseline TFP
        self.tfp = self.initialize_tfp(
            fixed_savings_rate=self.get_fixed_savings_rate(self.data_time_horizon),
        )

        # Check if timestep is not equal to data timestep #If not, then interpolate

        if self.timestep != self.data_timestep:
            # Interpolate GDP
            self._interpolate_tfp()
            self._interpolate_gdp()
            self._interpolate_population()

        # Initializing the capital array Unit: Trill 2005 USD PPP
        self.capital = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Intializing the investment array Unit: Trill 2005 USD PPP / year
        self.investment = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the gross output array Unit: Trill 2005 USD PPP / year
        self.gross_output = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the net output array Unit: Trill 2005 USD PPP / year
        self.net_output = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the damage array Unit: Trill 2005 USD PPP / year
        self.damage = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the abatement array Unit: Trill 2005 USD PPP / year
        self.abatement = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Calculate the baseline per capita growth #TODO to be used in the complex version of Damage Function
        # Initializing the 4D array for baseline per capita growth
        # self.baseline_per_capita_growth = np.zeros(
        #     (
        #         len(self.region_list),
        #         len(self.model_time_horizon),
        #         self.NUM_OF_ENSEMBLES,
        #     )
        # )
        # self.calculate_baseline_per_capita_growth()

    def run(self, timestep, savings_rate):
        # TODO: Check savings rate shape for 2D or 3D
        # Reshaping savings rate
        if len(savings_rate.shape) == 1:
            savings_rate = savings_rate.reshape(-1, 1)

        if timestep == 0:  # Initialize the capital and investment

            # Initialize the investment
            self._calculate_investment(timestep, savings_rate)

            # Initialize the capital
            self._calculate_capital(timestep)

            # Calculate the Output
            self._calculate_output(timestep)

        else:
            # Calculate the gross output
            self._calculate_output(timestep)

        return self.gross_output[:, timestep, :]

    def get_optimal_long_run_savings_rate(self):
        """
        This method returns the optimal long run savings rate.
        """
        # Calculate the Optimal long-run Savings Rate
        # This will depend on the input paramters. This is also a upper limit of the savings rate
        optimal_long_run_savings_rate = (
            (self.depreciation_rate_capital + self.elasticity_of_output_to_capital)
            / (
                self.depreciation_rate_capital
                + self.elasticity_of_output_to_capital
                * self.elasticity_of_marginal_utility_of_consumption
                + self.pure_rate_of_social_time_preference
            )
        ) * self.capital_elasticity_in_production_function

        return optimal_long_run_savings_rate

    def initialize_tfp(self, fixed_savings_rate):
        """
        This method initializes the TFP.
        """
        # Calculate the investment_tfp
        investment_tfp = fixed_savings_rate * self.gdp_array

        self.capital_tfp_data[:, 0] = (self.capital_init_arr * self.mer_to_ppp).reshape(
            -1
        )

        for timestep in range(1, len(self.data_time_horizon)):
            # Calculate the capital_tfp
            self.capital_tfp_data[:, timestep] = self.capital_tfp_data[
                :, timestep - 1
            ] * np.power(
                (1 - self.depreciation_rate_capital),
                (self.data_timestep),
            ) + investment_tfp[
                :, timestep - 1
            ] * (
                (self.data_timestep)
            )

        # Calculate the TFP
        tfp = self.gdp_array / (
            np.power(
                (self.population_array / 1000),
                (1 - self.capital_elasticity_in_production_function),
            )
            * np.power(
                self.capital_tfp_data,
                self.capital_elasticity_in_production_function,
            )
        )

        return tfp

    def get_fixed_savings_rate(self, time_horizon):
        """
        This method returns the fixed savings rate. It takes the intial savings rate and increases
        it linearly to the optimal long run savings rate.
        """

        # fixed_savings_rate Validated with RICE50 for timestep 1 and 5
        fixed_savings_rate = np.copy(self.savings_rate_init_arr).reshape(-1, 1)

        # Calculate the Optimal long-run Savings Rate
        # This will depend on the input paramters. This is also a upper limit of the savings rate
        optimal_long_run_savings_rate = self.get_optimal_long_run_savings_rate()

        # for i, years in enumerate(set_year):
        for t in range(2, (len(time_horizon) + 1)):
            next_rate = self.savings_rate_init_arr + (
                optimal_long_run_savings_rate - self.savings_rate_init_arr
            ) * ((t - 1) / (len(time_horizon) - 1))
            # append to the fixed savings rate array for each year
            fixed_savings_rate = np.column_stack((fixed_savings_rate, next_rate))

        return fixed_savings_rate

    def _calculate_investment(self, timestep, savings_rate):
        if len(savings_rate.shape) == 1:
            savings_rate = savings_rate.reshape(-1, 1)

        if timestep == 0:
            self.investment[:, timestep, :] = (
                self.savings_rate_init_arr * self.gdp_array[:, timestep].reshape(-1, 1)
            )
        else:
            self.investment[:, timestep, :] = (
                savings_rate * self.net_output[:, timestep, :]
            )

    def _calculate_capital(self, timestep):  # Capital is calculated one timestep ahead
        if timestep < (len(self.model_time_horizon) - 1):
            if timestep == 0:
                # Initalize capital tfp
                self.capital[:, timestep, :] = self.capital_init_arr * self.mer_to_ppp

                self.capital[:, timestep + 1, :] = self.capital[
                    :, timestep, :
                ] * np.power(
                    (1 - self.depreciation_rate_capital),
                    (self.timestep),
                ) + self.investment[
                    :, timestep, :
                ] * (
                    (self.timestep)
                )
            else:
                # Calculate capital
                self.capital[:, timestep + 1, :] = self.capital[
                    :, timestep, :
                ] * np.power(
                    (1 - self.depreciation_rate_capital),
                    (self.timestep),
                ) + self.investment[
                    :, timestep, :
                ] * (
                    (self.timestep)
                )

            # Check if any element is negative in capital
            # Need to do this because capital can negative with very high abatement rates leading to
            # propagation of nan values in gross output, net output, consumption and utility
            self.capital = np.where(self.capital < 0, 0, self.capital)

    def _calculate_output(self, timestep):
        # Calculate the Output based on gross output

        self.gross_output[:, timestep, :] = self.tfp[:, timestep, np.newaxis] * (
            np.power(
                self.capital[:, timestep, :],
                self.capital_elasticity_in_production_function,
            )
            * np.power(
                (self.population_array[:, timestep, np.newaxis] / 1000),
                (1 - self.capital_elasticity_in_production_function),
            )
        )
        # Setting net output to gross output before any damage or abatement
        self.net_output[:, timestep, :] = self.gross_output[:, timestep, :]

    def _apply_damage_to_output(self, timestep, damage_fraction):
        """
        This method applies damage to the output.
        Damage calculated
        """

        # Mutiplying damage to get Net Output # YGROSS(t,n) * (1 - DAMFRAC_UNBOUNDED(t,n))
        self.damage[:, timestep, :] = (
            self.gross_output[:, timestep, :] * damage_fraction
        )

        self.net_output[:, timestep, :] -= self.damage[:, timestep, :]

    def _apply_abatement_to_output(self, timestep, abatement):
        """
        This method applies abatement to the output.
        """
        self.abatement[:, timestep, :] = abatement

        self.net_output[:, timestep, :] -= self.abatement[:, timestep, :]

    def feedback_loop_for_economic_output(
        self, timestep, savings_rate, damage_fraction, abatement
    ):
        """
        This method calculates the capital and investment.
        """
        # TODO: Add checks for whether damage and abatement are enabled
        # TODO: Check shape of savings rate
        # Apply damage to the output
        self._apply_damage_to_output(timestep, damage_fraction)

        # Apply abatement to the output
        self._apply_abatement_to_output(timestep, abatement)

        # Calculate the investment
        self._calculate_investment(timestep, savings_rate)

        # Calculate the capital
        self._calculate_capital(timestep)

    def calculate_consumption(self, savings_rate):  # Validated
        """
        This method calculates the consumption.
        Unit: Trill 2005 USD PPP / year

        """
        # TODO: Check shape of savings rate

        # Reshape savings rate from 2D to 3D
        savings_rate = savings_rate[:, :, np.newaxis]  # Validated

        investment = (
            savings_rate * self.net_output
        )  # Validated - Net output nan on region 25
        consumption = self.net_output - investment  # Valdated - Shape (57, 286, 1001)

        return consumption

    def calculate_consumption_per_timestep(self, savings_rate, timestep):  #
        """
        This method calculates the consumption per timestep.
        """
        # TODO: Check shape of savings rate
        # Reshape savings rate from 1D to 2D
        savings_rate = savings_rate[:, np.newaxis]

        investment = savings_rate * self.net_output[:, timestep, :]
        consumption_per_timestep = self.net_output[:, timestep, :] - investment

        return consumption_per_timestep

    def get_consumption_per_capita_per_timestep(self, savings_rate, timestep):

        consumption_per_timestep = self.calculate_consumption_per_timestep(
            savings_rate, timestep
        )
        consumption_per_capita_per_timestep = (
            1e3
            * consumption_per_timestep
            / self.population_array[:, timestep, np.newaxis]
        )

        return consumption_per_capita_per_timestep

    def get_damage_cost_per_capita_per_timestep(self, timestep):

        damages_per_timestep = self.damage[:, timestep, :]
        damages_per_capita_per_timestep = (
            1e3 * damages_per_timestep / self.population_array[:, timestep, np.newaxis]
        )

        return damages_per_capita_per_timestep

    def get_abatement_cost_per_capita_per_timestep(self, timestep):

        abatement_per_timestep = self.abatement[:, timestep, :]
        abatement_per_capita_per_timestep = (
            1e3
            * abatement_per_timestep
            / self.population_array[:, timestep, np.newaxis]
        )

        return abatement_per_capita_per_timestep

    def get_net_output(self):
        return self.net_output

    def get_gross_output(self):
        return self.gross_output

    def get_net_output_by_timestep(self, timestep):
        return self.net_output[:, timestep, :]

    def get_abatement(self):
        return self.abatement

    def get_damages(self):
        return self.damage

    def get_population(self):  # Validated

        population = self.population_array
        # Convert population to 3D by broadcasting across climate ensembles
        population = population[:, :, np.newaxis]
        population = np.broadcast_to(
            population,
            (
                population.shape[0],
                population.shape[1],
                self.NUM_OF_ENSEMBLES,
            ),
        )
        return population

    def get_consumption_per_capita(self, savings_rate):
        """
        This method calculates the consumption per capita.
        Unit: Thousands 2005 USD PPP per year
        """
        consumption = self.calculate_consumption(savings_rate)
        consumption_per_capita = (  # Validated
            1e3 * consumption / self.population_array[:, :, np.newaxis]
        )

        return consumption_per_capita

    def get_damage_cost_per_capita(self):
        """
        This method calculates the damages per capita.
        Unit: Thousands 2005 USD PPP per year
        """
        damages_per_capita = (  # Validated
            1e3 * self.damage / self.population_array[:, :, np.newaxis]
        )

        return damages_per_capita

    def get_abatement_cost_per_capita(self):
        """
        This method calculates the abatement cost per capita.
        Unit: Thousands 2005 USD PPP per year
        """
        abatement_cost_per_capita = (  # Validated
            1e3 * self.abatement / self.population_array[:, :, np.newaxis]
        )

        return abatement_cost_per_capita

    def _interpolate_gdp(self):
        interp_data = np.zeros(
            (
                self.gdp_array.shape[0],
                len(self.model_time_horizon),
                # self.gdp_array.shape[2],
            )
        )

        for i in range(self.gdp_array.shape[0]):
            # for j in range(self.gdp_array.shape[2]):
            f = interp1d(
                self.data_time_horizon, self.gdp_array[i, :], kind="linear"  # , j
            )
            interp_data[i, :] = f(self.model_time_horizon)  # , j

        self.gdp_array = interp_data

    def _interpolate_population(self):
        interp_data = np.zeros(
            (
                self.population_array.shape[0],
                len(self.model_time_horizon),
                # self.population_array.shape[2],
            )
        )

        for i in range(self.population_array.shape[0]):
            # for j in range(self.population_array.shape[2]):
            f = interp1d(
                self.data_time_horizon,
                self.population_array[i, :],  # , j
                kind="linear",
            )
            interp_data[i, :] = f(self.model_time_horizon)  # , j

        self.population_array = interp_data

    def _interpolate_tfp(self):
        interp_data = np.zeros(
            (
                self.tfp.shape[0],
                len(self.model_time_horizon),
            )
        )

        for i in range(self.tfp.shape[0]):
            f = interp1d(
                self.data_time_horizon,
                self.tfp[i, :],
                kind="linear",
            )
            interp_data[i, :] = f(self.model_time_horizon)

        self.tfp = interp_data

    # NOTE: Not implemented yet
    # def calculate_baseline_per_capita_growth(self):
    #     """
    #     This method calculates the baseline per capita growth.
    #     """
    #     # TODO : needs modification after change of gdp 4D array
    #     # Calculate the baseline per capita growth
    #     self.baseline_per_capita_growth = self.gdp / self.population

    #     # Divide all t+1 timestep by preceding timestep
    #     self.baseline_per_capita_growth[:, 1:, :, :] /= self.baseline_per_capita_growth[
    #         :, :-1, :, :
    #     ]
    #     # Take the power of 1/timestep and subtract 1
    #     self.baseline_per_capita_growth **= 1 / self.timestep
    #     self.baseline_per_capita_growth -= 1

    #     # Set the first timestep to zero
    #     self.baseline_per_capita_growth[:, 0, :, :] = 0

    # NOTE: Not implemented yet
    # def calculate_social_cost_of_carbon(
    #     self, fossil_and_land_use_emissions, savings_rate, regional=True
    # ):
    #     """
    #     This method calculates the social cost of carbon.
    #     """
    #     # TODO: Calculations are currently not correct. Need to fix it.

    #     # total_emissions = emissions + land_use_emissions
    #     # print("Total Emissions", fossil_and_land_use_emissions.shape)

    #     consumption = self.calculate_consumption(savings_rate)
    #     # print("consumption", consumption.shape)

    #     # Calculate the social cost of carbon
    #     #  scc[t, n] = (-1000 * eq_E[t, n]) / eq_cc[t, n]

    #     emissions_marginal = np.diff(fossil_and_land_use_emissions, axis=1)
    #     consumption_marginal = np.diff(consumption, axis=1)

    #     # print("emissions_marginal", emissions_marginal[0, 0, 0])
    #     # print("consumption_marginal", consumption_marginal[0, 0, 0])

    #     social_cost_of_carbon = (emissions_marginal / consumption_marginal) * -1000

    #     return social_cost_of_carbon

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
