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
        climate_ensembles,
        **kwargs,  # variable keyword argument so that we can analyze uncertainty range of any parameters
    ):  # TODO maybe move the kwargs to the calculate economy
        # Create an instance of EconomyDefaults
        econ_defaults = EconomyDefaults()

        # Saving the climate ensembles
        self.NUM_OF_ENSEMBLES = climate_ensembles

        # Fetch the defaults for neoclassical submodule
        econ_neoclassical_defaults = econ_defaults.get_defaults(
            Economy.NEOCLASSICAL.name
        )

        # Assign retrieved values to instance variables if kwargs is empty

        self.capital_elasticity_in_production_function = kwargs.get(
            "capital_elasticity_in_production_function",
            econ_neoclassical_defaults["capital_elasticity_in_production_function"],
        )
        self.depreciation_rate_capital = kwargs.get(
            "depreciation_rate_capital",
            econ_neoclassical_defaults["depreciation_rate_capital"],
        )

        self.elasticity_of_output_to_capital = kwargs.get(
            "elasticity_of_output_to_capital",
            econ_neoclassical_defaults["elasticity_of_output_to_capital"],
        )

        self.region_list = input_dataset.REGION_LIST
        self.gdp_array = copy.deepcopy(input_dataset.GDP_ARRAY)
        self.population_arr = copy.deepcopy(input_dataset.POPULATION_ARRAY)

        self.capital_init_arr = input_dataset.CAPITAL_INIT_ARRAY
        self.savings_rate_init_arr = (
            input_dataset.SAVING_RATE_INIT_ARRAY
        )  # Probably won't be used. Can be passed while setting up the Savings Rate Lever
        self.ppp_to_mer = input_dataset.PPP_TO_MER_CONVERSION_FACTOR
        self.mer_to_ppp = 1 / self.ppp_to_mer  # Conversion of MER to PPP

        # mer_to_ppp is a 2D array. Need to convert it to (regions, 1)
        self.mer_to_ppp = self.mer_to_ppp[:, 0:1]

        self.timestep = time_horizon.timestep
        self.data_timestep = time_horizon.data_timestep
        self.data_time_horizon = time_horizon.data_time_horizon
        self.model_time_horizon = time_horizon.model_time_horizon

        # Check if timestep is not equal to data timestep #If not, then interpolate

        if self.timestep != self.data_timestep:
            # Interpolate GDP
            self._interpolate_gdp()
            self._interpolate_population()

        # Initializing the capital and TFP array
        self.capital_tfp = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the investment array
        self.investment_tfp = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )
        # Initializing the TFP array
        self.tfp = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the output array
        self.output = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # TODO: Create a Gross Output array instead of damage fraction array

        # Initializing the damage fraction array # TODO: Need to remove this
        self.damage_fraction = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the damage array
        self.damage = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initializing the abatement array
        self.abatement = np.zeros(
            (len(self.region_list), len(self.model_time_horizon), self.NUM_OF_ENSEMBLES)
        )

        # Initial 4D array for gdp and population.
        self.gdp = np.zeros(
            (
                len(self.region_list),
                len(self.model_time_horizon),
                self.NUM_OF_ENSEMBLES,
                self.gdp_array.shape[2],
            )
        )
        self.population = np.zeros(
            (
                len(self.region_list),
                len(self.model_time_horizon),
                self.NUM_OF_ENSEMBLES,
                self.population_arr.shape[2],
            )
        )

        # Initializing the 4D array for baseline per capita growth
        self.baseline_per_capita_growth = np.zeros(
            (
                len(self.region_list),
                len(self.model_time_horizon),
                self.NUM_OF_ENSEMBLES,
                self.population_arr.shape[2],
            )
        )
        # Assert that the number of scenarios in GDP and Population are the same.
        assert (
            self.gdp_array.shape[2] == self.population_arr.shape[2]
        ), "Number of scenarios in GDP and Population are not the same."

        # Loop through the scenarios to broadcast each one to the ensemble dimension.
        for idx in range(self.gdp_array.shape[2]):
            self.gdp[:, :, :, idx] = np.broadcast_to(
                self.gdp_array[:, :, idx, np.newaxis],
                (
                    self.gdp_array.shape[0],
                    self.gdp_array.shape[1],
                    self.NUM_OF_ENSEMBLES,
                ),
            )
            self.population[:, :, :, idx] = np.broadcast_to(
                self.population_arr[:, :, idx, np.newaxis],
                (
                    self.population_arr.shape[0],
                    self.population_arr.shape[1],
                    self.NUM_OF_ENSEMBLES,
                ),
            )

        # Calculate the baseline per capita growth #TODO to be used in the complex version of Damage Function
        self.calculate_baseline_per_capita_growth()

    def run(self, scenario, timestep, savings_rate):  # **kwargs
        scenario = get_economic_scenario(scenario)
        # Reshaping savings rate
        if len(savings_rate.shape) == 1:
            savings_rate = savings_rate.reshape(-1, 1)

        if timestep == 0:
            self.investment_tfp[:, 0, :] = (
                self.savings_rate_init_arr * self.gdp[:, 0, :, scenario]
            )

            # Initalize capital tfp
            self.capital_tfp[:, 0, :] = self.capital_init_arr * self.mer_to_ppp

            # Calculate the TFP
            self._calculate_tfp(timestep, scenario)

            # Calculate the Output
            self._calculate_output(timestep, scenario)

        else:
            # Calculate the investment_tfp
            self.investment_tfp[:, timestep, :] = (
                savings_rate * self.gdp[:, timestep, :, scenario]
            )

            # Calculate capital_tfp

            self.capital_tfp[:, timestep, :] = self.capital_tfp[
                :, timestep - 1, :
            ] * np.power(
                (1 - self.depreciation_rate_capital),
                (self.timestep / self.data_timestep),
            ) + self.investment_tfp[
                :, timestep - 1, :
            ] * (
                (self.timestep / self.data_timestep)
            )

            # Calculate the TFP
            self._calculate_tfp(timestep, scenario)

            # Calculate the Output based on gross output
            self._calculate_output(timestep, scenario)

        return self.output[:, timestep, :]

    def get_optimal_long_run_savings_rate(
        self,
        elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference,
    ):
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
                * elasticity_of_marginal_utility_of_consumption
                + pure_rate_of_social_time_preference
            )
        ) * self.capital_elasticity_in_production_function

        return optimal_long_run_savings_rate

    def get_fixed_savings_rate(
        self,
        elasticity_of_marginal_utility_of_consumption,
        pure_rate_of_social_time_preference,
    ):
        """
        This method returns the fixed savings rate. It takes the intial savings rate and increases
        it linearly to the optimal long run savings rate.
        """
        # Calculate the Optimal long-run Savings Rate
        # This will depend on the input paramters. This is also a upper limit of the savings rate

        fixed_savings_rate = np.copy(self.savings_rate_init_arr).reshape(-1, 1)
        # fixed_savings_rate Validated with RICE50 for timestep 1 and 5

        optimal_long_run_savings_rate = self.get_optimal_long_run_savings_rate(
            elasticity_of_marginal_utility_of_consumption,
            pure_rate_of_social_time_preference,
        )

        # for i, years in enumerate(set_year):
        for i in range(len(self.model_time_horizon)):
            t = i + 1  # index starts at 0, so add 1 to get the year
            if t != 1:  # no need to repeat for the first year
                next_rate = self.savings_rate_init_arr + (
                    optimal_long_run_savings_rate - self.savings_rate_init_arr
                ) * ((t - 1) / (len(self.model_time_horizon) - 1))
                # append to the fixed savings rate array for each year
                fixed_savings_rate = np.column_stack((fixed_savings_rate, next_rate))

        return fixed_savings_rate

    def _calculate_tfp(self, timestep, scenario):
        # Calculate the TFP
        self.tfp[:, timestep, :] = self.gdp[:, timestep, :, scenario] / (
            np.power(
                (self.population[:, timestep, :, scenario] / 1000),
                (1 - self.capital_elasticity_in_production_function),
            )
            * np.power(
                self.capital_tfp[:, timestep, :],
                self.capital_elasticity_in_production_function,
            )
        )

    def _calculate_output(self, timestep, scenario):
        # Calculate the Output based on gross output

        self.output[:, timestep, :] = self.tfp[:, timestep, :] * (
            np.power(
                self.capital_tfp[:, timestep, :],
                self.capital_elasticity_in_production_function,
            )
            * np.power(
                (self.population[:, timestep, :, scenario] / 1000),
                (1 - self.capital_elasticity_in_production_function),
            )
        )
        # Apply damages to output
        # Check if the timestep is not zero
        # Check if timestep is 0 or 1

        # if timestep == 0 or timestep == 1:  # Damage is zero in the first timestep
        # if timestep <= 2:
        #     self.output[:, timestep, :] = self.output[:, timestep, :]
        # else:
        # self.damage[:, timestep, :] = (
        #     self.output[:, timestep, :] * self.damage_fraction[:, timestep, :]
        # )
        # # Mutiplying damage to get Net Output # YGROSS(t,n) * (1 - DAMFRAC_UNBOUNDED(t,n))
        # self.output[:, timestep, :] = (
        #     self.output[:, timestep, :] - self.damage[:, timestep, :]
        # )
        # Subtract abatement from output
        # self.output[:, timestep, :] = (
        #     self.output[:, timestep, :] - self.abatement[:, timestep, :]
        # )

    def apply_damage_to_output(self, timestep, damage):
        """
        This method applies damage to the output.
        Damage calculated
        """
        self.damage_fraction[:, timestep, :] = damage

        self.damage[:, timestep, :] = (
            self.output[:, timestep, :] * self.damage_fraction[:, timestep, :]
        )
        # Mutiplying damage to get Net Output # YGROSS(t,n) * (1 - DAMFRAC_UNBOUNDED(t,n))
        self.output[:, timestep, :] = (
            self.output[:, timestep, :] - self.damage[:, timestep, :]
        )

    def apply_abatement_to_output(self, timestep, abatement):
        """
        This method applies abatement to the output.
        """
        self.abatement[:, timestep, :] = abatement

        self.output[:, timestep, :] = (
            self.output[:, timestep, :] - self.abatement[:, timestep, :]
        )

    def calculate_consumption(self, savings_rate):  # Validated
        """
        This method calculates the consumption.
        """
        # Reshape savings rate from 2D to 3D
        savings_rate = savings_rate[:, :, np.newaxis]

        investment = savings_rate * self.output
        consumption = self.output - investment

        return consumption

    def calculate_consumption_per_timestep(self, savings_rate, timestep):  #
        """
        This method calculates the consumption per timestep.
        """
        # Reshape savings rate from 1D to 2D
        savings_rate = savings_rate[:, np.newaxis]

        investment = savings_rate * self.output[:, timestep, :]
        consumption_per_timestep = self.output[:, timestep, :] - investment

        return consumption_per_timestep

    def get_consumption_per_capita_per_timestep(self, scenario, savings_rate, timestep):
        scenario = get_economic_scenario(scenario)
        consumption_per_timestep = self.calculate_consumption_per_timestep(
            savings_rate, timestep
        )
        consumption_per_capita_per_timestep = (
            1e3 * consumption_per_timestep / self.population[:, timestep, :, scenario]
        )

        return consumption_per_capita_per_timestep

    def calculate_social_cost_of_carbon(
        self, fossil_and_land_use_emissions, savings_rate, regional=True
    ):
        """
        This method calculates the social cost of carbon.
        """
        # TODO: Calculations are currently not correct. Need to fix it.

        # total_emissions = emissions + land_use_emissions
        # print("Total Emissions", fossil_and_land_use_emissions.shape)

        consumption = self.calculate_consumption(savings_rate)
        # print("consumption", consumption.shape)

        # Calculate the social cost of carbon
        #  scc[t, n] = (-1000 * eq_E[t, n]) / eq_cc[t, n]

        emissions_marginal = np.diff(fossil_and_land_use_emissions, axis=1)
        consumption_marginal = np.diff(consumption, axis=1)

        # print("emissions_marginal", emissions_marginal[0, 0, 0])
        # print("consumption_marginal", consumption_marginal[0, 0, 0])

        social_cost_of_carbon = (emissions_marginal / consumption_marginal) * -1000

        return social_cost_of_carbon

    def get_net_output(self):
        return self.output

    def get_net_output_by_timestep(self, timestep):
        return self.output[:, timestep, :]

    def get_abatement(self):
        return self.abatement

    def get_damages(self):
        return self.damage

    def get_population(self, scenario):
        scenario = get_economic_scenario(scenario)
        return self.population[:, :, :, scenario]

    def get_consumption_per_capita(self, scenario, savings_rate):
        scenario = get_economic_scenario(scenario)
        consumption = self.calculate_consumption(savings_rate)
        consumption_per_capita = 1e3 * consumption / self.population[:, :, :, scenario]

        return consumption_per_capita

    def get_capital_stock(self, scenario, savings_rate):
        """
        This method returns the capital stock.
        """

        scenario = get_economic_scenario(scenario)
        # Reshape savings rate from 2D to 3D
        savings_rate = savings_rate[:, :, np.newaxis]

        investment = savings_rate * self.output
        capital_stock = np.zeros(
            (
                len(self.region_list),
                len(self.model_time_horizon),
                self.NUM_OF_ENSEMBLES,
            )
        )

        # Setting the initial capital stock
        capital_stock[:, 1, :] = self.capital_tfp[:, 0, :]
        # print(capital_stock[0, 1, 0])

        for t in range(2, len(self.model_time_horizon)):
            capital_stock[:, t, :] = (
                capital_stock[:, t - 1, :]
                * (
                    (1 - (self.depreciation_rate_capital / self.data_timestep))
                    ** self.data_timestep
                )
            ) + (investment[:, t - 1, :] * self.timestep)

        return capital_stock

    def get_interest_rate(
        self,
        scenario,
        savings_rate,
        pure_rate_of_social_time_preference,
        elasticity_of_marginal_utility_of_consumption,
    ):
        """
        This method returns the interest rate.
        """

        consumption_per_capita = self.get_consumption_per_capita(
            scenario=scenario, savings_rate=savings_rate
        )

        interest_rate = (
            (1 + pure_rate_of_social_time_preference)
            * (consumption_per_capita[:, 1:, :] / consumption_per_capita[:, :-1, :])
            ** (elasticity_of_marginal_utility_of_consumption / self.timestep)
        ) - 1

        return interest_rate

    def _interpolate_gdp(self):
        interp_data = np.zeros(
            (
                self.gdp_array.shape[0],
                len(self.model_time_horizon),
                self.gdp_array.shape[2],
            )
        )

        for i in range(self.gdp_array.shape[0]):
            for j in range(self.gdp_array.shape[2]):
                f = interp1d(
                    self.data_time_horizon, self.gdp_array[i, :, j], kind="linear"
                )
                interp_data[i, :, j] = f(self.model_time_horizon)

        self.gdp_array = interp_data

    def _interpolate_population(self):
        interp_data = np.zeros(
            (
                self.population_arr.shape[0],
                len(self.model_time_horizon),
                self.population_arr.shape[2],
            )
        )

        for i in range(self.population_arr.shape[0]):
            for j in range(self.population_arr.shape[2]):
                f = interp1d(
                    self.data_time_horizon, self.population_arr[i, :, j], kind="linear"
                )
                interp_data[i, :, j] = f(self.model_time_horizon)

        self.population_arr = interp_data

    def calculate_baseline_per_capita_growth(self):
        """
        This method calculates the baseline per capita growth.
        """
        # Calculate the baseline per capita growth
        self.baseline_per_capita_growth = self.gdp / self.population

        # Divide all t+1 timestep by preceding timestep
        self.baseline_per_capita_growth[:, 1:, :, :] /= self.baseline_per_capita_growth[
            :, :-1, :, :
        ]
        # Take the power of 1/timestep and subtract 1
        self.baseline_per_capita_growth **= 1 / self.timestep
        self.baseline_per_capita_growth -= 1

        # Set the first timestep to zero
        self.baseline_per_capita_growth[:, 0, :, :] = 0

    def __getattribute__(self, __name: str) -> Any:
        """
        This method returns the value of the attribute of the class.
        """
        return object.__getattribute__(self, __name)
