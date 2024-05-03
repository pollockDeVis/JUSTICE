# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:20:58 2024

@author: apoujon
"""
import numpy as np


class Household:
    DISTRIB_RESOLUTION = 0.01
    # Resolution of distribution in CELSIUS
    DISTRIB_MIN_VALUE = -2
    # Minimum temperature deviation relative to preindustrial levels in CELSIUS
    DISTRIB_MAX_VALUE = +8
    # Maximum temperature deviation relative to preindustrial levels in CELSIUS
    DISTRIB_X_AXIS = np.arange(DISTRIB_MIN_VALUE, DISTRIB_MAX_VALUE, DISTRIB_RESOLUTION)
    # Support for the distribution
    BELIEF_YEAR_OFFSET = np.array([0, 10, 50, 99])
    # The years in the future for which each agent has to have a specific belief of temperature elevation
    N_CLIMATE_BELIEFS = len(BELIEF_YEAR_OFFSET)
    # Number of beliefs
    DEFAULT_INFORMATION_STD = 0.01
    P_INFORMATION = 0.5
    # Probability to assimilate information at a given timestep
    GAMMA = 0.3
    # Imperfect memory parameter

    # global_temperature_information = []; get from the information module
    # local_temperature_information = []; get from the information module
    # global_distrib_flsi = [[] for i in range(N_CLIMATE_BELIEFS)]; get from the information module
    # regional_distrib_flsi = [[[]] for r in range(56)] get from the information module

    def __init__(self, region, utility_params):
        self.model_region = region

        # Initialising climate intuition
        self.climate_init_mean_beliefs = np.array(
            [
                np.random.random() * Household.DISTRIB_MAX_VALUE,
                np.random.random() * Household.DISTRIB_MAX_VALUE,
                np.random.random() * Household.DISTRIB_MAX_VALUE,
                np.random.random() * Household.DISTRIB_MAX_VALUE,
            ]
        )
        # Temperature expected for BELIEF_YEAR_OFFSET years
        self.climate_init_var_beliefs = np.array([0.01, 0.01, 0.01, 0.01])
        # Confidance in the expectation

        distrib_beliefs = np.array(
            [
                Household.gaussian_distrib(
                    g_mean=self.climate_init_mean_beliefs[i],
                    g_std=self.climate_init_var_beliefs[i],
                )
                for i in range(self.N_CLIMATE_BELIEFS)
            ]
        )
        # Here range(nb_beliefs) depending on the number of beliefs to be modelled
        norm_coeff = np.sum(distrib_beliefs, axis=1)
        self.climate_distrib_beliefs = np.array(
            [
                distrib_beliefs[i, :] / norm_coeff[i]
                for i in range(self.N_CLIMATE_BELIEFS)
            ]
        )

        # used in the opinion update
        self.climate_old_distrib_beliefs = np.array(
            [
                distrib_beliefs[i, :] / norm_coeff[i]
                for i in range(self.N_CLIMATE_BELIEFS)
            ]
        )

        # Initialising Climate worry
        self.internal_climate_worry = 0
        # Worry coming from interaction with other individuals
        self.external_climate_worry = 0
        # Worry coming from extreme weather events and media
        self.climate_worry = 0
        # Aggregation of above worries

        # Initialising Abatement Cost worry
        self.internal_abatement_worry = 0
        # Worry coming from interaction with other individuals
        self.external_abatement_worry = 0
        # Worry coming from economy and media
        self.abatement_worry = 0
        # Aggregation of above worries

        # Initialising perceived willingness to Contribute [See "representative evidence on the actual and perceived support for climate action", 2023 Peter Andre]
        self.perceived_WTC = 0

        # Utility function

        # TODO APN: The following is completely wrong and ad hoc, but it's for the purpose of testing the shifting_policy function (having negative utility values)
        if self.model_region.id % 2 == 0:
            self.utility_parameters = [-10, -10, +10, +10]
            # - expected temperature elevation and CC dmgs (future) - experienced temperature elevation and Xtrm weather events (present)
            # + expected economic cost of abatement (future) + experienced economic context (present)
        else:
            self.utility_parameters = [-1, -10, +10, -10]

        # Coefficient for climate damage evaluation (see RICE2013R: psi_2)
        self.psi_2 = 2.67 * 10**-3

        self.utility_parameters = utility_params

    def expected_climate_damages(self):
        temp_profile = Household.belief_to_projection(
            self.climate_distrib_beliefs, self.BELIEF_YEAR_OFFSET
        )
        return np.sum(self.psi_2 * temp_profile**2)

    def experienced_weather_events(self):
        return 0

    def expected_abatement_costs(self):
        return 0

    def experienced_economic_context(self):
        return 0

    def filtered_climate_information(self):
        """
        Get the global_distrib_flsi and regional_distrib_flsi and define a personalized distribution of the information for the agent.
        Possibilities are:
            -> global_distrib_flsi
            -> regional_distrib_flsi
            -> f * global_distrib_flsi + (1-f) * regional_distrib_flsi with f between 0 and 1
            -> global_distrib_flsi * self belief fold
            -> global_distrib_flsi * group belief fold

        (See: Pawel Sobkowicz, 2018, Opinion Dynamics Model Based on Cognitive Biases of Complex Agents)
        """
        return (
            self.model_region.twolevelsgame_model.justice_model.information_model.global_distrib_flsi
        )

    def update_climate_distrib_beliefs(self, rng):
        """
        Updating the climate beliefs distributions for an agent based on available information

        Returns
        -------
        None.

        """
        p = rng.random()
        if p < self.P_INFORMATION:
            # Updating with information
            self.climate_distrib_beliefs = (
                self.climate_distrib_beliefs * self.filtered_climate_information()
            )
        else:
            # TODO ADP: Let the belief unchanged (because parametrization hard)
            # Imperfect memory
            mean_beliefs = Household.mean_distribution(self.climate_distrib_beliefs)
            distrib_beliefs_save = self.climate_distrib_beliefs
            # Compute distribution based on initial std for agent
            self.climate_distrib_beliefs = np.array(
                [
                    Household.gaussian_distrib(
                        g_mean=mean_beliefs[i], g_std=self.climate_init_var_beliefs[i]
                    )
                    for i in range(self.N_CLIMATE_BELIEFS)
                ]
            )
            norm_coeff = np.sum(self.climate_distrib_beliefs, axis=1)
            self.climate_distrib_beliefs = np.array(
                [
                    self.climate_distrib_beliefs[i, :] / norm_coeff[i]
                    for i in range(self.N_CLIMATE_BELIEFS)
                ]
            )
            # Merging of learned belief and belief with initial std
            self.climate_distrib_beliefs = (
                self.GAMMA * distrib_beliefs_save
                + (1 - self.GAMMA) * self.climate_distrib_beliefs
            )

        norm_coeff = np.sum(self.climate_distrib_beliefs, axis=1)
        self.climate_distrib_beliefs = np.array(
            [
                self.climate_distrib_beliefs[i, :] / norm_coeff[i]
                for i in range(self.N_CLIMATE_BELIEFS)
            ]
        )

    def assess_policy(self, timestep):
        """
        Compute the Utility equivalent of the current policy. If U is positive, then the agent is pushing for more stringent emission reductions;
        Else, the agent if pushing for less stringent policy.

        Returns
        -------
        U : TYPE
            DESCRIPTION.

        """
        # TODO APN: self.utility_parameters[0] = -1\coeff * np.log(GDP/Capita)   [See Peter Andre 2023, "representative evidence on the actual and perceived support for climate action"]

        # expected_temperature_evaluation = +1 -1 0
        temp = Household.mean_distribution(self.climate_distrib_beliefs[-1])
        if temp > 2:
            expected_temperature_evaluation = 1  # Support
        else:
            expected_temperature_evaluation = 0  # Neutral

        # experienced_economic_context = +1 -1 +0
        regional_consumption_per_capita = (
            self.model_region.twolevelsgame_model.justice_model.consumption_per_capita[
                self.model_region.id
            ]
        )
        global_mean_consumption_per_capita = np.mean(
            self.model_region.twolevelsgame_model.justice_model.consumption_per_capita
        )

        experienced_economic_context = 0  # Neutral
        if regional_consumption_per_capita > 1.2 * global_mean_consumption_per_capita:
            experienced_economic_context = 1  # Support
        elif (
            regional_consumption_per_capita < 0.99 * global_mean_consumption_per_capita
        ):
            experienced_economic_context = -1  # Opposition

        (self.model_region.twolevelsgame_model.f_household)[1].writerow(
            [self.model_region.id, timestep]
            + [expected_temperature_evaluation]
            + [experienced_economic_context]
        )
        U = (
            expected_temperature_evaluation
            + experienced_economic_context
            - self.model_region.negotiator.policy[1, 0]
        )

        # + expected temperature elevation and CC dmgs (future) + experienced temperature elevation and Xtrm weather events (present)
        # - expected economic cost of abatement (future) - experienced economic context (present)
        """
        baseline_expected_climate_damages = self.expected_climate_damages()
        support_expected_climate_damages = self.expected_climate_damages()
        opposition_expected_climate_damages = self.expected_climate_damages()

        baseline_expected_abatement_costs = self.expected_abatement_costs()
        support_expected_abatement_costs = self.expected_abatement_costs()
        opposition_expected_abatement_costs = self.expected_abatement_costs()

        experienced_climate_change_and_weather_events = self.experienced_weather_events()
        experienced_economic_context = self.experienced_economic_context()

        
        U = self.utility_parameters @ np.array(
            [
                self.expected_climate_damages(),
                self.experienced_weather_events(),
                self.expected_abatement_costs(),
                self.experienced_economic_context()
            ]
        )
        """

        return U

    ###########################################################################
    ###                                 UTILS
    ###########################################################################
    def gaussian_distrib(
        g_mean=0,
        g_std=1,
        min_val=DISTRIB_MIN_VALUE,
        max_val=DISTRIB_MAX_VALUE,
        step=DISTRIB_RESOLUTION,
    ):
        """
        Parameters
        ----------
        g_mean : TYPE, optional
            DESCRIPTION. The default is 0.
        g_std : TYPE, optional
            DESCRIPTION. The default is 1.
        min_val : TYPE, optional
            DESCRIPTION. The default is Household.DISTRIB_MIN_VALUE.
        max_val : TYPE, optional
            DESCRIPTION. The default is Household.DISTRIB_MAX_VALUE.
        step : TYPE, optional
            DESCRIPTION. The default is Household.DISTRIB_RESOLUTION.

        Returns
        -------
        A vector of the gaussian distribution N(g_mean, g_std) over the range [min_val; max_val] with resolution "step"

        """

        # TODO APN It is not normal if we have to enter th following... I have seen some weird values for the local temperatures (18C???)
        if g_mean > Household.DISTRIB_MAX_VALUE:
            g_mean = Household.DISTRIB_MAX_VALUE

        possible_values = np.arange(min_val, max_val, step)
        return np.exp(
            np.array(
                [-1.0 * (x - g_mean) ** 2 / (2.0 * (g_std**2)) for x in possible_values]
            )
        )

    def mean_distribution(
        distribution,
        min_val=DISTRIB_MIN_VALUE,
        max_val=DISTRIB_MAX_VALUE,
        step=DISTRIB_RESOLUTION,
    ):
        """
        Parameters
        ----------
        distribution : TYPE
            DESCRIPTION. A matrix which rows are distributions over the range [min_val; max_val] with resolution "step"
        min_val : TYPE, optional
            DESCRIPTION. The default is Household.DISTRIB_MIN_VALUE.
        max_val : TYPE, optional
            DESCRIPTION. The default is Household.DISTRIB_MAX_VALUE.
        step : TYPE, optional
            DESCRIPTION. The default is Household.DISTRIB_RESOLUTION.

        Returns
        -------
        A scalar representing the mean of the distribution for each rows of "distribution"

        """
        return distribution @ np.arange(min_val, max_val, step)

    def belief_to_projection(distrib_beliefs, belief_year_offset):
        """
        Computes a piecewise-linear profile of temperature elevations in the future based on the average values of the beliefs for a set of future years

        Parameters
        ----------
        distrib_beliefs : TYPE
            DESCRIPTION. A matrix which rows are distributions for a belief.
        belief_year_offset : TYPE
            DESCRIPTION. A vector which elements correspond a given year to a belief in distrib_beliefs.

        Returns
        -------
        temperature_projection : TYPE
            DESCRIPTION.

        """
        temperature_projection = np.array([])
        year = belief_year_offset[0]
        temp0 = Household.mean_distribution(distrib_beliefs[0])
        for i in range(len(distrib_beliefs)):
            temp = Household.mean_distribution(distrib_beliefs[i])
            step = (temp - temp0) / (belief_year_offset[i] - year + 1)
            if step != 0:
                temperature_projection = np.append(
                    temperature_projection, np.arange(temp0, temp, step)
                )
            year = belief_year_offset[i] + 1
            temp0 = temp

        return temperature_projection

    def belief_to_projection_uncertainty(distrib_beliefs, belief_year_offset):
        """
        Computes a profile of temperature elevations in the future based on the distribution of beliefs provided for a set of future years

        Parameters
        ----------
        distrib_beliefs : TYPE
            DESCRIPTION. A matrix which rows are distributions for a belief.
        belief_year_offset : TYPE
            DESCRIPTION. A vector which elements correspond a given year to a belief in distrib_beliefs.

        Returns
        -------
        temperature_projection : TYPE
            DESCRIPTION.

        """
        temperature_projection = np.array([])
        year = belief_year_offset[0]
        temp0 = Household.mean_distribution(distrib_beliefs[0])
        for i in range(len(distrib_beliefs)):
            temp = Household.mean_distribution(distrib_beliefs[i])
            step = (temp - temp0) / (belief_year_offset[i] - year + 1)
            lin_proj = []
            if step != 0:
                lin_proj = np.arange(temp0, temp, step)
                projection = (
                    lin_proj
                    - temp
                    + np.random.choice(
                        np.arange(-2, 8, 0.01), len(lin_proj), p=distrib_beliefs[i]
                    )
                )

                temperature_projection = np.append(temperature_projection, projection)
            year = belief_year_offset[i] + 1
            temp0 = temp

        return temperature_projection
