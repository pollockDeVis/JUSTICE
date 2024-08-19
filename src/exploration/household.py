# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:20:58 2024

@author: apoujon
"""
import numpy as np

from src.exploration.DataLoaderTwoLevelGame import XML_init_values


class Household():
    DISTRIB_RESOLUTION = XML_init_values.Household_DISTRIB_RESOLUTION
    # Resolution of distribution in CELSIUS
    DISTRIB_MIN_VALUE = XML_init_values.Household_DISTRIB_MIN_VALUE
    # Minimum temperature deviation relative to preindustrial levels in CELSIUS
    DISTRIB_MAX_VALUE = XML_init_values.Household_DISTRIB_MAX_VALUE
    # Maximum temperature deviation relative to preindustrial levels in CELSIUS
    DISTRIB_X_AXIS = np.arange(DISTRIB_MIN_VALUE, DISTRIB_MAX_VALUE, DISTRIB_RESOLUTION)
    # Support for the distribution
    BELIEF_YEAR_OFFSET = np.array([-1, XML_init_values.Household_BELIEF_YEAR_OFFSET])
    # The years offset (compared to 2015) for which each agent has to have a specific belief of temperature elevation
    # The first belief is always about the current temperature elevation (it is moving)
    N_CLIMATE_BELIEFS = len(BELIEF_YEAR_OFFSET)
    # Number of beliefs
    DEFAULT_INFORMATION_STD = XML_init_values.Household_DEFAULT_INFORMATION_STD
    P_INFORMATION = XML_init_values.Household_P_INFORMATION
    # Probability to assimilate information at a given timestep
    GAMMA = XML_init_values.Household_GAMMA
    # Imperfect memory parameter

    # global_temperature_information = []; get from the information module
    # local_temperature_information = []; get from the information module
    # global_distrib_flsi = [[] for i in range(N_CLIMATE_BELIEFS)]; get from the information module
    # regional_distrib_flsi = [[[]] for r in range(56)] get from the information module

    def __init__(self, region, quintile, id):
        self.model_region = region
        self.id = id

        # Initialising climate intuition
        self.climate_init_mean_beliefs = np.array(
            [
                np.random.random()
                * XML_init_values.Household_climate_init_mean_beliefs_var
                + XML_init_values.Household_climate_init_mean_beliefs_floor,
                np.random.random()
                * XML_init_values.Household_climate_init_mean_beliefs_var
                + XML_init_values.Household_climate_init_mean_beliefs_floor,
            ]
        )
        # Temperature expected for BELIEF_YEAR_OFFSET years
        self.climate_init_var_beliefs = np.array(
            [
                XML_init_values.Household_climate_init_var_beliefs_current,
                XML_init_values.Household_climate_init_var_beliefs_offset1,
            ]
        )
        # Confidence in the expectation

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

        ###################################################
        ################## Sensitivities ##################
        ###################################################
        # between 0 and 1
        self.sensitivity_to_threshold_information = 0.2
        # > Between 0 and 2
        self.sensitivity_to_costs = 1
        # > Between 0 and 8
        self.threshold_expected_temperature_elevation = np.random.random() * 8
        # >Between -1 and 1
        self.elasticity_expected_mitigation_distribution = 0
        self.elasticity_expected_loss_damages_distribution = 0
        # Coefficient for climate damage evaluation (see RICE2013R: psi_2)
        self.psi_2 = 2.67 * 10**-3

        # Wages and consumption
        self.quintile = quintile

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

    def update_climate_distrib_beliefs(self, rng, timestep):
        """
        Updating the climate beliefs distributions for an agent based on available information

        Returns
        -------
        None.

        """
        p = rng.random()

        if p < self.P_INFORMATION:
            # Updating from information module: Expectation about temperature elevation
            self.climate_distrib_beliefs = (
                self.climate_distrib_beliefs * self.filtered_climate_information()
            )
            # Objective temperature threshold
            self.threshold_expected_temperature_elevation = (
                (1 - self.sensitivity_to_threshold_information)
                * self.threshold_expected_temperature_elevation
                + self.sensitivity_to_threshold_information * 1.5
            )
        # TODO ADP: In case of no information (ELSE case) we can have a memory effect, or biases taking place
        # But for now, I let it all empty

        norm_coeff = np.sum(self.climate_distrib_beliefs, axis=1)
        # print(norm_coeff)

        try:
            self.climate_distrib_beliefs = np.array(
                [
                    self.climate_distrib_beliefs[i, :] / norm_coeff[i]
                    for i in range(self.N_CLIMATE_BELIEFS)
                ]
            )
        except Warning:
            print(norm_coeff)

        self.model_region.twolevelsgame_model.f_household_beliefs[1].writerow(
            [timestep, self.model_region.id, self.id]
            + self.climate_distrib_beliefs[-1].tolist()
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

        # Expected temperature elevation (worry)
        temp = Household.mean_distribution(self.climate_distrib_beliefs[-1])
        expected_temperature_elevation = 1 + (
            temp - self.threshold_expected_temperature_elevation
        )

        # Experienced climate change with shifting baeline OR extreme weather events
        # TODO: find database of extreme weather events

        # FIXME: Put real values down below
        # Experienced climate damages
        loss_and_damages = self.model_region.distribution_cost_damages[self.quintile]
        # Experienced mitigation costs
        mitigation_costs = self.model_region.distribution_cost_mitigation[self.quintile]
        # Experienced Economic Context
        if mitigation_costs > loss_and_damages:
            experienced_economic_context = -1  # opposition
        else:
            experienced_economic_context = 1  # support

        # FIXME: Put real values down below
        # Expected climate damages
        expected_loss_and_damages = 0  # self.expected_climate_damages #+*-/...
        # Expected mitigation costs
        expected_mitigation_costs = 0
        # Expected Economic Context
        if expected_mitigation_costs > expected_loss_and_damages:
            expected_economic_context = -1  # opposition
        else:
            expected_economic_context = 1  # support

        # FIXME: delete next line when real values are use for expectation of costs
        expected_economic_context = 0

        U = (
            self.sensitivity_to_costs
            * (expected_economic_context + experienced_economic_context)
            + expected_temperature_elevation
            - self.model_region.negotiator.policy[1, 0]
        )

        self.model_region.twolevelsgame_model.f_household_assessment[1].writerow(
            [
                timestep,
                self.model_region.id,
                self.id,
                self.sensitivity_to_costs,
                loss_and_damages,
                mitigation_costs,
                experienced_economic_context,
                expected_loss_and_damages,
                expected_mitigation_costs,
                expected_economic_context,
                expected_temperature_elevation,
                self.model_region.negotiator.policy[1, 0],
                U,
            ]
        )

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
