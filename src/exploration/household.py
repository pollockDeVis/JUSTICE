# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:20:58 2024

@author: apoujon
"""
import numpy as np

from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.Emotions import Emotion_opinion


class Household:
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

    def __init__(self, rng, region, quintile, id, list_dicts):
        """
        region::class Region::a reference to the parent of the household
        quintile::int::income quintile of the household
        id::int::identifier of the household
        list_dicts::list dict::[dict_regions_distribution_income,
                                dict_regions_climate_worry,
                                dict_regions_economic_impact,
                                dict_regions_climate_awareness,
                                dict_regions_threat_20_years,
                                dict_regions_harm_future_gen,
                                dict_regions_gov_priority,
                                dict_regions_most_responsible,
                                dict_regions_country_responsibility,
                                dict_regions_climate_happening,
                                dict_regions_freq_hear]

        """
        self.model_region = region
        self.id = id
        self.rng = rng

        ### Assimilating/Considering information (regional) ###
        # freq_hear: Never, At most once a year, Several times a year, At least once a month, At least once a week, Don't know, Refused
        # Here I associate Never, Don't know and Refused with a probability of considering information = 0
        self.p_consider_information = (
            1
            / 100
            * (
                0.3 * list_dicts[10][region.code]['Once a year or less often']
                + 0.5 * list_dicts[10][region.code]['Several times a year']
                + 0.7 * list_dicts[10][region.code]['At least once a month']
                + 0.9 * list_dicts[10][region.code]['At least once a week']
            )
        )
        #self.opdyn_threshold_sensitivity = self.p_consider_information * self.model_region.opdyn_influence_close + self.model_region.opdyn_influence_far

        ### Initialising climate intuition ###
        # -> Temperature increase now, and in 2200
        choices = np.array(
            [
                1.3,
                0,
                0.9,
                0.9,
            ]
        )  # Climate change is happening: "Yes, No, Don't know, refused"
        mean_now = rng.choice(choices, p=list_dicts[9][region.code][:-1] / 100)

        means_2200 = [
            4,
            3,
            2,
            1,
            4,
            4,
        ]
        variance = [
            0.2,
            0.3,
            0.4,
            0.2,
            2,
            2,
        ]
        # harm future generation: "A great deal, A moderate amount, Only a little, Not at all, Don't know, refused"
        choice = rng.choice([0, 1, 2, 3, 4, 5], p=list_dicts[5][region.code][:-1] / 100)
        mean_2200 = means_2200[choice]
        variance = variance[choice]

        self.climate_init_mean_beliefs = np.array(
            [
                mean_now + variance * (np.random.random() - 0.5) * 2,
                mean_2200 + variance * (np.random.random() - 0.5) * 2,
            ]
        )
        # -> Confidence in the expectation
        self.climate_init_var_beliefs = np.array(
            [
                XML_init_values.Household_climate_init_var_beliefs_current,
                variance,
            ]
        )

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
        # > temperature Threshold: Between 0C and 8C (compared to baseline)
        choices = [
            1.5,
            2,
            3,
            2,
            2,
        ]  # Threat at 20y: "Very Serious, Somewhat Serious, Not a threat, Don't know, Refused"
        choice = rng.choice(choices, p=list_dicts[4][region.code][:-1] / 100)
        self.threshold_expected_temperature_elevation = choice + (
            choice == 2
        ) * rng.beta(a=2, b=5)

        # Coefficient for climate damage evaluation (see RICE2013R: psi_2)
        self.psi_2 = 2.67 * 10**-3

        # Wages and consumption
        self.quintile = quintile

        ######################
        ###### EMOTIONS ######
        ######################
        # Opinion on: "Are we mitigating enough?"
        choices = [
            -0.5,
            -0.2,
            0.2,
            0.5,
            0,
        ]  # "Very worried, Somewhat worried, Not very worried, Not at all woried, Refused"
        choice = rng.choice(choices, p=list_dicts[1][region.code][:-1] / 100)
        valence = choice + rng.normal(0, 0.01, 1)[0]
        opinion = choice + rng.normal(0, 0.01, 1)[0] #0
        self.emotion_climate_change = Emotion_opinion(valence, opinion)
        # Opinion on: "Am I willing to pay for mitigation?"
        choices = [
            0.5,
            -0.5,
            0,
            0,
        ]  # "Taking action will improve economy, will damage economy, will not have any effect, Refused"
        choice = rng.choice(choices, p=list_dicts[2][region.code][:-1] / 100)
        valence = choice + rng.normal(0, 0.1, 1)[0]
        opinion = choice + rng.normal(0, 0.1, 1)[0] #0
        self.emotion_economy = Emotion_opinion(valence, opinion)

    def expected_climate_damages(self):
        temp_profile = Household.belief_to_projection(self.BELIEF_YEAR_OFFSET)
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

    def update_climate_distrib_beliefs(self, timestep):
        """
        Updating the climate beliefs distributions for an agent based on available information

        Returns
        -------
        None.

        """
        p = self.rng.random()

        if p < self.p_consider_information:
            # Updating from information module: Expectation about temperature elevation
            self.climate_distrib_beliefs = (
                self.climate_distrib_beliefs * self.filtered_climate_information()
            )


            """#Using a global treshold:
            self.threshold_expected_temperature_elevation = (
                (1 - self.sensitivity_to_threshold_information)
                * self.threshold_expected_temperature_elevation
                + self.sensitivity_to_threshold_information * 1.5
            )"""
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

        # VVV This save file is extremely heavy, so I'm commenting it out for now
        """self.model_region.twolevelsgame_model.log_files.f_household_beliefs[1].writerow(
            [timestep, self.model_region.id, self.id]
            + self.climate_distrib_beliefs[-1].tolist()
        )"""

    def update_emotion_opinion(self):
        """
        Compute the Utility equivalent of the current policy. If U is positive, then the agent is pushing for more stringent emission reductions;
        Else, the agent if pushing for less stringent policy.

        Returns
        -------
        U : TYPE
            DESCRIPTION.

        """

        #######################################
        ### EXTERNAL INFLUENCES ON EMOTIONS ###
        #######################################
        # >>> INFLUENCES ON SATISFACTION RELATIVE TO CLIMATE CHANGE, answering the question "Are we mitigating enough"
        # -> Expected Temperature Elevation
        expected_temp_elevation_2200 = Household.mean_distribution(
            self.climate_distrib_beliefs[-1]
        )
        if (
            expected_temp_elevation_2200
            >= self.threshold_expected_temperature_elevation
        ):
            # Disatisfied, no we are not mitigating enough
            self.emotion_climate_change.b0 = max(
                self.emotion_climate_change.b0 - 0.1, -0.5
            )

        elif (
            expected_temp_elevation_2200
            <= 0.5 * self.threshold_expected_temperature_elevation
        ):
            # Satisfied, yes we are mitigating enough
            self.emotion_climate_change.b0 = min(
                self.emotion_climate_change.b0 + 0.1, 0.5
            )

        # >>> INFLUENCES ON SATISFACTION RELATIVE TO THE ECONOMY, Do I wish to pay for mitigation?
        # -> Experienced climate damages and mitigation costs
        loss_and_damages = self.model_region.distribution_cost_damages[self.quintile]
        mitigation_costs = self.model_region.distribution_cost_mitigation[self.quintile]
        consumption = self.model_region.disaggregated_post_costs_consumption[self.quintile]
        if loss_and_damages/consumption > 0.1:
            # Damages from climate change exceed 10% of the final consumption in my quintile, I will pay
            self.emotion_economy.b0 = min(self.emotion_economy.b0 + 0.1, 0.5)

        else:
            # Loss and damages do not exceed 10% of my quintile consumption, I will NOT pay
            self.emotion_economy.b0 = max(self.emotion_economy.b0 - 0.1, -0.5)

        #######################################
        ### EXTERNAL INFLUENCES ON OPINIONS ###
        #######################################

    ###########################################################################
    ###                                 UTILS
    ###########################################################################
    @staticmethod
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
            # We get there in the case of the local temperature elevation distribution (but I don't why yet)
            # It does not matter (28/08/2024) as I do not use the local temperature information yet
            g_mean = Household.DISTRIB_MAX_VALUE

        possible_values = np.arange(min_val, max_val, step)
        return np.exp(
            np.array(
                [-1.0 * (x - g_mean) ** 2 / (2.0 * (g_std**2)) for x in possible_values]
            )
        )

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
