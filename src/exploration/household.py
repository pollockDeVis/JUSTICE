# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:20:58 2024

@author: apoujon
"""
import numpy as np

from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.Emotions import Emotion_opinion
from src.exploration.LogFiles import print_log, LogFiles


class Household:
    DISTRIB_RESOLUTION = XML_init_values.dict["Household_DISTRIB_RESOLUTION"]
    # Resolution of distribution in CELSIUS
    DISTRIB_MIN_VALUE = XML_init_values.dict["Household_DISTRIB_MIN_VALUE"]
    # Minimum temperature deviation relative to preindustrial levels in CELSIUS
    DISTRIB_MAX_VALUE = XML_init_values.dict["Household_DISTRIB_MAX_VALUE"]
    # Maximum temperature deviation relative to preindustrial levels in CELSIUS
    DISTRIB_X_AXIS = np.arange(DISTRIB_MIN_VALUE, DISTRIB_MAX_VALUE, DISTRIB_RESOLUTION)
    # Support for the distribution
    BELIEF_YEAR_OFFSET = np.array(
        [-1, XML_init_values.dict["Household_BELIEF_YEAR_OFFSET"]]
    )
    # The years offset (compared to 2015) for which each agent has to have a specific belief of temperature elevation
    # The first belief is always about the current temperature elevation (it is moving)
    N_CLIMATE_BELIEFS = len(BELIEF_YEAR_OFFSET)
    # Number of beliefs
    DEFAULT_INFORMATION_STD = XML_init_values.dict["Household_DEFAULT_INFORMATION_STD"]
    P_INFORMATION = XML_init_values.dict["Household_P_INFORMATION"]
    # Probability to assimilate information at a given timestep
    GAMMA = XML_init_values.dict["Household_GAMMA"]
    # Imperfect memory parameter

    # global_temperature_information = []; get from the information module
    # local_temperature_information = []; get from the information module
    # global_distrib_flsi = [[] for i in range(N_CLIMATE_BELIEFS)]; get from the information module
    # regional_distrib_flsi = [[[]] for r in range(56)] get from the information module

    def __init__(self, rng, region, quintile, id, quintile_number, list_dicts):
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
            1 / 100 * (list_dicts[10][region.code]["At least once a week"])
        )
        # self.opdyn_threshold_sensitivity = self.p_consider_information * self.model_region.opdyn_influence_close + self.model_region.opdyn_influence_far

        ### Initialising climate intuition ###
        # -> Temperature increase now, and in 2200
        means_now = np.array(
            [
                1.3,
                0,
                0.9,
                0.9,
            ]
        )  # Climate change is happening: "Yes, No, Don't know, refused"
        means_2200 = [
            2,
            0,
            rng.random() * 8,
            rng.random() * 8,
        ]  # Climate change is happening: "Yes, No, Don't know, refused"
        variances = [
            0.5,
            0.001,
            0.5,
            0.5,
        ]  # Climate change is happening: "Yes, No, Don't know, refused"
        choice_cc_happening = rng.choice(
            [0, 1, 2, 3], p=list_dicts[9][region.code][:-1] / 100
        )
        mean_now = means_now[choice_cc_happening]
        mean_2200 = means_2200[choice_cc_happening]
        variance = variances[choice_cc_happening]

        self.climate_init_mean_beliefs = np.array(
            [
                mean_now + variance * (rng.random() - 0.5) * 2,
                mean_2200 + variance * (rng.random() - 0.5) * 2,
            ]
        )
        # -> Confidence in the expectation
        self.climate_init_var_beliefs = np.array(
            [
                XML_init_values.dict["Household_climate_init_var_beliefs_current"],
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
        norm_coeff = np.sum(distrib_beliefs, axis=1)
        self.climate_distrib_beliefs = np.array(
            [
                distrib_beliefs[i, :] / norm_coeff[i]
                for i in range(self.N_CLIMATE_BELIEFS)
            ]
        )

        self.damage_init_mean_beliefs = np.array(
            [
                rng.random(),
                rng.random(),
            ]
        )
        # -> Confidence in the expectation
        self.damage_init_var_beliefs = np.array(
            [
                0.1,
                0.1,
            ]
        )
        self.damage_beliefs = np.array(
            [
                [self.damage_init_mean_beliefs[i], self.climate_init_var_beliefs[i]]
                for i in range(self.N_CLIMATE_BELIEFS)
            ]
        )

        self.policy_support = rng.random()

        ###################################################
        ################## Sensitivities ##################
        ###################################################
        # > temperature Threshold: Between 0C and 8C (compared to baseline)
        choices = [
            1,
            2,
            4,
            2.5,
            2.5,
        ]  # "Very worried, Somewhat worried, Not very worried, Not at all woried, Refused"
        choice = rng.choice(choices, p=list_dicts[1][region.code][:-1] / 100)
        # choice_cc_happening == 1 indicate an household not believing that climate change is real.
        # they have low temperature increase belief (T increase = 0) with high confidence (var = 0.01), but their threshold temperature is normal
        self.threshold_expected_temperature_elevation = (
            choice_cc_happening != 1
        ) * choice + (choice_cc_happening == 1) * 1.5

        # Coefficient for climate damage evaluation (see RICE2013R: psi_2)
        self.psi_2 = 2.67 * 10**-3

        # Wages and consumption
        self.quintile = quintile
        self.quintile_number = quintile_number

        ######################
        ###### EMOTIONS ######
        ######################
        # Opinion on: "Is your expected temperature increase worrying to you?"
        choices = XML_init_values.dict["sentiment_temperature_increase"]
        # "Very worried, Somewhat worried, Not very worried, Not at all woried, Refused"
        # choice_cc_happening == 1 indicate an household not believing that climate change is real.
        # so they are not worried
        choice = (choice_cc_happening != 1) * rng.choice(
            choices, p=list_dicts[1][region.code][:-1] / 100
        ) - (choice_cc_happening == 1) * 0.8
        valence = choice + rng.normal(0, 0.01, 1)[0]
        opinion = choice + rng.normal(0, 0.01, 1)[0]  # 0
        self.emotion_climate_change = Emotion_opinion(valence, opinion)
        # Opinion on: "Am I willing to pay for mitigation?"
        choices = XML_init_values.dict["sentiment_willingness_to_pay"]
        # "Taking action will improve economy, will damage economy, will not have any effect, Refused"
        # choice_cc_happening == 1 indicate an household not believing that climate change is real.
        choice = (choice_cc_happening != 1) * rng.choice(
            choices, p=list_dicts[2][region.code][:-1] / 100
        )
        valence = choice + rng.normal(0, 0.1, 1)[0]
        opinion = choice + rng.normal(0, 0.1, 1)[0]  # 0
        self.emotion_economy = Emotion_opinion(valence, opinion)

        ####################
        ###
        ####################
        self.expected_dmg_opinion = 0
        self.perceived_income_opinion = 0
        self.literacy_opinion = 0

        self.neighbours_damage = 0
        self.neighbours_support = 0
        self.conflict_coefficient = XML_init_values.dict["factor_conflict_coefficient"]
        self.weight_info_dmg_local = XML_init_values.dict["weight_info_dmg_local"]

    def update_expected_dmg_opinion(self):
        # Using DICE 2013R damage function and comparing to 10% GPD damages as 100% wanting more for mitigation policy
        self.expected_dmg_opinion = (
            0.00267 * Household.mean_distribution(self.climate_distrib_beliefs[-1]) ** 2
        )

    def update_perceived_inequalities_dmg(self):
        loss_and_damages = self.model_region.distribution_cost_damages[self.quintile]
        # mitigation_costs = self.model_region.distribution_cost_mitigation[self.quintile]
        consumption_predmg = self.model_region.disaggregated_predmg_consumption[
            self.quintile
        ]
        x = loss_and_damages / consumption_predmg * 10
        out = (1 - 8 / 9 * (x >= 0.1)) * (x * 10 - 1)
        self.expected_dmg_opinion = min(1, out)

    def update_literacy_opinion(self):
        self.literacy_opinion = (
            self.model_region.regional_literacy + 0.025 * self.quintile_number
        )

    def internal_HK_mode2(self):
        """
        Makes the sum of beliefs close to 1. This means the beliefs are being ranked (or is equivalent to the relative
        probability of adopting one of the belief). This is NOT FITTED to our issue in which both belief are though to
        be complementary rather than opposed.
        """
        c = 1 / 2 * abs(self.damage_beliefs[1][0] - self.policy_support)
        A = (1 - c) * self.damage_beliefs[1][0] + c * self.damage_beliefs[1][0] / (
            self.damage_beliefs[1][0] + self.policy_support
        )
        self.policy_support = (
            1 - c
        ) * self.policy_support + c * self.policy_support / (
            self.damage_beliefs[1][0] + self.policy_support
        )
        self.damage_beliefs[1][0] = A

    def update_perceived_income_opinion(self):
        self.perceived_income_opinion = (
            np.sign(self.model_region.perceived_avg_income[self.id])
            * (
                self.model_region.disaggregated_post_costs_consumption[self.quintile]
                - self.model_region.perceived_avg_income[self.id]
            )
            / self.model_region.perceived_avg_income[self.id]
        )

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

    def internal_HK_mode1(self):
        if (self.damage_beliefs[1][0]) != (self.policy_support):
            c = self.conflict_coefficient * abs(
                (self.damage_beliefs[1][0] - 1 / 2) - (self.policy_support - 1 / 2)
            )
            if abs(self.damage_beliefs[1][0] - 1 / 2) > abs(
                self.policy_support - 1 / 2
            ):
                self.policy_support = (
                    self.policy_support + np.sign(self.damage_beliefs[1][0] - 1 / 2) * c
                )
            elif abs(self.damage_beliefs[1][0] - 1 / 2) < abs(
                self.policy_support - 1 / 2
            ):
                self.damage_beliefs[1][0] = (
                    self.damage_beliefs[1][0] + np.sign(self.policy_support - 1 / 2) * c
                )
            else:
                if self.rng.random() > 0.5:
                    self.policy_support = (
                        self.policy_support
                        + np.sign(self.damage_beliefs[1][0] - 1 / 2) * c
                    )
                else:
                    self.damage_beliefs[1][0] = (
                        self.damage_beliefs[1][0]
                        + np.sign(self.policy_support - 1 / 2) * c
                    )

    def damage_information(self, timestep):
        year = Household.BELIEF_YEAR_OFFSET[-1]
        i_max_temp = np.argmax(
            self.model_region.twolevelsgame_model.justice_model.information_model.local_economic_damage_information[
                0
            ][
                self.model_region.id
            ][
                timestep:year
            ]
        )

        i_worst_temp = np.argmax(
            self.model_region.twolevelsgame_model.justice_model.information_model.maximum_damage_information[
                0
            ][
                timestep:year
            ]
        )
        # Multiplying the mean economic damages by 25 allows to bring 0.02 damages (ig. 2% of GDP) to 0.5 beliefs for damages (which is neutral)
        # This parameters is extremely important as it refers to the level of damages the households are considering okay versus not-okay.
        economic_dmg_normalized = (
            self.model_region.twolevelsgame_model.justice_model.information_model.local_economic_damage_information[
                0
            ][
                self.model_region.id
            ][
                i_max_temp
            ]
            * 0.5
            / XML_init_values.dict["loss_and_damages_neutral"]
        )
        economic_dmg_normalized = max(min(economic_dmg_normalized, 1), 0)
        economic_dmg_normalized_std = (
            self.model_region.twolevelsgame_model.justice_model.information_model.local_economic_damage_information[
                1
            ][
                self.model_region.id
            ][
                i_max_temp
            ]
            * 0.5
            / XML_init_values.dict["loss_and_damages_neutral"]
        )

        economic_worst_dmg_normalized = (
            self.model_region.twolevelsgame_model.justice_model.information_model.maximum_damage_information[
                0
            ][
                i_worst_temp
            ]
            * 0.5
            / XML_init_values.dict["loss_and_damages_neutral"]
        )
        economic_dmg_worst_global_normalized = max(
            min(economic_worst_dmg_normalized, 1), 0
        )

        return [
            economic_dmg_normalized,
            max(0.01, economic_dmg_normalized_std),
            economic_dmg_worst_global_normalized,
            max(0.01, economic_dmg_normalized_std),
        ]

    def internal_HK_mode3(self):
        """
        Influence according to the number of neighbours
        """
        if (self.damage_beliefs[1][0]) != (self.policy_support):
            c = self.conflict_coefficient * abs(
                (self.damage_beliefs[1][0] - 1 / 2) - (self.policy_support - 1 / 2)
            )

            if self.neighbours_damage > self.neighbours_support:
                self.policy_support = (
                    self.policy_support + np.sign(self.damage_beliefs[1][0] - 1 / 2) * c
                )
            elif self.neighbours_damage < self.neighbours_support:
                self.damage_beliefs[1][0] = (
                    self.damage_beliefs[1][0] + np.sign(self.policy_support) * c
                )
            else:
                if self.rng.random() > 0.5:
                    self.policy_support = (
                        self.policy_support
                        + np.sign(self.damage_beliefs[1][0] - 1 / 2) * c
                    )
                else:
                    self.damage_beliefs[1][0] = (
                        self.damage_beliefs[1][0] + np.sign(self.policy_support) * c
                    )

    def update_damage_distrib_beliefs(self, timestep):
        p = self.rng.random()
        if p < self.p_consider_information:
            # Updating from information module: Expectation about temperature elevation
            if self.model_region.id == 54:
                print_log.write_log(
                    print_log.MASKLOG_Household,
                    "household.py",
                    "update_damage_distrib_beliefs",
                    f"BEFORE update with information --> {self.damage_beliefs[1]}",
                )

            damage_information = self.damage_information(timestep)

            if self.model_region.id == 54:
                print_log.write_log(
                    print_log.MASKLOG_Household,
                    "household.py",
                    "update_damage_distrib_beliefs",
                    f"Information is --> {damage_information}",
                )

            new_mean = (damage_information[1]) ** 2 * self.damage_beliefs[1][
                0
            ] + self.damage_beliefs[1][1] ** 2 * (
                self.weight_info_dmg_local * damage_information[0]
                + (1 - self.weight_info_dmg_local) * damage_information[2]
            )
            new_std = max(
                0.1, damage_information[1] ** 2 * self.damage_beliefs[1][1] ** 2
            )
            self.damage_beliefs[1] = [new_mean, new_std] / (
                damage_information[1] ** 2 + self.damage_beliefs[1][1] ** 2
            )

            if self.model_region.id == 54:
                print_log.write_log(
                    print_log.MASKLOG_Household,
                    "household.py",
                    "update_damage_distrib_beliefs",
                    f"AFTER update with information --> {self.damage_beliefs[1]}",
                )

    def update_climate_distrib_beliefs(self, timestep):
        """
        Updating the climate beliefs distributions for an agent based on available information
        Gets called once every 5 year

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

            val_for_log = self.mean_distribution(
                self.filtered_climate_information(),
                min_val=self.DISTRIB_MIN_VALUE,
                max_val=self.DISTRIB_MAX_VALUE,
                step=self.DISTRIB_RESOLUTION,
            )
            print_log.write_log(
                LogFiles.MASKLOG_Information,
                "information.py",
                "construct_flsi",
                f"Mean temperature from information is (current year, and +185): {val_for_log[0]:0.2f}, {val_for_log[1]:0.2f} ",
            )

        norm_coeff = np.sum(self.climate_distrib_beliefs, axis=1)
        self.climate_distrib_beliefs = np.array(
            [
                self.climate_distrib_beliefs[i, :] / norm_coeff[i]
                for i in range(self.N_CLIMATE_BELIEFS)
            ]
        )

        # VVV This save file can be extremely heavy if too many regions are taken into account
        if self.model_region.id in [32, 54, 16, 56, 8]:
            print_log.f_household_beliefs[1].writerow(
                [timestep, self.model_region.id, self.id]
                + self.climate_distrib_beliefs[-1].tolist()
            )

    def update_temperature_threshold(self):
        # Gets called 5 time a year
        p = self.rng.random()
        if p < self.p_consider_information:
            self.threshold_expected_temperature_elevation = (
                (
                    1
                    - 1
                    / 5
                    * 1
                    / (
                        1
                        + np.exp(
                            abs(self.threshold_expected_temperature_elevation - 1.5)
                        )
                    )
                )
                * self.threshold_expected_temperature_elevation
            ) + 1 / 5 * 1 / (
                1 + np.exp(abs(self.threshold_expected_temperature_elevation - 1.5))
            ) * 1.5

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
