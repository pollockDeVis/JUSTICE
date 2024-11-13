# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:17:02 2024

@author: apoujon
"""
import numpy as np
import scipy

from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.LogFiles import print_log, LogFiles
from src.exploration.Opinions import Opinion
from src.exploration.household import Household
from src.exploration.negotiator import Negotiator


""""
Definition of transition functions for emission rate
"""


def exponential_first_order(X, startY, endY, transY):
    c = (transY - startY) - (endY - startY) / 2.0
    # print(startY, transY, endY, c)
    return (np.exp(c * (X - startY) / endY) - 1.0) / (
        np.exp(c * (endY - startY) / endY) - 1.0
    )


def linear(X, startR, startY, endR, endY):
    a = (endR - startR) / (endY - startY)
    return np.clip((a * X + startR - a * startY), 0, 1)


class Region:
    """
    Defines a region constituted by its constituencies, a negotiator and a policy configuration.

    """

    def __init__(
        self,
        rng,
        twolevelsgame_model,
        id,
        code,
        timestep,
        list_dicts,
    ):
        """
        policy_model the overarching policy class
        id a unique identifier for the region
        N the number of households for the opinion dynamics model of the region
        list_dicts = [dict_regions_distribution_income,
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
        self.rng = rng
        self.twolevelsgame_model = twolevelsgame_model
        self.id = id
        self.code = code
        dict_regions_distribution_income = list_dicts[0]
        self.distribution_income = dict_regions_distribution_income[self.code] / 100
        self.distribution_cost_mitigation = []
        self.distribution_cost_damages = []
        self.disaggregated_predmg_consumption = []
        self.disaggregated_post_costs_consumption = []
        self.perceived_avg_income = []
        # print("=======V " + self.code + " V========")
        # print(self.distribution_income)
        # print(np.sum(self.distribution_income))

        # Generating the N households (ie, constituency. N = 100 by default from Policy() )
        self.n_households = XML_init_values.dict["Region_n_households"]
        self.households = []

        self.model_HK = XML_init_values.dict["model_HK"]
        self.HK_influence_close = XML_init_values.dict["HK_influence_close"]
        self.HK_epsilon_dmg = XML_init_values.dict["HK_epsilon_dmg"]
        self.HK_epsilon_support = XML_init_values.dict["HK_epsilon_support"]

        # ------ Opinions Dynamics Parameters For Temperature Threshold -----
        self.opdyn_max_iter = XML_init_values.dict["Region_opdyn_max_iter"]
        # Only one iteration per step
        self.opdyn_influence_close = XML_init_values.dict[
            "Region_opdyn_influence_close"
        ]
        self.opdyn_influence_far = XML_init_values.dict["Region_opdyn_influence_far"]
        self.opdyn_agreement = XML_init_values.dict["Region_opdyn_agreement"]
        self.opdyn_threshold_close = XML_init_values.dict[
            "Region_opdyn_threshold_close"
        ]
        self.opdyn_threshold_far = XML_init_values.dict["Region_opdyn_threshold_far"]

        self.network = np.zeros((self.n_households, self.n_households))
        for i in range(self.n_households):
            # Initialisation of different households and their perspectives.
            quintile = int(i / (0.2 * self.n_households))
            self.households += [
                Household(
                    self.rng,
                    self,
                    self.distribution_income.index[quintile],
                    i,
                    quintile,
                    list_dicts,
                )
            ]
            self.network[i, :] = [
                rng.random()
                > (1 / 2 + 0.49 * (quintile != int(r / (0.2 * self.n_households))))
                for r in range(self.n_households)
            ]

        ###############################
        ###### Emotions-Opinions ######
        ###############################
        self.opinion_climate_change = Opinion(
            self.n_households,
            [hh.emotion_climate_change for hh in self.households],
        )
        self.opinion_economy = Opinion(
            self.n_households, [hh.emotion_economy for hh in self.households]
        )

        self.regional_literacy = list_dicts[10][self.code]["At least once a week"] / 100

        self.relative_wealth = 0
        self.gini = 0
        self.initial_responsibility = (
            list_dicts[8][self.code]["Regardless of what other countries do"] / 100
        )

        self.alpha1 = XML_init_values.dict["Region_alpha1"]
        self.alpha2 = XML_init_values.dict["Region_alpha2"]
        self.beta1 = XML_init_values.dict["Region_beta1"]
        self.beta2 = XML_init_values.dict["Region_beta2"]
        self.gamma = XML_init_values.dict["Region_gamma"]

        # Negotiator, negotiation strategy depends on constituency
        self.negotiator = Negotiator(self)
        self.update_state_policy_from_constituency(timestep)

    def aggregate_households_opinions(self, timestep):

        # hh.emotion_climate_change.o "Are we mitigating enough?": if yes (>0), then we want less (or equal) mitigation hence a negative coefficient
        # hh.emotion_economy.o "Am I willing to pay for mitigation?": if yes (>0) then we can keep the same level or more mitigation, hence a positive coefficient
        """array_utility = np.array(
            [
                hh.emotion_climate_change.o + hh.emotion_economy.o
                for hh in self.households
            ]
        )"""
        array_utility = np.array([hh.policy_support - 1 / 2 for hh in self.households])
        array_utility = np.clip(array_utility, a_min=-1, a_max=1)

        print_log.f_HK_opinion_dynamics[1].writerow(
            [timestep, self.id]
            + [hh.policy_support for hh in self.households]
            + [hh.damage_beliefs[1][0] for hh in self.households]
        )

        array_support = array_utility > 0.05
        array_opposition = array_utility < -0.05
        # Dividing by two because each opinion comprised between -1 and 1 so difference is between -2 and 2
        mean_utility = np.mean(array_utility) / 2

        share_support = np.count_nonzero(array_support) / self.n_households
        temp = array_utility[array_support]
        if len(temp) == 0:
            strength_support = 0
        else:
            strength_support = np.mean(temp)

        share_opposition = np.count_nonzero(array_opposition) / self.n_households
        temp = array_utility[array_opposition]
        if len(temp) == 0:
            strength_opposition = 0
        else:
            strength_opposition = np.mean(temp)
        share_neutral = 1 - (share_support + share_opposition)

        print_log.f_share_opinions[1].writerow(
            [
                timestep,
                self.id,
                share_opposition,
                share_neutral,
                share_support,
                strength_opposition,
                mean_utility,
                strength_support,
            ]
        )

        return [
            share_opposition,
            share_neutral,
            share_support,
            strength_opposition,
            mean_utility,
            strength_support,
        ]

    def update_regional_opinion(self, timestep):
        # Update on observations (uses FaIR-Perspectives ==> Understanding)
        self.update_from_information(timestep)
        self.update_from_interactions(timestep)

    def update_from_interactions(self, timestep):
        print_log.write_log(
            LogFiles.MASKLOG_region,
            "Region.py",
            "update_from_interactions",
            "region " + self.code + " (" + str(self.id) + ")",
        )
        # 5 steps per year (Ypolicy)
        if timestep % self.twolevelsgame_model.Y_policy == 0:
            self.HK2D(timestep)
            for hh in self.households:
                if self.model_HK==1:
                    hh.internal_HK_mode1()
                elif self.model_HK==3:
                    hh.internal_HK_mode3()

    def update_from_information(self, timestep):
        for hh in self.households:
            hh.update_damage_distrib_beliefs(timestep)
            # hh.update_climate_distrib_beliefs(timestep)
        return

    def update_state_policy_from_constituency(self, timestep):
        self.negotiator.shifting_policy(timestep)
        return

    def HK2D(self, timestep):
        n = len(self.households)
        I = np.eye(n + 1)
        k = 0

        v1 = np.ones((n + 1, 1))

        # Agents' mean expected economic damages
        vect_expected_damages = np.array(
            [[hh.damage_beliefs[1][0]] for hh in self.households]
        )
        vect_p_consider_information = np.array(
            [[hh.p_consider_information] for hh in self.households]
        )

        # Agent's support for current policy
        vect_support = np.array([[hh.policy_support] for hh in self.households])

        # Adding the information agent
        # Damages coming from INFORMATION model
        year = Household.BELIEF_YEAR_OFFSET[-1]
        i_max_temp = np.argmax(
            self.twolevelsgame_model.justice_model.information_model.local_economic_damage_information[
                0
            ][
                self.id
            ][
                timestep:year
            ]
        )
        information = self.twolevelsgame_model.justice_model.information_model.local_economic_damage_information[
            0
        ][
            self.id
        ][
            i_max_temp
        ]
        vect_expected_damages = np.concatenate(
            (vect_expected_damages, [[information]]), axis=0
        )
        vect_support = np.concatenate((vect_support, [[information]]), axis=0)
        # Filling to match size
        vect_p_consider_information = np.concatenate(
            (vect_p_consider_information, [[0]]), axis=0
        )

        # Resulting dispersion on damages
        dispersion_damages = np.max(
            np.abs(vect_expected_damages @ v1.T - v1 @ vect_expected_damages.T)
        )
        # Resulting dispersion on support
        dispersion_support = np.max(np.abs(vect_support @ v1.T - v1 @ vect_support.T))

        epsilon_damages = self.HK_epsilon_dmg
        epsilon_support = self.HK_epsilon_support
        agreement_threshold = 0.0000001
        max_iter = 1
        while (dispersion_support + dispersion_damages > agreement_threshold) & (
            k < max_iter
        ):
            k += 1
            # Create the network
            # TODO: The networks can be constructed based on the distance for 1 of the dimensions, of for both of these dimensions.
            L_dmg = (
                np.abs(vect_expected_damages @ v1.T - (v1 @ vect_expected_damages.T))
                <= epsilon_damages
            ) - I

            # The coefficient in front of self.n_households represents the trust in scientific information
            # The parameter vect_p_consider_information represents the probability to be exposed to the scientific information
            L_dmg[:, -1:] = 0 * (
                1
                / 2
                * self.n_households
                * (
                    vect_p_consider_information
                    > self.rng.random((self.n_households + 1, 1))
                )
            )

            # For policy support, we put everything to 0: there is no influence of scientific information (in such a direct way)
            L_support = (
                np.abs(vect_support @ v1.T - (v1 @ vect_support.T)) <= epsilon_support
            ) - I
            L_support[:, -1:] = 0 * L_support[:, -1:]

            L_dmgclose = np.diag(np.sum(L_dmg, 1)) - L_dmg
            L_supportclose = np.diag(np.sum(L_support, 1)) - L_support

            # Forcing information agent to be close to all, according to information access
            # ==> Adding it at the end column of each row

            L_dmg = (
                np.abs(vect_expected_damages @ v1.T - (v1 @ vect_expected_damages.T))
                > epsilon_damages
            ) - I
            L_support = (
                np.abs(vect_support @ v1.T - (v1 @ vect_support.T)) > epsilon_support
            ) - I

            L_dmgfar = np.diag(np.sum(L_dmg, 1)) - L_dmg
            L_supportfar = np.diag(np.sum(L_support, 1)) - L_support

            # L = generateAdjacencyMatrix(n,'random', 1-lbd);
            # Lalea_info = np.diag(np.sum(L,1))- L;

            # Update thresholds
            influence_close = 0.01
            influence_far = 0
            vect_expected_damages = np.clip(
                (
                    (I - self.HK_influence_close * L_dmgclose + influence_far * L_dmgfar)
                    @ vect_expected_damages
                ),
                0,
                100,
            )
            vect_support = np.clip(
                (
                    (
                        I
                        - self.HK_influence_close * L_supportclose
                        + influence_far * L_supportfar
                    )
                    @ vect_support
                ),
                0,
                100,
            )

            dispersion_damages = np.max(
                np.abs(vect_expected_damages @ v1.T - v1 @ vect_expected_damages.T)
            )
            dispersion_support = np.max(
                np.abs(vect_support @ v1.T - v1 @ vect_support.T)
            )

        if k != 0:
            i = 0
            N_damage = np.sum(L_dmgclose, 1)
            N_support = np.sum(L_supportclose, 1)
            for hh in self.households:
                hh.damage_beliefs[1][0] = vect_expected_damages[i][0]
                hh.policy_support = vect_support[i][0]
                hh.neighbours_damage = N_damage[i]
                hh.neighbours_support = N_support[i]
                i = i + 1

    def spreading_climate_threshold(self):
        """
        Climate threshold is a field of a household. The value of this field represents the limit on the climate temperature elevation compared to preinductrial levels
        that the household thinks can be handled. This climate threshold has an effect on the policy support, as it translate into more or less strong worry level for the
        households.
        """
        n = len(self.households) + 1
        I = np.eye(n)
        k = 0

        v1 = np.ones((n, 1))

        # Agents' thresholds related to temperature elevation
        vect_thresholds = np.array(
            [[hh.threshold_expected_temperature_elevation] for hh in self.households]
        )

        vect_p_consider_information = np.array(
            [[hh.p_consider_information] for hh in self.households]
        )

        # Adding the information agent
        # 1.5C threshold
        vect_thresholds = np.concatenate((vect_thresholds, [[3]]), axis=0)
        # Filling to match size
        vect_p_consider_information = np.concatenate(
            (vect_p_consider_information, [[0]]), axis=0
        )

        # Resulting dipersion
        dispersion = np.max(np.abs(vect_thresholds @ v1.T - v1 @ vect_thresholds.T))

        while (dispersion > self.opdyn_agreement) & (k < self.opdyn_max_iter):
            k += 1
            # Create the network
            L = (
                np.abs(vect_thresholds @ v1.T - (v1 @ vect_thresholds.T))
                <= self.opdyn_threshold_close
            ) - I

            # The coefficient in front of self.n_households represents the trust in scientific information
            # The parameter vect_p_consider_information represents the probability to be exposed to the scientific information
            L[:, -1:] = (
                1
                / 2
                * self.n_households
                * (
                    vect_p_consider_information
                    > self.rng.random((self.n_households + 1, 1))
                )
            )

            Lclose = np.diag(np.sum(L, 1)) - L

            # Forcing information agent to be close to all, according to information access
            # ==> Adding it at the end column of each row

            L = (
                np.abs(vect_thresholds @ v1.T - (v1 @ vect_thresholds.T))
                > self.opdyn_threshold_far
            ) - I

            Lfar = np.diag(np.sum(L, 1)) - L

            # L = generateAdjacencyMatrix(n,'random', 1-lbd);
            # Lalea_info = np.diag(np.sum(L,1))- L;

            L = Lclose - Lfar  # + Lalea;

            # Update thresholds
            vect_thresholds = np.clip(
                (
                    (
                        I
                        - self.opdyn_influence_close * Lclose
                        + self.opdyn_influence_far * Lfar
                    )
                    @ vect_thresholds
                ),
                0,
                8,
            )

            dispersion = np.max(np.abs(vect_thresholds @ v1.T - v1 @ vect_thresholds.T))

        i = 0
        for hh in self.households:
            hh.threshold_expected_temperature_elevation = vect_thresholds[i][0]
            i = i + 1

    def emission_control_rate(self):
        """PARAMETRIZED EMISSION POLICY
        #TODO Linspace seems to work in case of timestep = 1. Perhaps not working for other values...
        nb_pts = (self.policy_model.justice_model.time_horizon.end_year - self.policy_model.justice_model.time_horizon.start_year)//self.policy_model.justice_model.time_horizon.timestep +1;
        X = np.linspace(self.policy_model.justice_model.time_horizon.start_year,self.policy_model.justice_model.time_horizon.end_year, nb_pts);
        np.clip(X,self.policy[0],self.policy[2]) #TODO this clip() function might not be necessary
        ecr_projection=exponential_first_order(X, self.policy[0], self.policy[2], self.policy[1]);
        #ecr_projection=linear(X, 0.2, self.policy[0], 1, self.policy[1]);
        """

        """ PIECEWISE LINEAR POLICY """
        start_year = self.twolevelsgame_model.justice_model.time_horizon.start_year
        pol_at_start_year = 0
        # TODO APN change policy at start year. 1) It should be defined somewhere else (maybe as an attribute of the negotiator) 2) it should be changeable at the creation of the abm-justice model
        end_year = self.twolevelsgame_model.justice_model.time_horizon.end_year
        timestep_size = 1
        # TODO APN ge the timestep size from the model (it is not necessarily always 1 -  could be 5 years)
        policy = self.negotiator.policy

        last_p_year = start_year
        last_pol = pol_at_start_year

        f = scipy.interpolate.interp1d(
            np.insert(np.append(policy[0], end_year), 0, start_year),
            np.insert(np.append(policy[1], policy[1, -1]), 0, pol_at_start_year),
            kind="linear",
        )

        ecr_projection = f(
            np.linspace(start_year, end_year, int(end_year - start_year + 1))
        )

        return np.tile(
            np.matrix(np.clip(ecr_projection, 0, 1)).T,
            self.twolevelsgame_model.justice_model.no_of_ensembles,
        )

    def compute_gini(self):
        cumulative_quintile = np.cumsum(
            self.disaggregated_post_costs_consumption
            / np.sum(self.disaggregated_post_costs_consumption)
        )
        cumulative_quintile = np.insert(cumulative_quintile, 0, 0)
        area = sum((cumulative_quintile[:-1] + cumulative_quintile[1:]) * 0.2 / 2)
        self.gini = 1 - 2 * area
