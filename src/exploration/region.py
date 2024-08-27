# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:17:02 2024

@author: apoujon
"""
import numpy as np
import scipy

from src.exploration.DataLoaderTwoLevelGame import XML_init_values
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
                    dict_regions_climate_happening]
        """

        self.twolevelsgame_model = twolevelsgame_model
        self.id = id
        self.code = code
        dict_regions_distribution_income = list_dicts[0]
        self.distribution_income = dict_regions_distribution_income[self.code] / 100
        self.distribution_cost_mitigation = []
        self.distribution_cost_damages = []
        # print("=======V " + self.code + " V========")
        # print(self.distribution_income)
        # print(np.sum(self.distribution_income))

        # Generating the N households (ie, constituency. N = 100 by default from Policy() )
        self.n_households = XML_init_values.Region_n_households
        self.households = []

        for i in range(self.n_households):
            # Initialisation of different households and their perspectives.
            quintile = int(i / (0.2 * self.n_households))
            self.households += [
                Household(self, self.distribution_income.index[quintile], i, list_dicts)
            ]

        # ------ Opinions Dynamics Parameters For Temperature Threshold -----
        self.opdyn_max_iter = XML_init_values.Region_opdyn_max_iter
        # Only one iteration per step
        self.opdyn_influence = XML_init_values.Region_opdyn_influence
        self.opdyn_learning = XML_init_values.Region_opdyn_learning
        self.opdyn_agreement = XML_init_values.Region_opdyn_agreement
        self.opdyn_lambda_noise = XML_init_values.Region_opdyn_lambda_noise
        self.opdyn_threshold_close = XML_init_values.Region_opdyn_threshold_close
        self.opdyn_threshold_far = XML_init_values.Region_opdyn_threshold_far
        self.opdyn_external_worry_decay = (
            XML_init_values.Region_opdyn_external_worry_decay
        )

        ###############################
        ###### Emotions-Opinions ######
        ###############################
        self.opinion_climate_change = Opinion(
            self.n_households, [hh.emotion_climate_change for hh in self.households],
        )
        self.opinion_economy = Opinion(
            self.n_households, [hh.emotion_economy for hh in self.households]
        )

        # Negotiator, negotiation strategy depends on constituency
        self.negotiator = Negotiator(self)
        self.update_state_policy_from_constituency(timestep)

    def aggregate_households_opinions(self, timestep):

        # 25 = 5 steps per year time 5 years for each mandate (Ypolicy)
        for hh in self.households:
            hh.update_emotion_opinion()

        for rounds in range(25):
            self.opinion_climate_change.step()
            self.opinion_economy.step()

            self.twolevelsgame_model.log_files.f_household[1].writerow(
                [timestep - 0.2 * (24 - rounds), self.id]
                + [
                    hh.threshold_expected_temperature_elevation
                    for hh in self.households
                ]
                + [hh.emotion_climate_change.b0 for hh in self.households]
                + [hh.emotion_climate_change.v for hh in self.households]
                + [hh.emotion_climate_change.o for hh in self.households]
                + [hh.emotion_economy.b0 for hh in self.households]
                + [hh.emotion_economy.v for hh in self.households]
                + [hh.emotion_economy.o for hh in self.households],
            )

        # hh.emotion_climate_change.o "Are we mitigating enough?": if yes (>0), then we want less (or equal) mitigation hence a negative coefficient
        # hh.emotion_economy.o "Am I willing to pay for mitigation?": if yes (>0) then we can keep the same level or more mitigation, hence a positive coefficient
        array_utility = np.array(
            [
                -hh.emotion_climate_change.o + hh.emotion_economy.o
                for hh in self.households
            ]
        )
        array_support = array_utility > 0
        array_opposition = array_utility < 0

        share_support = np.count_nonzero(array_support) / self.n_households
        share_opposition = np.count_nonzero(array_opposition) / self.n_households
        share_neutral = 1 - (share_support + share_opposition)

        return [share_opposition, share_neutral, share_support]

    def update_regional_opinion(self, timestep):
        # Update on observations (uses FaIR-Perspectives ==> Understanding)
        self.update_from_information(timestep)

        # Update on opinions (==> Social learning)
        self.update_from_social_network()

    def update_from_information(self, timestep):
        for hh in self.households:
            hh.update_climate_distrib_beliefs(
                self.twolevelsgame_model.justice_model.rng, timestep
            )
        return

    def update_from_social_network(self):
        # calling the rng: self.twolevelsgame_model.justice_model.rng
        # TODO Update the content of this function
        self.spreading_climate_threshold()

        # self.update_climate_distrib_beliefs_from_social()
        # self.spreading_climate_worries()  # TODO APN: Distinguish monetary vs non-monetary aspects
        # self.spreading_abatement_worries(self.twolevelsgame_model.justice_model.rng)
        return

    def update_state_policy_from_constituency(self, timestep):
        self.negotiator.shifting_policy(timestep)
        return

    """
    This function is used to update the households distribution of expected future temperature elevation on interactions between the households.
    I have commented it, because it is a bit complicated process and most likely not that necessary. Instead, it seems more interesting to have different weights 
    on the expected dmgs due to climate change
    
    def update_climate_distrib_beliefs_from_social(self):
        
        #Update the means of the beliefs upon future local temperatures for agents.
        
        n = len(self.households)
        I = np.eye(n) != np.eye(n)
        k = 0

        v1 = np.ones((1, 1, 10))

        mat_mean_temperature = np.zeros(
            [self.n_households, Household.N_CLIMATE_BELIEFS]
        )
        # A table with all the beliefs distributions about temperature elevation from all households
        beliefs_table = np.zeros(
            [
                self.n_households,
                Household.N_CLIMATE_BELIEFS,
                int(len(Household.DISTRIB_X_AXIS)),
            ]
        )
        i = 0
        for hh in self.households:
            beliefs_table[i, :, :] = hh.climate_old_distrib_beliefs
            # Get expected mean temperatures for each households
            mat_mean_temperature[i, :] = np.array(
                [
                    Household.mean_distribution(hh.climate_old_distrib_beliefs[i])
                    for i in range(Household.N_CLIMATE_BELIEFS)
                ]
            )
            i += 1

        # Create a network structure for closest individuals expectations (they are 0.5C difference at most)
        tensor_mean_temperature = np.kron(mat_mean_temperature, v1.T)
        for i in range(Household.N_CLIMATE_BELIEFS):
            L = (
                abs(
                    tensor_mean_temperature[:, :, i]
                    - tensor_mean_temperature[:, :, i].T
                )
                < 2
            ) * I

            for ih in range(self.n_households):
                # Take beliefs distributions of all neighbors and sum them
                nb_neighbors = np.sum(L[ih, :])
                if nb_neighbors != 0:
                    mean_distrib_beliefs = np.sum(
                        beliefs_table[L[ih, :], i, :]
                    ) / np.sum(L[ih, :])

                    # Update current beliefs with summed beliefs from neighbors
                    self.households[ih].climate_distrib_beliefs = (
                        self.households[ih].climate_distrib_beliefs
                        * mean_distrib_beliefs
                    )

            # Ensure normalization and update the old distribution of beliefs
            for ih in range(self.n_households):
                norm_coeff = np.sum(self.households[ih].climate_distrib_beliefs, axis=1)
                self.households[ih].climate_distrib_beliefs = np.array(
                    [
                        self.households[ih].climate_distrib_beliefs[i, :]
                        / norm_coeff[i]
                        for i in range(Household.N_CLIMATE_BELIEFS)
                    ]
                )
                self.households[ih].climate_old_distrib_beliefs = self.households[
                    ih
                ].climate_distrib_beliefs.copy()
    """

    def spreading_climate_threshold(self):
        """
        Climate threshold is a field of a household. The value of this field represents the limit on the climate temperature elevation compared to preinductrial levels
        that the household thinks can be handled. This climate threshold has an effect on the policy support, as it translate into more or less strong worry level for the
        households.
        """
        n = len(self.households)
        I = np.eye(n)
        k = 0

        v1 = np.ones((n, 1))

        # Agents' thresholds related to temperature elevation
        vect_thresholds = np.array(
            [[hh.threshold_expected_temperature_elevation] for hh in self.households]
        )

        # Resulting dipersion
        dispersion = np.max(np.abs(vect_thresholds @ v1.T - v1 @ vect_thresholds.T))

        while (dispersion > self.opdyn_agreement) & (k < self.opdyn_max_iter):
            k += 1
            # Create the network
            L = (
                np.abs(vect_thresholds @ v1.T - (v1 @ vect_thresholds.T))
                < self.opdyn_threshold_close
            ) - I

            Lclose = np.diag(np.sum(L, 1)) - L

            L = (
                np.abs(vect_thresholds @ v1.T - (v1 @ vect_thresholds.T))
                > self.opdyn_threshold_far
            ) - I

            Lfar = np.diag(np.sum(L, 1)) - L

            # L = generateAdjacencyMatrix(n,'random', 1-lbd);
            # Lalea = np.diag(np.sum(L,1))- L;

            L = Lclose - Lfar  # + Lalea;

            # Update thresholds
            # TODO: opdyn_influence could be computed to change depending on how close/far the values are from each other
            vect_thresholds = np.clip(
                (I - self.opdyn_influence * L) @ vect_thresholds, 0, 8
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

        # End of can comment

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


