# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:17:02 2024

@author: apoujon
"""
import numpy as np
import scipy

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

    def __init__(self, twolevelsgame_model, id, N, timestep):
        """
        policy_model the overarching policy class
        id a unique identifier for the region
        N the number of households for the opinion dynamics model of the region
        """

        self.twolevelsgame_model = twolevelsgame_model
        self.id = id

        # Generating the N households (ie, constituency. N = 100 by default from Policy() )
        self.n_households = N
        self.households = []
        # self.constituency = np.matrix([0,0,0]).T #An agreggation of the wills of the households regarding future evolution of current Policy

        for i in range(N):
            # Initialisation of different households and their perspectives. TODO: Should be region dependant in the future.
            self.households += [Household(self)]

        # ------ Local Opinions Dynamics Parameters ------
        # TODO APN: All OD processes relies on same OD params. Could be interesting to have a specific class for OD defined with proper conf for each different considerations.
        # Also, another possibility is to combine all OD approaches (both kind of worries in one go), this is made possible by enlarging the Laplacian matrix...
        self.opdyn_max_iter = 1
        # Only one iteration per step
        self.opdyn_influence = 0.001
        self.opdyn_learning = 0.01
        self.opdyn_agreement = 0.001
        self.opdyn_lambda_noise = 0
        self.opdyn_threshold_close = 0.5
        self.opdyn_threshold_far = 1
        self.opdyn_external_worry_decay = 0.7

        # Media and Extreme weather events worry
        self.media_climate_worry = 0
        self.extreme_weather_events_worry = 0
        self.media_abatement_worry = 0
        self.economic_conjuncture_worry = 0

        # Negotiator, negotiation strategy depends on constituency
        self.negotiator = Negotiator(self)
        self.update_state_policy_from_constituency(timestep)

    def aggregate_households_opinions(self):
        array_utility = np.array(
            [hh.assess_policy() for hh in self.households]
        )
        array_support = array_utility > 0
        # Is there more support or opposition?
        if np.count_nonzero(array_support) / self.n_households > 0.5:
            # More support, average positive support
            return int(np.mean(array_utility[array_support]))
        else:
            # More opposition, average negative support
            return int(np.mean(array_utility[np.logical_not(array_support)]))
        return

    def update_regional_opinion(self):
        # Update on observations (uses FaIR-Perspectives ==> Understanding)
        self.update_from_information()

        # Update on opinions (==> Social learning)
        self.update_from_social_network()

    def update_from_information(self):
        for hh in self.households:
            hh.update_climate_distrib_beliefs(self.twolevelsgame_model.justice_model.rng)
        return

    def update_from_social_network(self):
        self.spreading_climate_worries(self.twolevelsgame_model.justice_model.rng)  # TODO APN: Distinguish monetary vs non-monetary aspects
        self.spreading_abatement_worries(self.twolevelsgame_model.justice_model.rng)
        return

    def update_state_policy_from_constituency(self, timestep):
        self.negotiator.shifting_policy(timestep)
        return

    def spreading_climate_worries(self, rng):
        n = len(self.households)
        I = np.eye(n)
        k = 0

        v1 = np.ones((n, 1))

        vect_internal_worry = np.matrix(
            [[hh.internal_climate_worry] for hh in self.households]
        )
        # Agents' worries related to climate change
        vect_external_worry = np.matrix(
            [[hh.external_climate_worry] for hh in self.households]
        )
        # Media/Extreme weather events generated worry related to climate change
        vect_aggregated_worry = np.clip(
            vect_internal_worry + vect_external_worry, -1, 1
        )
        # Resulting worries

        dispersion = np.max(
            np.abs(vect_aggregated_worry @ v1.T - vect_aggregated_worry @ v1.T)
        )
        while (dispersion > self.opdyn_agreement) & (k < self.opdyn_max_iter):
            k += 1
            # Create the network
            L = (
                np.abs(
                    vect_aggregated_worry @ v1.T - (vect_aggregated_worry @ v1)
                    < self.opdyn_threshold_close
                )
                - I
            )
            Lclose = np.diag(np.sum(L, 1)) - L

            L = (
                np.abs(
                    vect_aggregated_worry @ v1.T - (vect_aggregated_worry @ v1)
                    > self.opdyn_threshold_far
                )
                - I
            )
            Lfar = np.diag(np.sum(L, 1)) - L

            # L = generateAdjacencyMatrix(n,'random', 1-lbd);
            # Lalea = np.diag(np.sum(L,1))- L;

            L = Lclose - Lfar  # + Lalea;

            # Update "internal" worry
            vect_internal_worry = (
                I - self.opdyn_influence * L
            ) @ vect_aggregated_worry - vect_external_worry

            # Update "external" worry
            vect_external_worry = np.clip(
                vect_external_worry * self.opdyn_external_worry_decay
                + rng.random() * 0.1,
                -1,
                1,
            )

            vect_aggregated_worry = np.clip(
                vect_internal_worry + vect_external_worry, -1, 1
            )
            dispersion = np.max(
                np.abs(vect_aggregated_worry @ v1.T - vect_aggregated_worry @ v1.T)
            )

    def spreading_abatement_worries(self, rng):
        n = len(self.households)
        I = np.eye(n)
        k = 0

        v1 = np.ones((n, 1))

        vect_internal_worry = np.matrix(
            [[hh.internal_abatement_worry] for hh in self.households]
        )
        # Agents' worries related to climate change
        vect_external_worry = np.matrix(
            [[hh.external_abatement_worry] for hh in self.households]
        )
        # Media/Extreme weather events generated worry related to climate change
        vect_aggregated_worry = np.clip(
            vect_internal_worry + vect_external_worry, -1, 1
        )
        # Resulting worries

        dispersion = np.max(
            np.abs(vect_aggregated_worry @ v1.T - vect_aggregated_worry @ v1.T)
        )
        while (dispersion > self.opdyn_agreement) & (k < self.opdyn_max_iter):
            k += 1
            # Create the network
            L = (
                np.abs(
                    vect_aggregated_worry @ v1.T - (vect_aggregated_worry @ v1)
                    < self.opdyn_threshold_close
                )
                - I
            )
            Lclose = np.diag(np.sum(L, 1)) - L

            L = (
                np.abs(
                    vect_aggregated_worry @ v1.T - (vect_aggregated_worry @ v1)
                    > self.opdyn_threshold_far
                )
                - I
            )
            Lfar = np.diag(np.sum(L, 1)) - L

            # L = generateAdjacencyMatrix(n,'random', 1-lbd);
            # Lalea = np.diag(np.sum(L,1))- L;

            L = Lclose - Lfar  # + Lalea;

            # Update "internal" worry
            vect_internal_worry = (
                I - self.opdyn_influence * L
            ) @ vect_aggregated_worry - vect_external_worry

            # Update "external" worry
            vect_external_worry = np.clip(
                vect_external_worry * self.opdyn_external_worry_decay
                + rng.random() * 0.1,
                -1,
                1,
            )

            vect_aggregated_worry = np.clip(
                vect_internal_worry + vect_external_worry, -1, 1
            )
            dispersion = np.max(
                np.abs(vect_aggregated_worry @ v1.T - vect_aggregated_worry @ v1.T)
            )

        return

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
