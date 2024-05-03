"""
This is the negotiations module that determines the emission rates for next run step(s)
#'CO2 emissions in [GtCO2/year]'
"""

import os
from typing import Any
import numpy as np
from scipy.interpolate import interp1d
import copy
from src.exploration.region import Region
import csv
from datetime import datetime


class TwoLevelsGame:
    """
    This class defines a model for a regional policy using a two-level game
    approach, taking into consideration both the local constituencies and the international
    negotiations.
    """

    def __init__(
        self,
        justice_model,
        number_regions=57,
        population_size_by_region=10,
        timestep=0,
        utility_params=[0, 0, 0, 0, 0],
    ):
        """
        justice_model is a reference to the overarching justice model
        number_of_region refers to the number of regions in the model
        population_size_by_region refers to the number of households in each opinion dynamics model
        """

        # Saving data (TODO APN: Create a class for saving methods and structures)
        # -> Create folder for current simulation
        path = "data/output/" + datetime.now().strftime("SAVE_%Y_%m_%d_%H%M") + "/"
        os.makedirs(path, exist_ok=True)

        f1 = open(path + "regions.csv", "w", newline="")
        self.f_region = (
            f1,
            csv.writer(f1, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )
        f2 = open(path + "policy.csv", "w", newline="")
        self.f_policy = (
            f2,
            csv.writer(f2, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )
        # self.f_policy[1].writerow(['Region ID', 'Timestep', 'Policy Size', 'Policy Shift', 'Policy Values', 'Policy Years'])
        f3 = open(path + "negotiator.csv", "w", newline="")
        self.f_negotiator = (
            f3,
            csv.writer(f3, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )
        f4 = open(path + "information.csv", "w", newline="")
        self.f_information = (
            f4,
            csv.writer(f4, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )
        f5 = open(path + "household.csv", "w", newline="")
        self.f_household = (
            f5,
            csv.writer(f5, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )
        f6 = open(path + "emissions.csv", "w", newline="")
        self.f_emissions = (
            f6,
            csv.writer(f6, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )

        self.f_parameters = open(path + "parameters.txt", "w", newline="")
        self.f_parameters.write("Utility Parameters : " + str(utility_params))

        # Initialise Agents (ie. regions)
        self.justice_model = justice_model
        self.N = number_regions
        self.regions = [
            Region(self, i, population_size_by_region, timestep, utility_params)
            for i in range(self.N)
        ]

        # Coalitions (ad hoc, now a cycle graph)
        """ Coalitions could appear in the context of international agreement. They would result from a change in the 
        structure of the network graph linking all regions together. By default the graph is a connected graph of
        valency 2 (eg. a cycle graph).
        This is not used yet. """
        edges = np.array([[i, (i + 1) % self.N] for i in range(self.N)])
        self.coalitions = np.zeros((edges.max() + 1, edges.max() + 1))
        self.coalitions[edges[:, 0], edges[:, 1]] = 1
        self.coalitions[edges[:, 1], edges[:, 0]] = 1

        # Parameters for international negotiations
        self.Y_nego = 5 # How many years between a new set of international negotiation rounds
        self.Y_policy = 5 # How many years between a new set of regional policy update

    def step(self, timestep):
        """
        Defines a step for the Policy() class.
        """
        # 1 - Updating the Opinions of the constituencies for all regions
        self.update_regions(timestep)

        # 2 - Negotiating an international agreement (every Y_nego years)
        if timestep % self.Y_nego == 0:
            self.international_negotiations(timestep)

        # 3 - Deduce an emission control for all region depending on constituency, international agreement, and inertia
        self.update_emission_control_rate(timestep)

    def update_regions(self, timestep):
        """
        Updates opinions of constituencies for all regions
            - Taking into account information from climate models
            - Taking into account opinion dynamics
            - Assessing the current policy
        """

        # TODO uncomment to use the projections of alternative ecr scenarios over shortime
        # self.justice_model.information_model.generate_projections(timestep, self.regions)

        # Size is 57*1
        self.justice_model.consumption_per_capita = np.mean(
            self.justice_model.economy.get_consumption_per_capita_per_timestep(
                self.justice_model.savings_rate[:, timestep],
                timestep
            ),
            axis=1,
        )

        for a in self.regions:
            a.update_regional_opinion()
            if timestep % self.Y_policy == 0:
                a.update_state_policy_from_constituency(timestep)

    def international_negotiations(self, timestep):
        """
        This function allows negotiators to make pledges of emissions cuts for the future years.
        Each negotiator have to come up with a target year associated to a cutting rate goal.
        Pledges are assumed to be linear by other negotiators in their implementation.
        """

        # TODO APN: This is very simplified way of seeing international negotiations. It might also not work properly when the delta is negative and large (not respecting maximum changing rate of emission control rate)
        if False:
            avg_global_net_zero_year = 0
            for region in self.regions:
                policy = region.negotiator.policy
                avg_global_net_zero_year += policy[0, -1]
            avg_global_net_zero_year = avg_global_net_zero_year / len(self.regions)

            for region in self.regions:
                delta_regional_pressure_ecr = (
                    region.negotiator.regional_pressure_later_ecr
                    - region.negotiator.regional_pressure_earlier_ecr
                )
                policy = region.negotiator.policy
                max_policy_rate = region.negotiator.max_cutting_rate_gradient
                current_time = timestep + self.justice_model.time_horizon.start_year

                region.negotiator.policy[0, -1] = (
                    policy[0, -1] + delta_regional_pressure_ecr
                )
        return
        # For now nothing happens

        # """
        # International negotiations as a simple (DeGroot) Opinion Dynamics model.
        # """
        # max_iter = 10;
        # d_min = 1e-1;
        # eps = 0.1;
        # I = np.eye(self.N);
        # iter = 0;
        # #For each region, draw an international proposal
        # proposal = np.zeros((self.N, max_iter));
        # proposal[:,0] = np.array([p.negotiator.proposal() for p in self.regions]);

        # #Adjacent matrix (ie. coalitions) to Laplacian
        # L = np.diag(np.sum(self.coalitions, 0))-self.coalitions;
        # d = np.max(np.abs(proposal[:,iter] @ np.ones((self.N, self.N)) - proposal[:,iter]))

        # while (d > d_min) and (iter < max_iter-1):
        #     iter += 1;
        #     proposal[:,iter] = (I - eps*L)@proposal[:,iter-1];
        #     d = np.max(np.abs(proposal[:,iter] @ np.ones((self.N, self.N)) - proposal[:,iter]))

        # #Update target 0 emissions year in negotiators and at the local policy level
        # for i in range(self.N):
        #     self.regions[i].negotiator.zero_emissions_target_year = proposal[i,iter];
        #     self.regions[i].policy[2] = proposal[i,iter];

        return

    def update_emission_control_rate(self, timestep):
        """
        Updates the emission control rate for each regions for future timesteps based on the current regional policies.
        """
        # TODO OPTIM See if we really need to compute all the curve or only next step
        for a in self.regions:
            ecr = a.emission_control_rate()
            self.justice_model.emission_control_rate[a.id, timestep:, :] = ecr[
                timestep:, :
            ]

            self.f_emissions[1].writerow(
                [a.id]
                + [em for em in self.justice_model.emission_control_rate[a.id, :, 0]]
            )  # ECR are all the same for all ensemble, hence we onyl register the one for ensemble 0
        self.justice_model.data["emission_cutting_rate"][:, timestep, :] = (
            self.justice_model.emission_control_rate[:, timestep, :]
        )

    def close_files(self):
        self.f_policy[0].close()
        self.f_region[0].close()
        self.f_household[0].close()
        self.f_negotiator[0].close()
        self.f_information[0].close()
        self.f_parameters.close()
        return
