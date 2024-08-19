"""
This is the negotiations module that determines the emission rates for next run step(s)
#'CO2 emissions in [GtCO2/year]'
"""

import os
from typing import Any
import numpy as np
from scipy.interpolate import interp1d
import copy

from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.household import Household
from src.exploration.region import Region
import csv
import json
from datetime import datetime
import pandas as pd



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
        timestep=0,
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

        # average net consumption, loss and damage per quintile, abatement costs per quintile, post-damage concumption per quintile, average of post damage concumption per quintile
        f1 = open(path + "regions.csv", "w", newline="")
        self.f_region = (
            f1,
            csv.writer(f1, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )
        self.f_region[1].writerow(
            [
                "Timestep",
                "Region ID",
                "Region code",
                "First 20%",
                "Second 20%",
                "Third 20%",
                "Fourth 20%",
                "Fifth 20%",
                "JUSTICE net avg. cons.",
                "JUSTICE dmg.",
                "JUSTICE abt.",
                "First dmg.",
                "Second dmg.",
                "Third dmg.",
                "Fourth dmg.",
                "Fifth dmg.",
                "First abt.",
                "Second abt.",
                "Third abt.",
                "Fourth abt.",
                "Fifth abt.",
                "First cons. pre",
                "Second cons. pre",
                "Third cons. pre",
                "Fourth cons. pre",
                "Fifth cons. pre",
                "First cons. post",
                "Second cons. post",
                "Third cons. post",
                "Fourth cons. post",
                "Fifth cons. post",
                "net avg. cons.",
            ]
        )

        # self.f_policy[1].writerow(['Region ID', 'Timestep', 'Policy Size', 'Range of Shift', 'Delta Shift', 'Policy goals', 'Policy Years', 'Support'])
        f2 = open(path + "policy.csv", "w", newline="")
        self.f_policy = (
            f2,
            csv.writer(f2, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )

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
        self.f_household[1].writerow(
            ["Timestep", "Region"]
            + ["Household Threshold" for i in range(XML_init_values.Region_n_households)]
        )

        f5_beliefs = open(path + "household_beliefs.csv", "w", newline="")
        self.f_household_beliefs = (
            f5_beliefs,
            csv.writer(
                f5_beliefs, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            ),
        )
        self.f_household_beliefs[1].writerow(
            ["Timestep", "Region", "Household ID"]
            + [
                "belief T(y+99)=" + str(i)
                for i in np.arange(
                    Household.DISTRIB_MIN_VALUE,
                    Household.DISTRIB_MAX_VALUE,
                    Household.DISTRIB_RESOLUTION,
                )
            ]
        )
        f_household_assessment = open(
            path + "household_policy_assessment.csv", "w", newline=""
        )
        self.f_household_assessment = (
            f_household_assessment,
            csv.writer(
                f_household_assessment, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            ),
        )
        self.f_household_assessment[1].writerow(
            [
                "Timestep",
                "Region",
                "Household ID",
                "sensitivity to costs",
                "loss_and_damages",
                "mitigation_costs",
                "experienced_economic_context",
                "expected_loss_and_damages",
                "expected_mitigation_costs",
                "expected_economic_context",
                "expected temperature",
                "current policy level",
                "Resulting Utility"
            ]
        )


        # id region, emission control rate profile (historical + pledges)
        f6 = open(path + "emissions.csv", "w", newline="")
        self.f_emissions = (
            f6,
            csv.writer(f6, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )

        self.f_parameters = open(path + "parameters.txt", "w", newline="")

        # Initialise Regions
        df = pd.read_excel(
            "data/input/inputs_ABM/Distribution_of_income_or_consumption.xlsx",
            header=0,
            index_col=0,
            usecols="A,E:I",
            engine="openpyxl",
            skiprows=[0, 2, 3],
            skipfooter=62,
        )

        dict_regions_distribution_income = {}
        with open(
            "data/input/inputs_ABM/rice50_region_names_to_world_bank.json"
        ) as rice50_region_names_to_world_bank:
            dict_regions_rows = json.load(rice50_region_names_to_world_bank)
            for key in dict_regions_rows.keys():
                if key != "_comment":
                    dict_regions_distribution_income[key] = df.loc[
                        dict_regions_rows[key]
                    ].mean(axis=0)

        self.justice_model = justice_model
        i = 0
        self.regions = []
        for code in self.justice_model.data_loader.REGION_LIST:
            self.regions += [
                Region(
                    self,
                    i,
                    code,
                    timestep,
                    dict_regions_distribution_income,
                )
            ]
            i += 1
        self.N_regions = i

        # Coalitions (ad hoc, now a cycle graph)
        """ Coalitions could appear in the context of international agreement. They would result from a change in the 
        structure of the network graph linking all regions together. By default the graph is a connected graph of
        valency 2 (eg. a cycle graph).
        This is not used yet. """
        edges = np.array([[i, (i + 1) % self.N_regions] for i in range(self.N_regions)])
        self.coalitions = np.zeros((edges.max() + 1, edges.max() + 1))
        self.coalitions[edges[:, 0], edges[:, 1]] = 1
        self.coalitions[edges[:, 1], edges[:, 0]] = 1

        # Parameters for international negotiations
        self.Y_nego = (
            XML_init_values.TwoLevelsGame_Y_nego  # How many years between a new set of international negotiation rounds
        )
        self.Y_policy = XML_init_values.TwoLevelsGame_Y_policy  # How many years between a new set of regional policy update

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

        # IDEA uncomment to use the projections of alternative ecr scenarios over shortime
        # self.justice_model.information_model.generate_projections(timestep, self.regions)

        # Size is 57*1
        self.justice_model.consumption_per_capita = np.mean(
            self.justice_model.economy.get_consumption_per_capita_per_timestep(
                self.justice_model.savings_rate[:, timestep], timestep
            ),
            axis=1,
        )

        # Update income of quintiles within regions
        # Size is 57*1 (mean over all possibilities)
        net_average_consumption = self.justice_model.consumption_per_capita

        average_damages = np.mean(
            self.justice_model.economy.get_damages()[:, timestep, :], axis=1
        )
        average_abatement = np.mean(
            self.justice_model.economy.get_abatement()[:, timestep, :], axis=1
        )

        for region in self.regions:

            if region.code == "usa" and timestep == 58:
                print("flag pour debug")

            # Size is 57*5: consumption (net of damages and mitigation costs) for regions and each quintiles
            test = 0
            disaggregated_predmg_consumption = (
                5
                * net_average_consumption[region.id]
                * (1 + average_damages[region.id])
                / (1 - 0 * average_abatement[region.id])
                * region.distribution_income
            )

            #
            xi_damage = 0
            xi_abatement = 0

            damage_share = (
                np.power(region.distribution_income, xi_damage)
                * 1
                / np.sum(np.power(region.distribution_income, xi_damage))
            )
            abatement_share = (
                np.power(region.distribution_income, xi_abatement)
                * 1
                / np.sum(np.power(region.distribution_income, xi_abatement))
            )

            quintiles_damage_costs = (
                5
                * net_average_consumption[region.id]
                * average_damages[region.id]
                * damage_share
            )

            region.distribution_cost_damages = quintiles_damage_costs.copy()

            # TODO: You can put a 0* in front to test for loss and damages distribution costs only
            quintiles_abatement_costs = (
                5
                * net_average_consumption[region.id]
                * ((1 + average_damages[region.id]) * average_abatement[region.id])
                / (1 - average_abatement[region.id])
                * abatement_share
            )

            region.distribution_cost_mitigation = quintiles_abatement_costs.copy()

            # Theoretically it should be such that the sum of disaggregated post costs is equivalent to the net average consumption
            disaggregated_post_costs_consumption = (
                disaggregated_predmg_consumption
                - quintiles_damage_costs
                - quintiles_abatement_costs
            )

            (self.f_region)[1].writerow(
                [timestep, region.id, region.code]
                + [d for d in region.distribution_income]
                + [net_average_consumption[region.id]]
                + [average_damages[region.id], average_abatement[region.id]]
                + [q for q in quintiles_damage_costs]
                + [a for a in quintiles_abatement_costs]
                + [p for p in disaggregated_predmg_consumption]
                + [p for p in disaggregated_post_costs_consumption]
                + [np.mean(disaggregated_post_costs_consumption)]
            )

            region.update_regional_opinion(timestep)
            if timestep % self.Y_policy == 0:
                region.update_state_policy_from_constituency(timestep)

    def international_negotiations(self, timestep):
        """
        This function allows negotiators to make pledges of emissions cuts for the future years.
        Each negotiator have to come up with a target year associated to a cutting rate goal.
        Pledges are assumed to be linear by other negotiators in their implementation.
        """

        # TODO APN: This is very simplified way of seeing international negotiations. It might also not work properly when the delta is negative and large (not respecting maximum changing rate of emission control rate)
        if True:
            # Compute the average target year for ECR=1 in current regional policies
            avg_global_net_zero_year = 0
            for region in self.regions:
                policy = region.negotiator.policy
                avg_global_net_zero_year += policy[0, -1]
            avg_global_net_zero_year = avg_global_net_zero_year / len(self.regions)

            for region in self.regions:
                # Take into account possible public opinion pressure to delay ECR=1 target
                regional_earliest_achievable_end_target = np.ceil(
                    (1 - region.negotiator.policy[1, 1])
                    / region.negotiator.max_cutting_rate_gradient
                    + region.negotiator.policy[0, 1]
                )

                # take into account track record for expected target year
                regional_expected_end_target = region.negotiator.expected_year_ecr_max()

                # New policy is at least the earliest achievable
                # It is also a mean between expected and pledged
                # And the result should be dragged toward mean target if above
                tentative_end_year_pledge = (
                    region.negotiator.policy[0, -1] + regional_expected_end_target
                ) / 2
                if tentative_end_year_pledge > avg_global_net_zero_year:
                    coeff = (2000 - avg_global_net_zero_year) / (
                        2000 - tentative_end_year_pledge
                    )
                    tentative_end_year_pledge = (coeff) * tentative_end_year_pledge + (
                        1 - coeff
                    ) * avg_global_net_zero_year
                if tentative_end_year_pledge < regional_earliest_achievable_end_target:
                    print(
                        "For region ",
                        region.id,
                        " new pledge impossible because ",
                        tentative_end_year_pledge,
                        " < ",
                        regional_earliest_achievable_end_target,
                    )
                    tentative_end_year_pledge = regional_earliest_achievable_end_target

                region.negotiator.policy[0, -1] = tentative_end_year_pledge
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
