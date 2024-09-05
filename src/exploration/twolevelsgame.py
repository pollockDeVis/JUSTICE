"""
This is the negotiations module that determines the emission rates for next run step(s)
#'CO2 emissions in [GtCO2/year]'
"""

import os
from typing import Any
import numpy as np
from scipy.interpolate import interp1d
import copy

from src.exploration.LogFiles import print_log, LogFiles
from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.household import Household
from src.exploration.region import Region
import csv
import json
import pandas as pd


class TwoLevelsGame:
    """
    This class defines a model for a regional policy using a two-level game
    approach, taking into consideration both the local constituencies and the international
    negotiations.
    """

    def __init__(
        self,
        rng,
        justice_model,
        number_regions=57,
        timestep=0,
    ):
        """
        justice_model is a reference to the overarching justice model
        number_of_region refers to the number of regions in the model
        population_size_by_region refers to the number of households in each opinion dynamics model
        """
        self.rng = rng

        # Parameters for international negotiations
        self.Y_nego = (
            XML_init_values.TwoLevelsGame_Y_nego  # How many years between a new set of international negotiation rounds
        )
        self.Y_policy = (
            XML_init_values.TwoLevelsGame_Y_policy
        )  # How many years between a new set of regional policy update

        # Loading Data to initialise the REGIONS AND THEIR HOUSEHOLDS
        dict_regions_distribution_income = self.regions_distribution_income()
        # Climate worry=>initial VALENCE of EMOTION relative to "Are we mitigating enough?"
        dict_regions_climate_worry = self.regions_distribution_fromMetaOpinionSurvey(
            "climate_worry"
        )
        # Economic impact==>initial VALENCE of EMOTION relative to "Am I willing to pay for mitigation?"
        dict_regions_economic_impact = self.regions_distribution_fromMetaOpinionSurvey(
            "economic_impact"
        )
        # Temperature threshold==> (Climate awareness * 1.5 + 1/Threat at 20 years * 4) / (Climate awareness+Threat at 20 years)
        dict_regions_climate_awareness = (
            self.regions_distribution_fromMetaOpinionSurvey("climate_awareness")
        )
        dict_regions_threat_20_years = self.regions_distribution_fromMetaOpinionSurvey(
            "threat_20_years"
        )
        # Temperature belief==> harm future generation : +0 --> +4    + Climate awareness for variance (from 0.01, i know a lot, to 4, I've never heard of it), don't know = +2 with 1*awareness variance
        dict_regions_harm_future_gen = self.regions_distribution_fromMetaOpinionSurvey(
            "harm_future_gen"
        )
        # Government priority==> For initial OPINION regarding 'Are we doing enough mitigation?"
        dict_regions_gov_priority = self.regions_distribution_fromMetaOpinionSurvey(
            "gov_priority"
        )
        # Most responsible==> For initial OPINION regarding 'Am I willing to pay for more mitigation?"
        dict_regions_most_responsible = self.regions_distribution_fromMetaOpinionSurvey(
            "most_responsible"
        )
        # Country responsibility==>  NOT USED YET, but could be interesting for INTERNATIONAL NEGOTIATIONS
        dict_regions_country_responsibility = (
            self.regions_distribution_fromMetaOpinionSurvey("country_responsibility")
        )
        # Climate happening ==> For TEMPERATURE BELIEF at present time
        dict_regions_climate_happening = (
            self.regions_distribution_fromMetaOpinionSurvey("climate_happening")
        )

        dict_regions_freq_hear = self.regions_distribution_fromMetaOpinionSurvey(
            "freq_hear"
        )

        list_dicts = [
            dict_regions_distribution_income,
            dict_regions_climate_worry,
            dict_regions_economic_impact,
            dict_regions_climate_awareness,
            dict_regions_threat_20_years,
            dict_regions_harm_future_gen,
            dict_regions_gov_priority,
            dict_regions_most_responsible,
            dict_regions_country_responsibility,
            dict_regions_climate_happening,
            dict_regions_freq_hear,
        ]

        # Instantiating and Initialising REGIONS and their HOUSEHOLDS
        self.justice_model = justice_model
        i = 0
        self.regions = []
        for code in self.justice_model.data_loader.REGION_LIST:
            self.regions += [Region(rng, self, i, code, timestep, list_dicts)]
            i += 1
        self.N_regions = i

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

            # Size is 57*5: consumption (net of damages and mitigation costs) for regions and each quintiles
            test = 0
            coeff_abatement = 0  # Put to 0 to ignore abatement (hence the quintiles only get the loss and damages)
            disaggregated_predmg_consumption = (
                5
                * net_average_consumption[region.id]
                * (1 + average_damages[region.id])
                / (1 - coeff_abatement * average_abatement[region.id])
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
            abatement_share = coeff_abatement * (
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

            region.disaggregated_post_costs_consumption = (
                disaggregated_post_costs_consumption
            )

            (print_log.f_region)[1].writerow(
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
                # Earliest achievable ECR=1 according to max_cutting_rate_gradient
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
                    print_log.write_log(
                        LogFiles.MASKLOG_twolevelsgame,
                        "twolevelsgame.py",
                        "international_negotiations",
                        "For region "
                        + str(region.id)
                        + " new pledge impossible because (tentative) "
                        + str(tentative_end_year_pledge)
                        + " < "
                        + str(regional_earliest_achievable_end_target)
                        + " (earliest possible considering max_ecr)",
                    )
                    tentative_end_year_pledge = regional_earliest_achievable_end_target

                region.negotiator.policy[0, -1] = tentative_end_year_pledge
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

            print_log.f_emissions[1].writerow(
                [a.id]
                + [em for em in self.justice_model.emission_control_rate[a.id, :, 0]]
            )  # ECR are all the same for all ensemble, hence we onyl register the one for ensemble 0
        self.justice_model.data["emission_cutting_rate"][:, timestep, :] = (
            self.justice_model.emission_control_rate[:, timestep, :]
        )

    def regions_distribution_income(self):
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
        return dict_regions_distribution_income

    def regions_distribution_fromMetaOpinionSurvey(self, sheetname):
        df = pd.read_excel(
            "data/input/inputs_ABM/climate_change_opinion_survey_2022_aggregated.xlsx",
            header=0,
            index_col=0,
            engine="openpyxl",
            sheet_name=sheetname,
            converters={"": 0},
        ).fillna(0)

        dict = {}
        with open(
            "data/input/inputs_ABM/rice50_region_names_to_MetaSurvey.json"
        ) as rice50_region_names_to_MetaSurvey:
            dict_regions_cols = json.load(rice50_region_names_to_MetaSurvey)
            for key in dict_regions_cols.keys():
                if key != "_comment":
                    dict[key] = df[dict_regions_cols[key]].mean(axis=1)
        return dict
