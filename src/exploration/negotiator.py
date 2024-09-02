# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:22:14 2024

@author: apoujon
"""
import numpy as np
from scipy import interpolate as scint

from src.exploration.DataLoaderTwoLevelGame import (
    DataLoaderTwoLevelGame,
    XML_init_values,
)
from src.exploration.LogFiles import print_log


class Negotiator:
    def __init__(
        self,
        region,
    ):
        self.region_model = region
        self.policy_start_year = XML_init_values.Negotiator_policy_start_year
        self.ecr_start_year = XML_init_values.Negotiator_ecr_start_year
        self.policy_period = XML_init_values.Negotiator_policy_period
        self.policy = np.array(
            [
                [
                    self.policy_start_year,
                    self.policy_start_year + self.policy_period,
                    XML_init_values.Negotiator_policy_end_year,
                ],
                [
                    XML_init_values.Negotiator_ecr_start_year,
                    XML_init_values.Negotiator_ecr_first_term,
                    XML_init_values.Negotiator_ecr_end_year,
                ],
            ]
        )

        # Some parameters
        # Maximum emission cutting rate gradient per year
        self.max_cutting_rate_gradient = (
            XML_init_values.Negotiator_max_cutting_rate_gradient
        )

        self.regional_pressure_later_ecr = XML_init_values.Negotiator_policy_end_year
        self.regional_pressure_earlier_ecr = 0

    def international_netzero_proposal(self):
        return self.policy[0, -1]

    def delta_shift_range(self):
        return (
            self.max_cutting_rate_gradient * (self.policy[0, 1] - self.policy[0, 0])
            + self.policy[1, 0]
            - self.policy[1, 1]
        )

    def shifting_policy(self, timestep):
        current_time = (
            timestep
            + self.region_model.twolevelsgame_model.justice_model.time_horizon.start_year
        )
        if (
            self.policy[0, 1] == current_time
            and self.policy[0, 1] + self.policy_period < self.policy[0, 2]
            and self.policy[1, 1] != 1
        ):
            # identify new target
            now = self.policy[:, 1].copy()
            f = scint.interp1d(self.policy[0], self.policy[1], kind="linear")
            self.policy[:, 1] = np.array(
                [
                    self.policy[0, 1] + self.policy_period,
                    f(self.policy[0, 1] + self.policy_period),
                ]
            )
            # Update current policy
            self.policy[:, 0] = now

            # get the shift
            support = self.region_model.aggregate_households_opinions(
                timestep
            )  # support = share of opposition, neutral and support

            range_of_shift = self.delta_shift_range()
            if support[2] > support[0]:
                delta_shift = range_of_shift * support[2]
            elif support[2] <= support[0]:
                delta_shift = (self.policy[1, 0] - self.policy[1, 1]) * support[0]
            else:
                delta_shift = 0

            max_cutting_rate_gradient = self.max_cutting_rate_gradient

            # Shifting policy target
            # TODO APN: Forbid carbon capture for now (look @ AR6 database later)
            self.policy[1, 1] = min(delta_shift + self.policy[1, 1], 1)

            (print_log.f_policy)[1].writerow(
                [
                    self.region_model.id,
                    current_time,
                    self.policy.shape[1],
                    range_of_shift,
                    delta_shift,
                ]
                + [p for p in self.policy[1]]
                + [y for y in self.policy[0]]
                + [s for s in support]
            )

            # Verifying new target compatibility with end goal
            gradient = (self.policy[1, 2] - self.policy[1, 1]) / (
                self.policy[0, 2] - self.policy[0, 1]
            )
            if gradient > max_cutting_rate_gradient:
                # Compute earliest achievable target:
                earliest_possible_target = (
                    self.policy[1, 2] - self.policy[1, 1]
                ) / max_cutting_rate_gradient + self.policy[0, 1]
                self.regional_pressure_later_ecr = np.maximum(
                    self.regional_pressure_later_ecr, earliest_possible_target
                )

    def expected_year_ecr_max(self):
        # The time it will take from policy goal in policy[:,1] to reach target ecr=1 given track record
        # Plus the time assigned for goal in policy[:,1]
        # Reminder that policy[0,:] = years and policy[1,:] = ecr targets

        p1 = (self.policy[1, 1]-self.ecr_start_year)/(self.policy[0, 1]-self.policy_start_year)
        p2 = (self.policy[1, 0]-self.policy[1, 1])/(self.policy[0, 0]-self.policy[0, 1])
        p = (p1+p2)/2
        if p!=0:
            return np.ceil((1-self.policy[1, 1])/p+self.policy[0, 1])
        else:
            return self.policy[0, -1]

        return
