# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:22:14 2024

@author: apoujon
"""
import numpy as np
from scipy import interpolate as scint


class Negotiator:
    def __init__(
        self,
        region,
        policy_start_year=2015.0,
        policy_end_year=2100.0,
        policy_period=5.0,
    ):
        self.region_model = region
        self.policy_start_year = policy_start_year
        self.policy = np.array(
            [
                [policy_start_year, policy_start_year + policy_period, policy_end_year],
                [0.0, 0.0, 1.0],
            ]
        )
        self.policy_period = policy_period

        # Some parameters
        # Maximum emission cutting rate gradient per year
        self.max_cutting_rate_gradient = 0.04

        self.regional_pressure_later_ecr = policy_end_year
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
            and self.policy[1,1] != 1
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
            if support[2]>support[0]:
                delta_shift = range_of_shift * support[2]
            elif support[2]<support[0]:
                delta_shift = - (self.policy[1, 0] - self.policy[1, 1]) * support[0]
            else:
                delta_shift = 0

            # delta_shift = (np.random.random() - 0.5) * 2
            # print(delta_shift)

            max_cutting_rate_gradient = self.max_cutting_rate_gradient

            # shift the target
            #TODO APN: to be removed not useful anylonger
            # Ensuring we are in the correct range of acceptable shift values
            if self.delta_shift_range() < delta_shift:
                delta_shift = self.delta_shift_range()
            elif delta_shift + self.policy[1, 1] - self.policy[1, 0] < 0:
                delta_shift = -self.policy[1, 1] + self.policy[1, 0]

            # Shifting policy target
            #TODO APN: Forbid carbon capture for now (look @ AR6 database later)
            self.policy[1, 1] = min(delta_shift + self.policy[1, 1], 1)

            (self.region_model.twolevelsgame_model.f_policy)[1].writerow(
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
                self.regional_pressure_later_ecr = np.max(
                    self.regional_pressure_later_ecr, earliest_possible_target
                )

    def expected_year_ecr_max(self):
        # The time it will take from policy goal in policy[:,1] to reach target ecr=1 given track record
        # Plus the time assigned for goal in policy[:,1]
        return (1 - self.policy[1, 1]) * self.policy[0, 1] / self.policy[
            1, 1
        ] + self.policy[0, 1]

        # policy = self.policy
        # shift_year = self.region_model.aggregate_households_opinions()
        # max_policy_rate = self.max_cutting_rate_gradient
        # current_time = (
        #     timestep
        #     + self.region_model.twolevelsgame_model.justice_model.time_horizon.start_year
        # )
        #
        # (self.region_model.twolevelsgame_model.f_policy)[1].writerow(
        #     [self.region_model.id, current_time, self.policy.shape[1], shift_year]
        #     + [p for p in self.policy[1]]
        #     + [y for y in self.policy[0]]
        # )
        #
        # # Choosing a year to modify at random (The year must be in the future)
        # p = policy[0, :-1] > current_time
        # if not p.any():
        #     return policy
        # year = self.region_model.twolevelsgame_model.justice_model.rng.choice(
        #     policy[0, :-1], 1, p=p / sum(p)
        # )[0]
        # year_ind = np.flatnonzero(year == policy[0, :])[0]
        # min_p = max(0, np.flatnonzero(policy[0, :-1] > current_time)[0] - 1)
        # regional_pressure_later_ecr = 0
        # regional_pressure_earlier_ecr = 0
        # max_iter = 10
        #
        #
        # temp = policy.T
        #
        # #Support for policy
        # if shift_year > 0:
        #     reshape_successful = False
        #     while not reshape_successful and shift_year > 0:
        #         # Reshape the policy curve until it fits with constrained emission rate gradient
        #         reshaped_policy = policy.copy()
        #         reshaped_policy[0, year_ind] = reshaped_policy[0, year_ind] + shift_year
        #         grad = np.gradient(reshaped_policy[1, :], reshaped_policy[0, :])
        #
        #         iter = 0
        #         while (
        #             (grad[:-1] > max_policy_rate).any()
        #             or (grad[:-1] < 0).any()
        #             or np.isnan(grad[:-1]).any()
        #         ) and iter < max_iter:
        #             ind = np.flatnonzero(
        #                 (grad[:-1] > max_policy_rate)
        #                 + (grad[:-1] < 0)
        #                 + (np.isnan(grad[:-1]))
        #             )[0]
        #             reshaped_policy[0, ind + 1] = (
        #                 1
        #                 / max_policy_rate
        #                 * abs(reshaped_policy[1, ind] - reshaped_policy[1, ind + 1])
        #                 + reshaped_policy[0, ind]
        #             )
        #             grad = np.gradient(reshaped_policy[1, :], reshaped_policy[0, :])
        #             iter += 1
        #
        #         # If some points are projected after end year for policy, increase regional pressure for slower international negotiations
        #         # Retry with a smaller shift (Shift_year-1) if Shift_year-1 > 0
        #         if policy[0, -1] < reshaped_policy[0, -1] and iter < max_iter:
        #             # print("too bad U:",shift_year," -> ",shift_year-1)
        #             shift_year = shift_year - 1
        #             regional_pressure_later_ecr += 1
        #         elif iter == max_iter:
        #             regional_pressure_later_ecr += 1
        #             # print("Impossible to find a right configuration")
        #             return policy
        #         else:
        #             # print("well done")
        #             policy = reshaped_policy
        #             reshape_successful = True
        #
        #     if shift_year == 0:
        #         return
        #         # print("Policy left unchanged")
        #
        # #Opposition to policy
        # else:
        #     reshape_successful = False
        #     while not reshape_successful and shift_year < 0:
        #         # Reshape the policy curve until it fits with constrained emission rate gradient
        #         reshaped_policy = policy.copy()
        #         reshaped_policy[0, year_ind] = reshaped_policy[0, year_ind] + shift_year
        #         grad = np.gradient(reshaped_policy[1, :], reshaped_policy[0, :])
        #
        #         iter = 0
        #         while (
        #             (grad[:-1] > max_policy_rate).any()
        #             or (grad[:-1] < 0).any()
        #             or np.isnan(grad[:-1]).any()
        #         ) and iter < max_iter:
        #             ind = np.flatnonzero(
        #                 (grad[:-1] > max_policy_rate)
        #                 + (grad[:-1] < 0)
        #                 + (np.isnan(grad[:-1]))
        #             )[0]
        #             # print("--------------------------------------")
        #             # print(grad)
        #             # print(reshaped_policy[0,ind])
        #             reshaped_policy[0, ind] = (
        #                 -1
        #                 / max_policy_rate
        #                 * abs(reshaped_policy[1, ind] - reshaped_policy[1, ind + 1])
        #                 + reshaped_policy[0, ind]
        #             )
        #             grad = np.gradient(reshaped_policy[1, :], reshaped_policy[0, :])
        #             iter += 1
        #             # print(-1/max_policy_rate * ( reshaped_policy[1,ind+1]-reshaped_policy[1,ind]))
        #             # print(reshaped_policy[0])
        #             # time.sleep(3)
        #
        #         # If some points are projected after end year for policy, increase regional pressure for slower international negotiations
        #         # Retry with a larger shift (Shift_year+1) if Shift_year+1 < 0
        #         if policy[0, min_p] > reshaped_policy[0, min_p] and iter < max_iter:
        #             # print("too bad Shift_year:",shift_year," -> ",shift_year+1)
        #             shift_year = shift_year + 1
        #             regional_pressure_earlier_ecr += 1
        #         elif iter == max_iter:
        #             regional_pressure_earlier_ecr += 1
        #             # print("Impossible to find a right configuration")
        #             return policy
        #         else:
        #             # print("well done")
        #             policy = reshaped_policy
        #             reshape_successful = True
        #
        #     if shift_year == 0:
        #         return
        #         # print("Policy left unchanged")
        #
        # self.policy = policy
        # self.regional_pressure_later_ecr = 0.7*self.regional_pressure_later_ecr +  0.3*regional_pressure_later_ecr
        # self.regional_pressure_earlier_ecr =  0.7*self.regional_pressure_earlier_ecr +  0.3*regional_pressure_earlier_ecr
        return
