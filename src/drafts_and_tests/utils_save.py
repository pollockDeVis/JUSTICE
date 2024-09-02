import time

import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_policy(directory, region):
    path_list = glob.glob(directory)
    # print(path_list)
    for save_path in path_list:

        f = open(save_path + "\parameters.txt", "r")

        df = pd.read_csv(save_path + "\policy.csv", header=None, dtype="float64")

        region_mask = df[0] == region
        count_rows = np.count_nonzero(region_mask)

        emissions_colums = df[df.columns[5:8]]
        years_colums = df[df.columns[8:11]]
        current_time_colums = df[df.columns[1]]
        max_possible_shift = df[df.columns[3]]
        year_shift = df[df.columns[4]]
        support_shares = df[df.columns[11:]]

        plt.figure()
        plt.plot(current_time_colums[region_mask], year_shift[region_mask], "*")
        plt.plot(current_time_colums[region_mask], max_possible_shift[region_mask], "-")
        plt.title("YEAR SHIFT region " + str(region) + "(" + save_path + ")")
        plt.legend(
            [
                "Shift in policy",
                "Maximum achievable shift\n(compared to next policy goal)",
            ]
        )
        plt.xlabel("years")
        plt.ylabel("year shift")

        alpha_space = np.geomspace(0.1, 1, count_rows)
        alpha_count = 0
        last_i = 0
        plt.figure()
        for i in range(years_colums.shape[0]):
            if region_mask[i]:
                plt.plot(
                    years_colums.iloc[i],
                    emissions_colums.iloc[i],
                    alpha=alpha_space[alpha_count],
                )
                last_i = i
                alpha_count += 1

        # plt.show()

        start_year = 2015.0
        pol_at_start_year = 0
        end_year = 2300.0
        timestep_size = 1
        policy = np.array([years_colums.iloc[last_i], emissions_colums.iloc[last_i]])

        last_p_year = start_year
        last_pol = pol_at_start_year

        plt.title("POLICY region " + str(region) + "(" + save_path + ")")
        plt.xlabel("years")
        plt.ylabel("emission control rate")

        plt.figure()
        cmap = plt.get_cmap("rainbow", 57)
        for r in range(57):
            mask = df[0] == r
            plt.scatter(df[1][mask], years_colums[10][mask], color=cmap(r))

        time.sleep(1)
        plt.figure()
        df = pd.read_csv(save_path + "\emissions.csv", header=None, sep=",")

        region_mask = df[0] == region
        count_rows = np.count_nonzero(region_mask)

        emissions_colums = df[df.columns[1:]]
        years_colums = np.linspace(start_year, end_year, num=286)

        alpha_space = np.geomspace(0.1, 1, count_rows)
        alpha_count = 0
        last_i = 0
        for i in range(emissions_colums.shape[0]):
            if region_mask[i]:
                plt.plot(
                    years_colums[:80],
                    emissions_colums.iloc[i][:80],
                    alpha=alpha_space[alpha_count],
                )
                last_i = i
                alpha_count += 1
        plt.title("ACTUAL CONTROL RATE region " + str(region) + "(" + save_path + ")")
        plt.xlabel("years")
        plt.ylabel("emission control rate")

        plt.figure()
        plt.plot(policy[0, :], policy[1, :], "k:")
        plt.plot(years_colums, emissions_colums.iloc[last_i])
        plt.legend(["Last policy", "Actual emission cutting rate pathway"])
        plt.title(
            "LAST POLICY vs FULL RUN region " + str(region) + "(" + save_path + ")"
        )
        plt.xlabel("years")
        ax = plt.gca()
        ax.set_xlim(2015, 2100)

        time.sleep(1)
        plt.figure()
        df = pd.read_csv(save_path + "\share_opinions.csv", header=0, sep=",")
        df = df[df["Region"] == region]

        plt.stackplot(
            df["Timestep"] + 2015,
            df["share opposed"],
            df["share neutral"],
            df["share support"],
            labels=["Opposition", "Neutral", "Support"],
        )
        plt.legend(loc="upper left")

    plt.show()


visualize_policy("../../data/output/SAVE_2024_08_28_1345", 32)
