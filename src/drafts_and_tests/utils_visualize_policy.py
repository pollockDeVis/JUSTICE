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

        df = pd.read_csv(save_path + "\policy.csv", header=0, dtype="float64")

        region_mask = df["Region ID"] == region
        count_rows = np.count_nonzero(region_mask)

        emissions_colums = df[df.columns[5:8]]
        years_colums = df[df.columns[8:11]]
        current_time_colums = df[df.columns[1]]
        max_possible_shift = df[df.columns[3]]
        feedback_policy_support = df["feedback shift to policy support"]
        year_shift = df[df.columns[4]]
        support_shares = df[["share_opposition",
                "share_neutral",
                "share_support"]]


        plt.figure()
        plt.plot(current_time_colums[region_mask], feedback_policy_support[region_mask],'*')

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
            mask = df["Region ID"] == r
            plt.plot(
                df["Timestep"][mask], years_colums["Net-zero year"][mask], color=cmap(r)
            )
        plt.xlabel("Negotiations year")
        plt.ylabel("Pledged year for ECR 100%")

        time.sleep(1)

        df = pd.read_csv(save_path + "\emissions.csv", header=None, sep=",")

        region_mask = df[0] == region
        count_rows = np.count_nonzero(region_mask)

        emissions_colums = df[df.columns[1:]]
        years_colums = np.linspace(start_year, end_year, num=286)

        alpha_space = np.geomspace(0.1, 1, count_rows)
        alpha_count = 0
        last_i = 0

        plt.figure()
        YEARS = []
        for i in range(emissions_colums.shape[0]):
            if region_mask[i]:
                plt.plot(
                    years_colums[:80],
                    emissions_colums.iloc[i][:80],
                    alpha=alpha_space[alpha_count],
                )
                last_i = i
                alpha_count += 1
                YEARS += [2015 + alpha_count * 5]
        plt.title("Region " + str(region) + " successive pledges and resulting policy")
        plt.xlabel("years")
        plt.ylabel("emission control rate")

        time.sleep(1)

        plt.figure()
        plt.plot(years_colums, emissions_colums.iloc[last_i])
        plt.legend(["Final emission cutting rate pathway"])
        plt.title("Final Policy for Region " + str(region))
        plt.xlabel("years")
        plt.ylabel("emission control rate")
        ax = plt.gca()
        ax.set_xlim(2015, 2100)

        time.sleep(1)
        plt.figure()
        df = pd.read_csv(save_path + "\share_opinions.csv", header=0, sep=",")
        df = df[df["Region"] == region]

        """plt.stackplot(
            df["Timestep"] + 2015,
            df["share opposed"],
            df["share neutral"],
            df["share support"],
            labels=["Opposition", "Neutral", "Support"],
        )"""

        ig, ax = plt.subplots(figsize=(12, 3))
        polygon_opposed = ax.fill_between(
            df["Timestep"] + 2015, 0, df["share opposed"], lw=0, color="none"
        )
        polygon_neutral = ax.fill_between(
            df["Timestep"] + 2015,
            df["share opposed"],
            df["share opposed"] + df["share neutral"],
            lw=0,
            color="none",
        )
        polygon_support = ax.fill_between(
            df["Timestep"] + 2015,
            df["share opposed"] + df["share neutral"],
            df["share opposed"] + df["share neutral"] + df["share support"],
            lw=0,
            color="none",
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        verts = np.vstack([p.vertices for p in polygon_opposed.get_paths()])
        filling = ax.imshow(
            np.abs(df["strength opposition"].to_numpy().reshape(1, -1)),
            cmap="Reds",
            vmax=1,
            vmin=0,
            aspect=35,
            extent=[
                verts[:, 0].min(),
                verts[:, 0].max(),
                verts[:, 1].min(),
                verts[:, 1].max(),
            ],
        )
        filling.set_clip_path(polygon_opposed.get_paths()[0], transform=ax.transData)

        verts = np.vstack([p.vertices for p in polygon_neutral.get_paths()])
        filling = ax.imshow(
            df["mean utility"].to_numpy().reshape(1, -1),
            cmap="RdYlGn",
            vmax=1,
            vmin=-1,
            aspect=35,
            extent=[
                verts[:, 0].min(),
                verts[:, 0].max(),
                verts[:, 1].min(),
                verts[:, 1].max(),
            ],
        )
        filling.set_clip_path(polygon_neutral.get_paths()[0], transform=ax.transData)

        verts = np.vstack([p.vertices for p in polygon_support.get_paths()])
        filling = ax.imshow(
            df["strength support"].to_numpy().reshape(1, -1),
            cmap="Greens",
            vmax=1,
            vmin=0,
            aspect=35,
            extent=[
                verts[:, 0].min(),
                verts[:, 0].max(),
                verts[:, 1].min(),
                verts[:, 1].max(),
            ],
        )
        filling.set_clip_path(polygon_support.get_paths()[0], transform=ax.transData)

        ax.set_xlim(
            xlim
        )  # the limits need to be set again because imshow sets zero margins
        ax.set_ylim(ylim)

        # plt.legend(["Share Opposition","Share Neutral","Share Support"],loc="upper left")

        plt.figure()
        plt.plot(df["Timestep"] + 2015, df["strength opposition"])
        plt.plot(df["Timestep"] + 2015, df["mean utility"])
        plt.plot(df["Timestep"] + 2015, df["strength support"])
        plt.legend(["Average Opposition", "Average All", "Average Support"])
        plt.xlabel("years")
        plt.ylabel("absolute value of opinion")

    plt.show()


def superpose_policies(directory):
    path_list = glob.glob(directory)
    # print(path_list)
    for save_path in path_list:

        f = open(save_path + "\parameters.txt", "r")
        print(f.read())
        df = pd.read_csv(save_path + "\emissions.csv", header=None, sep=",")

        years_colums = np.linspace(2015, 2300, num=286)
        emissions_colums = df[df.columns[1:]]
        plt.figure()
        #range 1 to 58 because we want to start a -1 and end a -57
        #if we started at 0, then -0 is zero and we print the first policy plan for one of the regions.
        for region in range(1,58):
            plt.plot(
                years_colums,
                emissions_colums.iloc[-region],
            )
            plt.title("Final ECR pathways for all Regions")
        plt.xlabel("years")
        plt.ylabel("emission control rate")
        ax = plt.gca()
        ax.set_xlim(2015, 2100)

    plt.show()


# visualize_policy("../../data/output/SAVE_2024_10_09_1403", 32)
superpose_policies("../../data/output/SAVE_2024_11_12_1301")
