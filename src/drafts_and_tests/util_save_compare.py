import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt

# "../../data/output/SAVE_2024_03_13_2004/"
path_list_1 = glob.glob("../../data/output/SAVE_2024_07_10_1202")
path_list_2 = glob.glob("../../data/output/SAVE_2024_07_09_1724")
# print(path_list)
for save_path_1 in path_list_1:
    for save_path_2 in path_list_2:

        print(save_path_1)
        f1 = open(save_path_1 + "\parameters.txt", "r")
        print("\t" + f1.read())
        df1 = pd.read_csv(save_path_1 + "\policy.csv", header=None)
        print(save_path_2)
        f2 = open(save_path_2 + "\parameters.txt", "r")
        print("\t" + f2.read())
        df2 = pd.read_csv(save_path_2 + "\policy.csv", header=None)

        region = 53

        region_mask1 = df1[0] == region
        count_rows1 = np.count_nonzero(region_mask1)
        emissions_colums1 = df1[df1.columns[5:8]]
        years_colums1 = df1[df1.columns[8:11]]
        current_time_colums1 = df1[df1.columns[1]]
        max_possible_shift1 = df1[df1.columns[3]]
        year_shift1 = df1[df1.columns[4]]
        support_shares1 = df1[df1.columns[11:]]

        region_mask2 = df2[0] == region
        count_rows2 = np.count_nonzero(region_mask2)
        emissions_colums2 = df2[df2.columns[5:8]]
        years_colums2 = df2[df2.columns[8:11]]
        current_time_colums2 = df2[df2.columns[1]]
        max_possible_shift2 = df2[df2.columns[3]]
        year_shift2 = df2[df2.columns[4]]
        support_shares2 = df2[df2.columns[11:]]

        alpha_space = np.geomspace(0.1, 1, count_rows1)
        alpha_count = 0
        last_i = 0
        plt.figure()
        for i in range(years_colums1.shape[0]):
            if region_mask1[i]:
                plt.plot(
                    years_colums1.iloc[i],
                    emissions_colums1.iloc[i],
                    alpha=alpha_space[alpha_count],
                )
                last_i = i
                alpha_count += 1

        policy1 = np.array([years_colums1.iloc[last_i], emissions_colums1.iloc[last_i]])

        alpha_space = np.geomspace(0.1, 1, count_rows2)
        alpha_count = 0
        last_i = 0
        plt.figure()
        for i in range(years_colums2.shape[0]):
            if region_mask2[i]:
                plt.plot(
                    years_colums2.iloc[i],
                    emissions_colums2.iloc[i],
                    alpha=alpha_space[alpha_count],
                )
                last_i = i
                alpha_count += 1

        # plt.show()

        start_year = 2015.0
        pol_at_start_year = 0
        end_year = 2300.0
        timestep_size = 1

        policy2 = np.array([years_colums2.iloc[last_i], emissions_colums2.iloc[last_i]])

        last_p_year = start_year
        last_pol = pol_at_start_year

        plt.title("POLICY region " + str(region) + "(" + save_path_1 + ")")
        plt.xlabel("years")
        plt.ylabel("emission control rate")

        plt.figure()
        df1 = pd.read_csv(save_path_1 + "\emissions.csv", header=None, sep=";")
        df2 = pd.read_csv(save_path_2 + "\emissions.csv", header=None, sep=";")

        region_mask1 = df1[0] == region
        count_rows1 = np.count_nonzero(region_mask1)
        emissions_colums1 = df1[df1.columns[1:]]
        years_colums1 = np.linspace(start_year, end_year, num=286)

        region_mask2 = df2[0] == region
        count_rows2 = np.count_nonzero(region_mask2)
        emissions_colums2 = df2[df2.columns[1:]]
        years_colums2 = np.linspace(start_year, end_year, num=286)

        alpha_space = np.geomspace(0.1, 1, count_rows1)
        alpha_count = 0
        last_i = 0
        for i in range(emissions_colums1.shape[0]):
            if region_mask1[i]:
                plt.plot(
                    years_colums1,
                    emissions_colums1.iloc[i],
                    alpha=alpha_space[alpha_count],
                )
                last_i = i
                alpha_count += 1

        alpha_space = np.geomspace(0.1, 1, count_rows2)
        alpha_count = 0
        last_i = 0
        for i in range(emissions_colums2.shape[0]):
            if region_mask2[i]:
                plt.plot(
                    years_colums2,
                    emissions_colums2.iloc[i],
                    alpha=alpha_space[alpha_count],
                )
                last_i = i
                alpha_count += 1
        plt.title("ACTUAL CONTROL RATE region " + str(region) + "(" + save_path_1 + ")")
        plt.xlabel("years")
        plt.ylabel("emission control rate")

        plt.figure()
        #plt.plot(policy1[0, :], policy1[1, :], "k:")
        plt.plot(years_colums1, emissions_colums1.iloc[last_i])
        #plt.plot(policy2[0, :], policy2[1, :], "k:")
        plt.plot(years_colums2, emissions_colums2.iloc[last_i])
        plt.legend(
            [
                "pathway under strong support",
                "pathway under strong opposition",
            ],
            loc="lower right"
        )
        plt.title(
            "Policy pathways"
        )
        plt.xlabel("years")
        plt.ylabel("emission control rates")


plt.show()
