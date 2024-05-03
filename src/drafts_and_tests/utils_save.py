import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt

# "../../data/output/SAVE_2024_03_13_2004/"
path_list = glob.glob("../../data/output/SAVE_2024_05_03_1656")
# print(path_list)
for save_path in path_list:
    print(save_path)
    f = open(save_path + "\parameters.txt", "r")
    print("\t" + f.read())
    df = pd.read_csv(save_path + "\policy.csv", header=None)

    region = 51

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
    plt.legend(["Shift in policy", "Maximum achievable shift"])
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
    df = pd.read_csv(save_path + "\emissions.csv", header=None, sep=";")

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
                years_colums, emissions_colums.iloc[i], alpha=alpha_space[alpha_count]
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
    plt.title("LAST POLICY vs FULL RUN region " + str(region) + "(" + save_path + ")")
    plt.xlabel("years")

    df = pd.read_csv(save_path + "\household.csv", header=None, sep=",")
    region_mask = df[0] == region
    timestep_column = df[df.columns[1]]
    expected_temperature_evaluation_column = df[df.columns[2]]
    experienced_economic_context_column = df[df.columns[3]]

    plt.figure()
    plt.plot(
        timestep_column[region_mask],
        expected_temperature_evaluation_column[region_mask],
        "+",
    )
    plt.legend(["Support related to temperature expectation"])
    plt.title(
        "HOUSEHOLDS support (temp expect based)\nregion "
        + str(region)
        + "("
        + save_path
        + ")"
    )
    plt.xlabel("timestep")

    plt.figure()
    plt.plot(
        timestep_column[region_mask],
        experienced_economic_context_column[region_mask],
        "+",
    )
    plt.legend(["Support related to economic context"])
    plt.title(
        "HOUSEHOLDS support (economic context based)\nregion "
        + str(region)
        + "("
        + save_path
        + ")"
    )
    plt.xlabel("timestep")

plt.show()
