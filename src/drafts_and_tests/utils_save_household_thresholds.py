import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_household_thresholds(directory, region):
    path_list = glob.glob(directory)
    # print(path_list)
    for save_path in path_list:

        f = open(save_path + "\parameters.txt", "r")
        df = pd.read_csv(
            save_path + "\household_opinion_and_trust.csv", header=0, dtype="float64"
        )
        region_mask = df["Region"] == region
        df_region = df[region_mask]
        expected_dmg_opinion = df_region.filter(like="expected_dmg_opinion")
        perceived_income_opinion = df_region.filter(like="perceived_income_opinion")
        literacy_opinion = df_region.filter(like="literacy_opinion")
        gini = df_region.filter(like="gini region")

        timesteps_hh = 2015 + df_region[df_region.columns[0]]

        plot_simple(
            timesteps_hh,
            expected_dmg_opinion.values,
            "expected_dmg_opinion (region " + str(region) + "; 100 households)",
            "Timesteps",
            "expected_dmg_opinion",
        )
        plot_simple(
            timesteps_hh,
            perceived_income_opinion.values,
            "perceived_income_opinion (region " + str(region) + "; 100 households)",
            "Timesteps",
            "perceived_income_opinion",
        )
        plot_simple(
            timesteps_hh,
            literacy_opinion.values,
            "literacy_opinion (region " + str(region) + "; 100 households)",
            "Timesteps",
            "literacy_opinion",
        )

        plot_simple(
            timesteps_hh,
            gini.values,
            "GINI (region " + str(region) + "; 100 households)",
            "Timesteps",
            "gini coeff",
        )


def plot_simple(timesteps, values, title, xstr, ystr):
    ##############################################
    # Plot the surface
    plt.figure()
    ax = plt.gca()
    surf = plt.plot(
        timesteps,
        values,
        # alpha=beliefs_array.transpose(),
    )

    # Set labels and title

    ax.set_title(title)
    ax.set_ylabel(xstr)
    ax.set_xlabel(ystr)


visualize_household_thresholds("../../data/output/SAVE_2024_09_18_1144", 32)
