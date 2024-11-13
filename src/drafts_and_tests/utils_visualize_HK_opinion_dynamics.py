import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_HK_OpDyn(directory, region):
    path_list = glob.glob(directory)
    # print(path_list)
    for save_path in path_list:

        f = open(save_path + "\parameters.txt", "r")
        df = pd.read_csv(
            save_path + "\HK_opinion_dynamics.csv", header=0, dtype="float64"
        )
        region_mask = df["Region"] == region
        df_region = df[region_mask]
        expected_dmg_opinion = df_region.filter(like="Belief Damages")
        support_opinion = df_region.filter(like="Support")

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
            support_opinion.values,
            "support_opinion (region " + str(region) + "; 100 households)",
            "Timesteps",
            "support_opinion",
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
    ax.set_xlabel(xstr)
    ax.set_ylabel(ystr)

