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

        df = pd.read_csv(save_path + "\household.csv", header=0, dtype="float64")

        region_mask = df["Region"] == region
        df_region = df[region_mask]
        thresholds_hh = df_region.filter(like="Household Threshold")
        b0_Climate_hh = df_region.filter(like="B0 Climate Change")
        emotion_Climate_hh = df_region.filter(like="Emotion Climate Change")
        opinion_Climate_hh = df_region.filter(like="Opinion Climate Change")
        b0_Economy_hh = df_region.filter(like="B0 Economy")
        emotion_Economy_hh = df_region.filter(like="Emotion Economy")
        opinion_Economy_hh = df_region.filter(like="Opinion Economy")
        h_Economy_hh = df_region.filter(like="H Economy")
        h_Climate_hh = df_region.filter(like="H Climate Change")
        timesteps_hh = 2015 + df_region[df_region.columns[0]]

        beliefs_array = thresholds_hh.values

        # Create a meshgrid for plotting

        # Create a figure and an axes object for the 3D plot

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps_hh,
            beliefs_array,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household Temperature Elevation Threshold")
        ax.set_ylabel("Threshold")
        ax.set_xlabel("Timesteps")

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps_hh,
            b0_Climate_hh.values,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household B0 for 'Are we mitigating enough?'")
        ax.set_ylabel("B0")
        ax.set_xlabel("Timesteps")

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps_hh,
            b0_Economy_hh.values,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household B0 for 'Am I willing to pay?'")
        ax.set_ylabel("B0")
        ax.set_xlabel("Timesteps")

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps_hh,
            emotion_Climate_hh.values,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household Emotion for 'Are we mitigating enough?'")
        ax.set_ylabel("Emotion")
        ax.set_xlabel("Timesteps")

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps_hh,
            h_Climate_hh.values,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household H for 'Are we mitigating enough?'")
        ax.set_ylabel("H")
        ax.set_xlabel("Timesteps")

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps_hh,
            emotion_Economy_hh.values,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household Emotion for 'Am I willing to pay?'")
        ax.set_ylabel("Emotion")
        ax.set_xlabel("Timesteps")

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps_hh,
            h_Economy_hh.values,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household H for 'Am I willing to pay?'")
        ax.set_ylabel("H")
        ax.set_xlabel("Timesteps")

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps_hh,
            opinion_Climate_hh.values,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household Opinion for 'Are we mitigating enough?'")
        ax.set_ylabel("Opinion")
        ax.set_xlabel("Timesteps")

        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = ax.plot(
            timesteps_hh,
            opinion_Economy_hh.values,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title("Household Opinion for 'Am I willing to pay?'")
        ax.set_ylabel("Opinion")
        ax.set_xlabel("Timesteps")

    plt.show()


visualize_household_thresholds("../../data/output/SAVE_2024_08_28_1624", 32)
