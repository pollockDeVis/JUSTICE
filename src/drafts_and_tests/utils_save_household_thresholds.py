import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import cm

# "../../data/output/SAVE_2024_03_13_2004/"
path_list = glob.glob("../../data/output/SAVE_2024_08_27_0906")
# print(path_list)
for save_path in path_list:
    print(save_path)
    f = open(save_path + "\parameters.txt", "r")
    print("\t" + f.read())
    df = pd.read_csv(save_path + "\household.csv", header=0, dtype="float64")

    region = 16

    region_mask = df["Region"] == region
    df_region = df[region_mask]
    thresholds_hh = df_region.filter(like="Household Threshold")
    b0_Climate_hh = df_region.filter(like="B0 Climate Change")
    emotion_Climate_hh = df_region.filter(like="Emotion Climate Change")
    opinion_Climate_hh = df_region.filter(like="Opinion Climate Change")
    b0_Economy_hh = df_region.filter(like="B0 Economy")
    emotion_Economy_hh = df_region.filter(like="Emotion Economy")
    opinion_Economy_hh = df_region.filter(like="Opinion Economy")
    timesteps_hh = df_region[df_region.columns[0]]

    beliefs_array = thresholds_hh.values

    # Create a meshgrid for plotting

    # Create a figure and an axes object for the 3D plot

    ##############################################
    # Plot the surface
    plt.figure()
    ax = plt.gca()
    surf = plt.plot(
        np.arange(beliefs_array.shape[0]),
        beliefs_array,
        "+",
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
        np.arange(b0_Climate_hh.values.shape[0]),
        b0_Climate_hh.values,
        "+",
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
        np.arange(b0_Economy_hh.values.shape[0]),
        b0_Economy_hh.values,
        "+",
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
        np.arange(emotion_Climate_hh.values.shape[0]),
        emotion_Climate_hh.values,
        "+",
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
        np.arange(emotion_Economy_hh.values.shape[0]),
        emotion_Economy_hh.values,
        "+",
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
        np.arange(opinion_Climate_hh.values.shape[0]),
        opinion_Climate_hh.values,
        "+",
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
        np.arange(opinion_Economy_hh.values.shape[0]),
        opinion_Economy_hh.values,
        "+",
        # alpha=beliefs_array.transpose(),
    )

    # Set labels and title

    ax.set_title("Household Opinion for 'Am I willing to pay?'")
    ax.set_ylabel("Opinion")
    ax.set_xlabel("Timesteps")


plt.show()
