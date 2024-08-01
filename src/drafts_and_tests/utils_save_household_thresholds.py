import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import cm

# "../../data/output/SAVE_2024_03_13_2004/"
path_list = glob.glob("../../data/output/SAVE_2024_08_01_1651")
# print(path_list)
for save_path in path_list:
    print(save_path)
    f = open(save_path + "\parameters.txt", "r")
    print("\t" + f.read())
    df = pd.read_csv(save_path + "\household.csv", header=0, dtype="float64")

    region = 50
    fig = plt.figure()
    ax = fig.add_subplot(111)


    region_mask = df["Region"] == region
    df_region = df[region_mask]
    thresholds_hh = df_region[df_region.columns[2:]]
    timesteps_hh = df_region[df_region.columns[0]]

    beliefs_array = thresholds_hh.values

    # Create a meshgrid for plotting


    # Create a figure and an axes object for the 3D plot

    # Plot the surface
    surf = ax.plot(
        np.arange(beliefs_array.shape[0]),
        beliefs_array,
        '+'
        #alpha=beliefs_array.transpose(),
    )

    # Set labels and title

    ax.set_title("Household Temperature Elevation Threshold")
    ax.set_ylabel("Threshold")
    ax.set_xlabel("Timesteps")

    # Add a color bar
    #fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()
