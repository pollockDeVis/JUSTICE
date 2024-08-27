import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import cm

# "../../data/output/SAVE_2024_03_13_2004/"
path_list = glob.glob("../../data/output/SAVE_2024_08_26_1330")
# print(path_list)
for save_path in path_list:
    print(save_path)
    f = open(save_path + "\parameters.txt", "r")
    print("\t" + f.read())
    df = pd.read_csv(save_path + "\household_beliefs.csv", header=0, dtype="float64")

    region = 50
    # household = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for household in range(10):

        region_mask = df["Region"] == region
        df_region = df[region_mask]
        household_mask = df_region["Household ID"] == household
        df_region_hh = df_region[household_mask]

        beliefs_hh = df_region_hh[df_region_hh.columns[3:]]
        timesteps_hh = df_region_hh[df_region_hh.columns[0]]

        beliefs_array = beliefs_hh.values

        # Create a meshgrid for plotting
        X, Y = np.meshgrid(np.arange(beliefs_array.shape[0]), np.arange(-2, 8, 0.1))

        # Create a figure and an axes object for the 3D plot

        # Plot the surface
        surf = ax.pcolor(
            X,
            Y,
            beliefs_array.transpose(),
            cmap=cm.viridis,
            alpha=beliefs_array.transpose(),
        )

    # Set labels and title

    ax.set_title("Household Beliefs")
    ax.set_ylabel("Temperature Increase")
    ax.set_xlabel("Timesteps")

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()
