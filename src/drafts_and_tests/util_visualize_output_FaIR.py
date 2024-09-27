import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_output_FaIR(directory):
    path_list = glob.glob(directory)
    # print(path_list)
    for save_path in path_list:

        f = open(save_path + "\parameters.txt", "r")

        df = pd.read_csv(save_path + "\outputOfFaIR.csv", header=0, dtype="float64")



        temperature = df.filter(like="temp increase")
        timesteps = 2015 + df['Timestep']


        ##############################################
        # Plot the surface
        plt.figure()
        ax = plt.gca()
        surf = plt.plot(
            timesteps,
            temperature,
            # alpha=beliefs_array.transpose(),
        )

        # Set labels and title

        ax.set_title(
            "Resulting Temperature Increase"
        )
        ax.set_ylabel("Temperature Increase (Celsius)")
        ax.set_xlabel("Years")

    plt.show()


visualize_output_FaIR("../../data/output/SAVE_2024_09_19_1105")