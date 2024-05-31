import time

import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
import json

import logging, logging.config

logging.config.fileConfig("../../data/input/inputs_ABM/logger.conf")
logger = logging.getLogger("debug")


df = pd.read_excel(
    "../../data/input/inputs_ABM/Distribution_of_income_or_consumption.xlsx",
    header=0,
    index_col=0,
    usecols="A,E:I",
    engine="openpyxl",
    skiprows=[0, 2, 3],
    skipfooter=62,
)

# print(df.head())

dict_regions_rows = json.load(
    open("../../data/input/inputs_ABM/rice50_region_names_to_world_bank.json")
)

dict_regions_distribution_income = {}

for key in dict_regions_rows.keys():
    if key != "_comment":

        dict_regions_distribution_income[key] = df.loc[dict_regions_rows[key]].mean(
            axis=0
        )
        if np.sum(dict_regions_distribution_income[key]) != 100:
            print("=======", key, "=======")
            print(df.loc[dict_regions_rows[key]])
            print(dict_regions_distribution_income[key])
            logger.info("=======" + key + "=======")
            logger.info(dict_regions_distribution_income[key])
            print("TOTAL : ", np.sum(dict_regions_distribution_income[key]))


region = "rjan57"
dict_regions_distribution_income[region].plot.bar(
    rot=0, title="Distribution of Income in region " + region
)
plt.show()
