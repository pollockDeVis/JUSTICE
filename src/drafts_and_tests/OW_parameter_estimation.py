import time

import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
import json

import scipy
from scipy.optimize import minimize

import logging, logging.config

logging.config.fileConfig("../../data/input/inputs_ABM/logger.conf")
logger = logging.getLogger("debug")


df = pd.read_excel(
    "../../data/input/inputs_ABM/world_100bin.xlsx",
    header=0,
    index_col=0,
    usecols="B,C,E,F,H,I",
    engine="openpyxl",
)

print(df.head())
print(df.index)

dict_regions_rows = json.load(
    open("../../data/input/inputs_ABM/rice50_regions_dict.json")
)

dict_regions_distribution_income = {}

# Loading the data of welfare shares into dict_regions_distribution_income
for key in dict_regions_rows.keys():
    if key != "_comment":

        try:
            temp_df = df.loc[dict_regions_rows[key]]
            # filter by income
            temp_df = temp_df.loc[temp_df["welfare_type"] == "income"]
            temp_df = temp_df.groupby("percentile").mean(numeric_only=True)
            dict_regions_distribution_income[key] = temp_df
            """if np.sum(dict_regions_distribution_income[key]) != 100:
                print("=======", key, "=======")
                print(df.loc[dict_regions_rows[key]])
                print(dict_regions_distribution_income[key])
                logger.info("=======" + key + "=======")
                logger.info(dict_regions_distribution_income[key])
                print("TOTAL : ", np.sum(dict_regions_distribution_income[key]))"""
        except KeyError:
            print("Error not found for key: ", key)




dict_regions_distribution_law = {}
for region in dict_regions_distribution_income.keys():

    data = dict_regions_distribution_income[region]["welfare_share"]
    if not data.empty:


        x = np.random.choice(data.index / 100, 2, p=data)
        shapelocscale = scipy.stats.weibull_min.fit(x)
        # print(loc, scale)
        dict_regions_distribution_law[region] = shapelocscale



region = "chn"
plt.figure()
dict_regions_distribution_income[region].plot.bar(
    y="welfare_share", rot=0, title="Distribution of Income in region " + region
)

plt.figure()
dict_regions_distribution_income[region].cumsum().plot(
    y="welfare_share", rot=0, title="Distribution of Income in region " + region
)
x=np.arange(1,100,step=0.5)
plt.plot(x,x/100,":")
plt.show()

