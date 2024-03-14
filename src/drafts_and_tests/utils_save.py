import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

#"../../data/output/SAVE_2024_03_13_2004/"
save_path = "../../data/output/SAVE_2024_03_14_0850/"
df = pd.read_csv(save_path+"policy.csv", header=None)

region = 0
region_mask = df[0] == region
count_rows = np.count_nonzero(region_mask)

emissions_colums = df[df.columns[3:15]]
years_colums = df[df.columns[15:27]]

alpha_space = np.geomspace(0.1,1, count_rows)
alpha_count = 0;
last_i = 0
for i in range(years_colums.shape[0]):
    if region_mask[i]:
        plt.plot(years_colums.iloc[i], emissions_colums.iloc[i], alpha=alpha_space[alpha_count])
        last_i = i
        alpha_count += 1

#plt.show()


start_year = 2015.
pol_at_start_year = 0;
end_year = 2300.
timestep_size = 1;
policy = np.array([years_colums.iloc[last_i],emissions_colums.iloc[last_i]]);

last_p_year = start_year
last_pol = pol_at_start_year

plt.title("region "+ str(region))
plt.xlabel("years")
plt.ylabel("emission control rate")


plt.figure(2)
df = pd.read_csv(save_path+"emissions.csv", header=None, sep=";")

region_mask = df[0] == region
count_rows = np.count_nonzero(region_mask)

emissions_colums = df[df.columns[1:]]
years_colums = np.linspace(start_year, end_year, num=286)

alpha_space = np.geomspace(0.1,1, count_rows)
alpha_count = 0;
last_i = 0
for i in range(emissions_colums.shape[0]):
    if region_mask[i]:
        plt.plot(years_colums, emissions_colums.iloc[i], alpha=alpha_space[alpha_count])
        last_i = i
        alpha_count += 1
plt.title("region "+ str(region))
plt.xlabel("years")
plt.ylabel("emission control rate")


plt.figure(3)
plt.plot(policy[0,:],policy[1,:],'k:')
plt.plot(years_colums, emissions_colums.iloc[last_i])
plt.legend(["Last policy", "Actual emission cutting rate pathway"])
plt.title("region "+ str(region))
plt.xlabel("years")
plt.show()


