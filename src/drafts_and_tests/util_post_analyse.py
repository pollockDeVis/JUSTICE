import glob
import linecache
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

parameters_files = glob.glob("../../data/output/SAVE_2024_10_09*/parameters.txt")
policies_files = glob.glob("../../data/output/SAVE_2024_10_09*/policy.csv")

parameters = []
for p in parameters_files:
    param = []
    for i in range(32, 37):
        line = linecache.getline(p, i)
        x = line.split(" ")
        param.append(float(x[1]))
    linecache.clearcache()

    parameters.append(param)

policies = []
policies_first = []
for p in policies_files:
    with open(p) as f:
        for last_line in f:
            pass
        last_line = last_line.split(",")
        policies.append(float(last_line[10]))
        policies_first.append(float(last_line[-1]))

policies_first[-6] = 2040
print(policies_first)
print(policies)
input = parameters
# output = [policies[p] - policies_first[p] for p in range(len(policies))]
output = policies
input_labels = ["alpha1", "alpha2", "beta1", "beta2", "gamma"]
output_label = ["global net zero"]

#####################################################
#####################################################
"""Printing the input parameters"""
for i in range(len(input[0])):
    plt.figure(figsize=(10, 6))  # Create a new figure for each feature
    plt.plot([row[i] for row in input], "+")
    plt.xlabel("Index")
    plt.ylabel(input_labels[i])
    plt.title(f"Feature: {input_labels[i]}")
    plt.show()


"""Printing the output net-zero years"""
plt.plot(output,'o')
plt.xlabel("Run Index")
plt.ylabel("Net-Zero Year Achieved (all regions)")
plt.show()

plt.plot(policies_first,'o')
plt.xlabel("Run Index")
plt.ylabel("Net-Zero Year Achieved (first region)")
plt.show()


#####################################################
#####################################################
# Create a Pandas DataFrame from the input and output data
data = pd.DataFrame(input, columns=input_labels)
data["output"] = output

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create a heatmap to visualize the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix between Input and Output")
plt.show()

# Create scatter plots to visualize the relationship between each input and output
for label in input_labels:
    plt.figure(figsize=(6, 4))
    plt.scatter(data[label], data["output"])
    plt.xlabel(label)
    plt.ylabel("Net-Zero Year Achieved (all regions)")
    plt.show()

# TODO: get a more comprehensive view of the relationship between variables.
# For example:
# sns.pairplot(data)
# plt.show()

plt.show()

