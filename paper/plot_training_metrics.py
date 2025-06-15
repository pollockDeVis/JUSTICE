"""

###############################
## TRAINING RESULTS PLOTTING ##
###############################

Use this script to compute per-seed and average metrics for the MOMARL experiments.
It also computes the final front and the extreme points of the final front.

"""


import os, json, numpy as np
from morl_baselines.common.performance_indicators import (
    hypervolume,
    sparsity,
    expected_utility,
    cardinality,
)
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.weights import equally_spaced_weights
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

NUM_SIMULATION_YEARS = 285

final_filtered_front = [] # Will contain the final filtered front considering all seeds
hypervolumes = [] # Will contain the hypervolumes of each seed, for final mean hv plot
sparsities = [] # Will contain the sparsities of each seed, for final mean sparsity plot
expected_utilities = [] # Will contain the expected utilities of each seed, for final mean eum plot
cardinalities = [] # Will contain the cardinalities of each seed, for final mean cardinality plot

extremes = {} # Will contain the extreme points of the final front for each seed
final_step_complete_data = {} # Will contain the complete data of the last timestep for each seed (for later use)

for seed in range(1, 11): # 10 training seeds

    path = f"paper/training_results/seed-{seed}/momarl/artefacts" # Path to the folder containing the weights of this seed
    data = {} # Will contain all the data, of all the weights, at each timestep, for this seed only
    for file in os.listdir(path): 
        # Load all the json files of all the weights (10 in each file)
        if file.endswith(".json"):
            with open(os.path.join(path, file), "r") as file:
                data_file = json.load(file)
    
                for key, value in data_file.items():
                    data[key] = value # This way we create a single dictionary with all the weights of this seed
    
    # Create a dictionary with the data organized by time steps
    time_steps = sorted(
        {int(step) for weight_data in data.values() for step in weight_data.keys()}
    )
    weights = [np.array(eval(key.replace(" ", ","))) for key in data.keys()]
    data_by_time = {ts: [] for ts in time_steps}
    for weight, metrics in data.items():
            for ts, values in metrics.items():
                if int(ts) in time_steps:
                    data_by_time[int(ts)].append(values)

    hv_ref_point = np.array([0, 0]) # Reference point for hypervolume calculation

    reward_dim = 2  # Dimensionality of objectives

    n_sample_weights = 10 # Number of sample weights for EUM, taken the default from the momappo code

    # Metrics initialization
    metrics = {
        "hypervolume": [],
        "sparsity": [],
        "expected_utility": [],
        "cardinality": [],
    }

    filtered_front = []

    # Compute metrics for each time step
    for ts in time_steps:
        current_points = np.array(data_by_time[ts]) # All the points at this timestep
        filtered_front = list(filter_pareto_dominated(current_points))  # Pareto filtering
        hv = hypervolume(hv_ref_point, filtered_front)  # Hypervolume
        sp = sparsity(filtered_front)  # Sparsity
        eum = expected_utility(
            filtered_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights)
        )  # EUM
        card = cardinality(filtered_front)  # Cardinality

        metrics["hypervolume"].append(hv) # Append the hypervolume of this timestep to the list of hypervolumes of this seed
        metrics["sparsity"].append(sp) # Append the sparsity of this timestep to the list of sparsities of this seed
        metrics["expected_utility"].append(eum) # Append the expected utility of this timestep to the list of expected utilities of this seed
        metrics["cardinality"].append(card) # Append the cardinality of this timestep to the list of cardinalities of this seed
    
    hypervolumes.append(metrics["hypervolume"]) # Append the hypervolumes of this seed to the list of hypervolumes (list of lists)
    sparsities.append(metrics["sparsity"]) # Append the sparsities of this seed to the list of sparsities (list of lists)
    expected_utilities.append(metrics["expected_utility"]) # Append the expected utilities of this seed to the list of expected utilities (list of lists)
    cardinalities.append(metrics["cardinality"]) # Append the cardinalities of this seed to the list of cardinalities (list of lists)
    
    final_filtered_front.extend(filtered_front) # Add the elements of the filtered front to the final front

    # Save individual seed plots for hypervolume, sparsity, expected utility and cardinality

    plt.plot(time_steps, metrics["hypervolume"], label="Hypervolume")
    plt.xlabel("Timesteps")
    plt.ylabel("Hypervolume")
    
    plt.legend()
    plt.savefig(f"paper/training_results/seed-{seed}/momarl/hypervolume.svg", format="svg")
    plt.close()
    
    plt.plot(time_steps, metrics["sparsity"], label="Sparsity", color="orange")
    plt.xlabel("Timesteps")
    plt.ylabel("Sparsity")
    plt.legend()
    plt.savefig(f"paper/training_results/seed-{seed}/momarl/sparsity.svg", format="svg")
    plt.close()
    
    plt.plot(time_steps, metrics["expected_utility"], label="EUM", color="green")
    plt.xlabel("Timesteps")
    plt.ylabel("Expected Utility Maximization")
    plt.legend()
    plt.savefig(f"paper/training_results/seed-{seed}/momarl/expected_utility.svg", format="svg")
    plt.close()

    plt.plot(time_steps, metrics["cardinality"], label="Cardinality", color="red")
    plt.xlabel("Timesteps")
    plt.ylabel("Cardinality")
    #plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.legend()
    plt.savefig(f"paper/training_results/seed-{seed}/momarl/cardinality.svg", format="svg")
    plt.close()

    # Dictionary with the weights as keys and the last timesteps's values of the weights for this seed as values
    weights_and_last_values = {weights: list(values.values())[-1] for weights, values in data.items()} 

    # Add it to the dictionary of the complete data of the last timestep for each seed (for later use)
    final_step_complete_data[seed] = weights_and_last_values 

    # From data_by_time, extract the data for the last timestep
    last_timestep_data = data_by_time[time_steps[-1]]
    last_timestep_data = np.array(last_timestep_data)

    # Find the maximum value of the first objective and the maximum value of the second objective
    left_extreme = np.max(last_timestep_data[:,0])
    right_extreme = np.max(last_timestep_data[:,1])

    # Find the corresponding point in the last timestep data
    left_extreme_point = last_timestep_data[np.where(last_timestep_data[:,0] == left_extreme)]
    right_extreme_point = last_timestep_data[np.where(last_timestep_data[:,1] == right_extreme)]

    # Find the corresponding weight in the weights_and_last_values dictionary
    left_extreme_weight = [key for key, value in weights_and_last_values.items() if value == left_extreme_point[0].tolist()]
    right_extreme_weight = [key for key, value in weights_and_last_values.items() if value == right_extreme_point[0].tolist()]

    print("Left extreme for seed", seed, ":", left_extreme_weight[0], ":", left_extreme_point[0])
    print("Right extreme for seed", seed, ":", right_extreme_weight[0], ":", right_extreme_point[0])
    print("")

    # Save the left and right extreme points for this seed
    extremes[seed] = {left_extreme_weight[0]: left_extreme_point[0].tolist(), right_extreme_weight[0] :right_extreme_point[0].tolist()}

mean_hypervolumes = np.mean(hypervolumes, axis=0) # Compute the mean hypervolume of all the seeds
std_hypervolumes = np.std(hypervolumes, axis=0) # Compute the standard deviation of the hypervolumes of all the seeds
mean_sparsities = np.mean(sparsities, axis=0) # Compute the mean sparsity of all the seeds
std_sparsities = np.std(sparsities, axis=0) # Compute the standard deviation of the sparsities of all the seeds
mean_expected_utilities = np.mean(expected_utilities, axis=0) # Compute the mean expected utility of all the seeds
std_expected_utilities = np.std(expected_utilities, axis=0) # Compute the standard deviation of the expected utilities of all the seeds
mean_cardinalities = np.mean(cardinalities, axis=0) # Compute the mean cardinality of all the seeds
std_cardinalities = np.std(cardinalities, axis=0) # Compute the standard deviation of the cardinalities of all the seeds

# Save the final plots for hypervolume, sparsity and expected utility averaged over all the seeds

plt.figure()
plt.plot(time_steps, mean_hypervolumes, label="Hypervolume")
plt.fill_between(time_steps, mean_hypervolumes - std_hypervolumes, mean_hypervolumes + std_hypervolumes, alpha=0.075)
plt.xlabel("Timesteps", fontsize=12)
#plt.ylabel("Hypervolume")
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.gca().xaxis.get_offset_text().set_fontsize(10)
plt.gca().yaxis.get_offset_text().set_fontsize(10)
plt.legend()
plt.savefig("paper/training_results/mean_hypervolume.svg", format="svg")
plt.close()

plt.figure()
plt.plot(time_steps, mean_sparsities, label="Sparsity", color="orange")
plt.fill_between(time_steps, mean_sparsities - std_sparsities, mean_sparsities + std_sparsities, alpha=0.075, color="orange")
plt.xlabel("Timesteps", fontsize=12)
#plt.ylabel("Sparsity")
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.gca().xaxis.get_offset_text().set_fontsize(10)
plt.gca().yaxis.get_offset_text().set_fontsize(10)
plt.legend()
plt.savefig("paper/training_results/mean_sparsity.svg", format="svg")
plt.close()

plt.figure()
plt.plot(time_steps, mean_expected_utilities, label="Expected Utility", color="green")
plt.fill_between(time_steps, mean_expected_utilities - std_expected_utilities, mean_expected_utilities + std_expected_utilities, alpha=0.075, color="green")
plt.xlabel("Timesteps", fontsize=12)
#plt.ylabel("Expected Utility")
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.gca().xaxis.get_offset_text().set_fontsize(10)
plt.gca().yaxis.get_offset_text().set_fontsize(10)
plt.legend()
plt.savefig("paper/training_results/mean_expected_utility.svg", format="svg")
plt.close()

plt.figure()
plt.plot(time_steps, mean_cardinalities, label="Cardinality", color="red")
plt.fill_between(time_steps, mean_cardinalities - std_cardinalities, mean_cardinalities + std_cardinalities, alpha=0.075, color="red")
plt.xlabel("Timesteps", fontsize=12)
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_offset_text().set_fontsize(10)
plt.gca().yaxis.get_offset_text().set_fontsize(10)
plt.legend()
plt.savefig("paper/training_results/mean_cardinality.svg", format="svg")
plt.close()

# Filter the final front to obtain the final pareto front
final_filtered_front = filter_pareto_dominated(final_filtered_front)
final_filtered_front=np.array(final_filtered_front)

# Modify the final front to have the global average annual temperature raise in the x axis
modified_final_filtered_front = np.copy(final_filtered_front)
modified_final_filtered_front[:,0] = 1/((final_filtered_front[:,0]/NUM_SIMULATION_YEARS)) # Inverse the values to have the global average annual temperature raise

print("Final Front:", "\n\n", modified_final_filtered_front)
print()

# We want to add another point to the front, which is the RICE-N model point
# This point is: [2.3, 260000]

rice_n_point = np.array([2.3, 260000])

right_extreme = np.max(modified_final_filtered_front[:,0])

right_extreme_point = modified_final_filtered_front[np.where(modified_final_filtered_front[:,0] == right_extreme)]
left_extreme_point = modified_final_filtered_front[-4]
compromise_point = modified_final_filtered_front[-2]

# remove the left extreme point from the final front

modified_final_filtered_front = np.delete(modified_final_filtered_front, -4, axis=0)

# remove the right extreme point from the final front

modified_final_filtered_front = np.delete(modified_final_filtered_front, np.where(modified_final_filtered_front[:,0] == right_extreme), axis=0)

# remove the compromise

modified_final_filtered_front = np.delete(modified_final_filtered_front, np.where(modified_final_filtered_front[:,0] == compromise_point[0]), axis=0)

# Save the final front plot with the RICE-N point and extreme points
plt.scatter(modified_final_filtered_front[:,0], modified_final_filtered_front[:,1], marker="o", facecolors='none', edgecolors='black')
plt.scatter(rice_n_point[0], rice_n_point[1], color='red', marker="*", label='RICE50+ Policy')
plt.scatter(left_extreme_point[0], left_extreme_point[1], color='purple', label='Climate Policy')
plt.scatter(right_extreme_point[:,0], right_extreme_point[:,1], color='lightgreen', label='Economic Policy')
plt.scatter(compromise_point[0], compromise_point[1], color='orange', label='Compromise Policy')
plt.xlabel("Global Average Annual Temperature Raise", fontsize=12)
plt.ylabel("Total Global Economic Output", fontsize=12)
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.gca().xaxis.get_offset_text().set_fontsize(10)
plt.gca().yaxis.get_offset_text().set_fontsize(10)
plt.legend()
plt.savefig("paper/training_results/final_front.svg", format="svg")
plt.close()

# Save the final_step_complete_data dictionary to a json file, in case you want to use it later
with open("paper/training_results/final_step_complete_data.json", "w") as file:
    json.dump(final_step_complete_data, file)



