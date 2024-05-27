from collections import defaultdict
import os
from ema_workbench.em_framework.optimization import ArchiveLogger, EpsilonProgress
from sklearn.preprocessing import MinMaxScaler
from deap.tools import _hypervolume
import numpy as np
import pandas as pd
from functools import partial
import itertools
import multiprocessing
import datetime
from sklearn.preprocessing import MinMaxScaler


def calculate_hypervolume(maxima, generation):
    return _hypervolume.hv.hypervolume(generation, maxima)


def transform_data(data, scaler):
    # data = data.copy()
    # setup a scaler

    # scale data
    transformed_data = scaler.transform(data)

    # handle directions
    # transformed_data = handle_directions(transformed_data, problem)

    return transformed_data


def load_archives(
    list_of_objectives=[
        "welfare_utilitarian",
        "years_above_temperature_threshold",
        "total_damage_cost",
        "total_abatement_cost",
    ],
    data_path="data/optimized_rbf_weights",
    file_name="UTILITARIAN_100000.tar.gz",
):
    """

    Returns
    -------


    """

    archives = ArchiveLogger.load_archives(f"{data_path}/{file_name}")

    list_of_archives = []
    number_of_objectives = len(list_of_objectives)

    # output_dir = f"../output/{name}/"
    for keys in archives.keys():
        print(keys)
        archives_by_keys = archives[keys]  # pd.read_csv(output_dir + i)

        generations = []
        for nfe, generation in archives_by_keys.groupby("Unnamed: 0"):
            # print("NFE: ", nfe, "\n")
            generation = generation.rename(
                {str(i): name for i, name in enumerate(list_of_objectives)},
                axis=1,
            )
            # Select only the last 4 columns
            generation = generation.iloc[:, -number_of_objectives:]

            # print(generation.columns)
            generations.append((nfe, generation))
            list_of_archives.append(generation)

            # archives[keys][int(i.split("_")[0])] = generations
    return archives, list_of_archives


def calculate_hypervolume_from_archives():

    archives, list_of_archives = load_archives()
    list_of_archives = pd.concat(list_of_archives).values

    scaler = MinMaxScaler()

    scaler.fit(list_of_archives)

    reference_set = transform_data(list_of_archives, scaler)  # .values

    print("reference_set", reference_set.shape)

    maxima = np.max(reference_set, axis=0)

    nfes = list(archives.keys())

    scores = []
    # for keys in archives_2.keys():
    #     archive = archives_2[keys]

    with multiprocessing.Pool() as pool:
        # Enumerate through the keys of the archives
        for key in archives.keys():
            if key != 0:
                # Extract the generation from the archives
                generation = archives[key]
                print("generation: ", generation.shape)
                # Select only the last 4 columns
                generation = generation.iloc[:, -4:]
                # Normalize the generation
                generation = transform_data(generation.values, scaler)
                # Calculate the hypervolume
                hv_results = pool.map(
                    partial(calculate_hypervolume, maxima), [generation]
                )

                scores.append(
                    pd.DataFrame.from_dict(dict(nfe=key, hypervolume=hv_results))
                )
            # print("len nfes and hv", len(nfes), len(hv_results))

            # print("shape", hv_results.shape)

        # hv_results = pool.map(partial(calculate_hypervolume, maxima), [reference_set])

        # scores.append(pd.DataFrame.from_dict(dict(nfe=nfes, hypervolume=hv_results)))
        # print("len nfes and hv", len(nfes), len(hv_results))

        # print("shape", hv_results.shape)
    # concat into single dataframe per rbf
    scores = pd.concat(scores, axis=0, ignore_index=True)
    scores.to_csv(f"./data/Util_100k_hv.csv")

    # hv = defaultdict(list)
    # for key in archives.keys():
    #     for nfe, generation in archives[key]:
    #         hv[key].append(calculate_hypervolume(maxima, generation.values))
    # return hv


if __name__ == "__main__":
    # Set path to two directories above current directory
    os.chdir("../../")
    print(os.getcwd())
    calculate_hypervolume_from_archives()
