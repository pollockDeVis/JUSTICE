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

    # scale data
    transformed_data = scaler.transform(data)
    return transformed_data


def load_archives(
    list_of_objectives=[],
    data_path=None,
    file_name=None,
):
    """

    Returns
    -------


    """
    # Assert if arguments are None
    assert data_path is not None, "data_path is None"
    assert file_name is not None, "file_name is None"
    # Assert if list_of_objectives is empty
    assert list_of_objectives, "list_of_objectives is empty"

    archives = ArchiveLogger.load_archives(f"{data_path}/{file_name}")

    list_of_archives = []
    number_of_objectives = len(list_of_objectives)
    print("Loading archives for ", file_name)
    for keys in archives.keys():
        # print(keys)

        archives_by_keys = archives[keys]

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
    print("Archives loaded")
    return archives, list_of_archives


def calculate_hypervolume_from_archives(
    list_of_objectives=[
        "welfare",
        "years_above_temperature_threshold",
        "welfare_loss_damage",
        "welfare_loss_abatement",
    ],
    input_data_path="data/optimized_rbf_weights/200k",
    file_name="PRIORITARIAN_200000.tar.gz",
    output_data_path="data/convergence_metrics",
    saving=True,
):

    archives, list_of_archives = load_archives(
        list_of_objectives=list_of_objectives,
        data_path=input_data_path,
        file_name=file_name,
    )
    list_of_archives = pd.concat(list_of_archives).values

    scaler = MinMaxScaler()

    scaler.fit(list_of_archives)

    reference_set = transform_data(list_of_archives, scaler)  # .values

    print("reference_set", reference_set.shape)

    maxima = np.max(reference_set, axis=0)

    nfes = list(archives.keys())

    # Check if there is a key with 0. If there is, remove it
    if 0 in nfes:
        nfes.remove(0)
    print("nfes: \n", nfes)
    scores = []
    overall_starttime = datetime.datetime.now()
    # TODO: Move multiprocessing to main
    with multiprocessing.Pool() as pool:
        # Enumerate through the keys of the archives
        nfe_archives = [
            transform_data(
                ((archives[key]).iloc[:, -(len(list_of_objectives)) :]).values, scaler
            )
            for key in nfes
            if key != 0
        ]
        print("Computing hypervolume for ", file_name)
        hv_results = pool.map(partial(calculate_hypervolume, maxima), nfe_archives)

        scores.append(pd.DataFrame.from_dict(dict(nfe=nfes, hypervolume=hv_results)))

    delta_time = datetime.datetime.now() - overall_starttime
    print(
        f"Time taken for Hypervolume Calculation: {delta_time.total_seconds():.3f} seconds"
    )
    scores = pd.concat(scores, axis=0, ignore_index=True)

    if saving:
        output_file = f"{output_data_path}/{file_name.split('.')[0]}_hv.csv"
        print(output_file)
        scores.to_csv(output_file)

    return scores


if __name__ == "__main__":
    filenames = [
        "UTILITARIAN_100000.tar.gz",
        "PRIORITARIAN_100000.tar.gz",
        "EGALITARIAN_100000.tar.gz",
        "SUFFICIENTARIAN_100000.tar.gz",
    ]

    # Enumerate through the filenames
    for filename in filenames:
        scores = calculate_hypervolume_from_archives(
            list_of_objectives=[
                "welfare",
                "years_above_temperature_threshold",
                "welfare_loss_damage",
                "welfare_loss_abatement",
            ],
            input_data_path="data/optimized_rbf_weights",
            file_name=filename,
            output_data_path="data/convergence_metrics",
            saving=True,
        )
