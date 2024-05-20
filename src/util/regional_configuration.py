"""
This module carries out different region mapping and configuration. Methods can generate region mapping dictionaries
and contains helper functions to aggregate JUSTICE regions into different aggregated regions.

# Note keep the country name convention consistent with the pycountry library and hence the ISO3 standard.
https://en.wikipedia.org/wiki/ISO_3166-1


"""

from collections import defaultdict
import json
import pycountry  # TODO Use this to get country names for ISO3 codes
import pandas as pd
import numpy as np


def get_region_mapping(
    aggregated_region_dict, disaggregated_region_dict, similarity_threshold=0.01
):
    mapping_dictionary = {}

    inverted_index = defaultdict(list)
    for k, v in aggregated_region_dict.items():
        for item in v:
            inverted_index[item].append(k)

    # Map using inverted index
    for key, value in disaggregated_region_dict.items():
        max_similarity = 0
        mapped_key = None

        # Calculate Jaccard similarity using inverted index
        similarity_scores = defaultdict(int)
        for item in value:
            for k in inverted_index[item]:
                similarity_scores[k] += 1

        for k, score in similarity_scores.items():
            similarity = score / (len(value) + len(aggregated_region_dict[k]) - score)
            if similarity > max_similarity:
                max_similarity = similarity
                mapped_key = k

        if max_similarity > similarity_threshold:  # Adjust threshold as needed
            mapping_dictionary[key] = mapped_key

        # Inverted mapping
        inverted_mapping = {}

        for key, value in mapping_dictionary.items():
            if value not in inverted_mapping:
                inverted_mapping[value] = [key]
            else:
                inverted_mapping[value].append(key)

    return inverted_mapping


def justice_region_aggregator(
    data_loader,
    region_config,
    data,
    similarity_threshold=0.01,
):
    """
    This function aggregates the emissions data from the JUSTICE model regions to the aggregated regions.

    Args:
    data_loader (DataLoader): An instance of the DataLoader class.
    region_config (dict): A dictionary containing the region configuration for the aggregation.
    data (np.ndarray): A 3D numpy array containing the emissions data.
    similarity_threshold (float): The similarity threshold for mapping regions, default is 0.01.

    """

    # Read the json file
    with open("data/input/rice50_regions_dict.json", "r") as f:
        rice_50_dict_ISO3 = json.load(f)

    with open("data/input/rice50_region_names.json", "r") as f:
        rice_50_names = json.load(f)

    aggregated_region_list = data_loader.REGION_LIST

    mapping_dictionary = get_region_mapping(
        aggregated_region_dict=region_config,
        disaggregated_region_dict=rice_50_dict_ISO3,
        similarity_threshold=similarity_threshold,
    )

    # Loop throught the keys and values of the mapping dictionary
    mapped_names = {}
    for key, value in mapping_dictionary.items():
        # Loop through the values and get the names
        mapped_names[key] = [rice_50_names[region] for region in value]

    # Create a dictionary to map regions to their indices
    region_index_map = {
        region: index for index, region in enumerate(aggregated_region_list)
    }

    # Preallocate the aggregated_emissions array
    aggregated_data = np.zeros((len(mapping_dictionary), data.shape[1], data.shape[2]))

    aggregated_region_list = []
    # Iterate through the inverted mapping and aggregate the emissions
    for index, (key, value) in enumerate(mapping_dictionary.items()):
        aggregated_region_list.append(key)
        # Get the indices of the regions in the value list
        indices = [region_index_map[region] for region in value]
        # Sum the emissions for the regions in the value list
        aggregated_data[index, :, :] = np.sum(data[indices, :, :], axis=0)

    return aggregated_region_list, aggregated_data
