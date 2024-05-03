"""
This module carries out different region mapping and configuration. Methods can generate region mapping dictionaries
and contains helper functions to aggregate JUSTICE regions into different aggregated regions.

# Note keep the country name convention consistent with the pycountry library and hence the ISO3 standard.
https://en.wikipedia.org/wiki/ISO_3166-1


"""

from collections import defaultdict
import json
import pycountry
import pandas as pd


def get_region_mapping(
    aggregated_region_dict, disaggregated_region_dict, similarity_threshold=0.02
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
            # print(key, rice_50_names[key], mapped_key)
            mapping_dictionary[key] = mapped_key

    return mapping_dictionary


def justice_region_aggregator(data_loader, region_config, data):

    # Read the json file
    with open("data/input/rice50_regions_dict.json", "r") as f:
        rice_50_dict_ISO3 = json.load(f)

    with open("data/input/rice50_region_names.json", "r") as f:
        rice_50_names = json.load(f)

    region_list = data_loader.REGION_LIST

    mapping_dictionary = get_region_mapping(
        aggregated_region_dict=region_config,
        disaggregated_region_dict=rice_50_dict_ISO3,
        similarity_threshold=0.02,
    )

    return mapping_dictionary
