import numpy as np
import pandas as pd
import json

# Education
"""
[dict_regions_distribution_income,
dict_regions_climate_worry,
dict_regions_economic_impact,
dict_regions_climate_awareness,
dict_regions_threat_20_years,
dict_regions_harm_future_gen,
dict_regions_gov_priority,
dict_regions_most_responsible,
dict_regions_country_responsibility,]"""


# Climate worry=>initial VALENCE of EMOTION relative to "Are we mitigating enough?"
df = pd.read_excel(
    "../../data/input/inputs_ABM/climate_change_opinion_survey_2022_aggregated.xlsx",
    header=0,
    index_col=0,
    engine="openpyxl",
    sheet_name='climate_worry'
)

dict_regions_climate_worry = {}
with open(
    "../../data/input/inputs_ABM/rice50_region_names_to_MetaSurvey.json"
) as rice50_region_names_to_MetaSurvey:
    dict_regions_cols = json.load(rice50_region_names_to_MetaSurvey)
    for key in dict_regions_cols.keys():
        if key != "_comment":
            dict_regions_climate_worry[key] = df[
                dict_regions_cols[key]
            ].mean(axis=1)


# Economic impact==>initial VALENCE of EMOTION relative to "Am I willing to pay for mitigation?"
df = pd.read_excel(
    "../../data/input/inputs_ABM/climate_change_opinion_survey_2022_aggregated.xlsx",
    header=0,
    index_col=0,
    engine="openpyxl",
    sheet_name='economic_impact'
)

dict_regions_economic_impact = {}
with open(
    "../../data/input/inputs_ABM/rice50_region_names_to_MetaSurvey.json"
) as rice50_region_names_to_MetaSurvey:
    dict_regions_cols = json.load(rice50_region_names_to_MetaSurvey)
    for key in dict_regions_cols.keys():
        if key != "_comment":
            dict_regions_economic_impact[key] = df[
                dict_regions_cols[key]
            ].mean(axis=1)


# Temperature threshold==> (Climate awareness * 1.5 + 1/Threat at 20 years * 4) / (Climate awareness+Threat at 20 years)
df = pd.read_excel(
    "../../data/input/inputs_ABM/climate_change_opinion_survey_2022_aggregated.xlsx",
    header=0,
    index_col=0,
    engine="openpyxl",
    sheet_name='climate_awareness'
)

dict_regions_climate_awareness = {}
with open(
    "../../data/input/inputs_ABM/rice50_region_names_to_MetaSurvey.json"
) as rice50_region_names_to_MetaSurvey:
    dict_regions_cols = json.load(rice50_region_names_to_MetaSurvey)
    for key in dict_regions_cols.keys():
        if key != "_comment":
            dict_regions_climate_awareness[key] = df[
                dict_regions_cols[key]
            ].mean(axis=1)

df = pd.read_excel(
    "../../data/input/inputs_ABM/climate_change_opinion_survey_2022_aggregated.xlsx",
    header=0,
    index_col=0,
    engine="openpyxl",
    sheet_name='threat_20_years'
)

dict_regions_threat_20_years = {}
with open(
    "../../data/input/inputs_ABM/rice50_region_names_to_MetaSurvey.json"
) as rice50_region_names_to_MetaSurvey:
    dict_regions_cols = json.load(rice50_region_names_to_MetaSurvey)
    for key in dict_regions_cols.keys():
        if key != "_comment":
            dict_regions_threat_20_years[key] = df[
                dict_regions_cols[key]
            ].mean(axis=1)



# Temperature belief==> harm future generation : +0 --> +4    + Climate awareness for variance (from 0.01, i know a lot, to 4, I've never heard of it), don't know = +2 with 1*awareness variance
df = pd.read_excel(
    "../../data/input/inputs_ABM/climate_change_opinion_survey_2022_aggregated.xlsx",
    header=0,
    index_col=0,
    engine="openpyxl",
    sheet_name='harm_future_gen'
)

dict_regions_harm_future_gen = {}
with open(
    "../../data/input/inputs_ABM/rice50_region_names_to_MetaSurvey.json"
) as rice50_region_names_to_MetaSurvey:
    dict_regions_cols = json.load(rice50_region_names_to_MetaSurvey)
    for key in dict_regions_cols.keys():
        if key != "_comment":
            dict_regions_harm_future_gen[key] = df[
                dict_regions_cols[key]
            ].mean(axis=1)


# gov_priority==> For initial OPINION regarding 'Are we doing enough mitigation?"
df = pd.read_excel(
    "../../data/input/inputs_ABM/climate_change_opinion_survey_2022_aggregated.xlsx",
    header=0,
    index_col=0,
    engine="openpyxl",
    sheet_name='gov_priority'
)

dict_regions_gov_priority = {}
with open(
    "../../data/input/inputs_ABM/rice50_region_names_to_MetaSurvey.json"
) as rice50_region_names_to_MetaSurvey:
    dict_regions_cols = json.load(rice50_region_names_to_MetaSurvey)
    for key in dict_regions_cols.keys():
        if key != "_comment":
            dict_regions_gov_priority[key] = df[
                dict_regions_cols[key]
            ].mean(axis=1)

# most_responsible==> For initial OPINION regarding 'Am I willing to pay for more mitigation?"
df = pd.read_excel(
    "../../data/input/inputs_ABM/climate_change_opinion_survey_2022_aggregated.xlsx",
    header=0,
    index_col=0,
    engine="openpyxl",
    sheet_name='most_responsible'
)

dict_regions_most_responsible = {}
with open(
    "../../data/input/inputs_ABM/rice50_region_names_to_MetaSurvey.json"
) as rice50_region_names_to_MetaSurvey:
    dict_regions_cols = json.load(rice50_region_names_to_MetaSurvey)
    for key in dict_regions_cols.keys():
        if key != "_comment":
            dict_regions_most_responsible[key] = df[
                dict_regions_cols[key]
            ].mean(axis=1)