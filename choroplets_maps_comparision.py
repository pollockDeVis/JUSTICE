import numpy as np
import os
import pandas as pd
import json
import pycountry
import plotly.express as px
import plotly.io as pio
from src.util.data_loader import DataLoader

def load_npz_file(npz_file_path):
    """
    Load a .npz file and return the relevant variables.
    """
    data = np.load(npz_file_path, allow_pickle=True)
    #print(f"Loaded data from {npz_file_path}. Variables: {data.files}")
    return data

def load_region_iso_mapping(json_path):
    """
    Load the region to ISO code mapping from a JSON file.
    """
    with open(json_path, 'r') as f:
        region_iso_mapping = json.load(f)
    #print(f"Loaded region ISO mapping from {json_path}. Regions: {list(region_iso_mapping.keys())}")
    return region_iso_mapping

def collect_variable_data(input_dir, variable_name):
    """
    Collect data for a specific variable from all scenarios.
    """
    scenarios_data = {}
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npz'):
            npz_file_path = os.path.join(input_dir, file_name)
            scenario_name = os.path.splitext(file_name)[0]
            data = load_npz_file(npz_file_path)
            scenarios_data[scenario_name] = data[variable_name]
    #print(f"Collected data for variable '{variable_name}' from {input_dir}. Scenarios: {list(scenarios_data.keys())}")
    return scenarios_data

def calculate_percentage_change(data_bau, data_ce):
    """
    Calculate the percentage change between BAU and CE data.
    """
    return (data_ce - data_bau) / data_bau * 100

def create_country_data(region_mapping, region_data):
    """
    Create country data from region data using the provided region mapping.
    """
    country_data = {}
    for region, countries in region_mapping.items():
        for country in countries:
            country_data[country] = region_data[region]
    #print(f"Created country data from region data. Countries: {list(country_data.keys())[:5]}...")  # Print first 5 countries for brevity
    return country_data

def create_region_country_dataframe(region_list, region_iso_mapping):
    """
    Create a dataframe with regions and their corresponding countries.
    """
    region_country_list = []

    for region in region_list:
        if region in region_iso_mapping:
            countries = region_iso_mapping[region]
            for country in countries:
                region_country_list.append({'Region': region, 'CountryCode': country})
        else:
            print(f"Region '{region}' not found in JSON mapping.")

    region_country_df = pd.DataFrame(region_country_list, columns=['Region', 'CountryCode'])
    #print(f"Created region-country dataframe with {len(region_country_df)} entries.")
    return region_country_df

def plot_choropleth(region_country_df, variable_name, title, output_dir, country_data):
    """
    Plot a choropleth map for percentage change of a specific variable.
    """
    # Add country names to the dataframe
    region_country_df["CountryName"] = region_country_df["CountryCode"].apply(
        lambda x: pycountry.countries.get(alpha_3=x).name if pycountry.countries.get(alpha_3=x) else None
    )
    
    # Add percentage change to the dataframe
    region_country_df["Percentage Change"] = region_country_df["CountryCode"].map(country_data)
    
    # Plotting the regions with different colors
    fig = px.choropleth(
        region_country_df,
        locations="CountryCode",
        color="Percentage Change",
        hover_name="CountryName",
        color_continuous_scale="Brwnyl",
    )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular')
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        fig.write_image(os.path.join(output_dir, f'{variable_name}_choropleth.png'))
    except ValueError as e:
        print("Image export failed, please install 'kaleido' package to enable this feature.")
        print(e)
    
    pio.renderers.default = 'browser'
    fig.show()

def process_choropleth_variables(input_dir_bau, input_dir_ce, output_dir, region_list, region_mapping_path):
    """
    Process all .npz files for both BAU and CE cases and generate choropleth maps for variables across scenarios and regions.
    """
    variables_of_interest = [
    #  'regional_temperature',
    #  'emissions',
    # 'net_economic_output',
    #  'recycling_cost',    
       #'emissions_avoided',    
    #  'waste',
    #  'depletion_ratio',
    #  'welfare_regional_temporal',
    #  'disentangled_utility',
    #  'consumption_per_capita'
    ]
    titles = [
    #  'Temperature Change (%) 2100',
    #  'Emissions Change (%) 2100',
    #   'Net Output Change (%) 2100',
    #  'Recycling Cost Change (%) 2100',    
       #'Emissions Avoided Change (%) 2100',
    #  'Waste Change (%) 2100',
    #  'Depletion Ratio Change (%) 2100',
      'Welfare Change (%) 2100',
    #  'Disentangled Utility Change (%) 2100',
    #  'Consumption per capita Change (%) 2100'
    ]

    scenarios = [
     'SSP119',
     #'SSP126',
     'SSP245',
     'SSP370',
     #'SSP434',
     'SSP460',
     #'SSP534',
     'SSP585'
    ]
    year_index = 2100 - 2015  # Calculate the index for the year 2050

    region_iso_mapping = load_region_iso_mapping(region_mapping_path)
    #region_list = region_list

    for variable_name, title in zip(variables_of_interest, titles):
        data_bau = collect_variable_data(input_dir_bau, variable_name)
        data_ce = collect_variable_data(input_dir_ce, variable_name)
        
        # Extract data for the year 2050 (index 35)
        bau_2050 = {scenario: data[:, year_index, :] for scenario, data in data_bau.items()}
        ce_2050 = {scenario: data[:, year_index, :] for scenario, data in data_ce.items()}
        
        # Calculate the mean across ensembles for each scenario and region at year 2050
        bau_mean_2050 = {scenario: np.mean(data, axis=1) for scenario, data in bau_2050.items()}
        ce_mean_2050 = {scenario: np.mean(data, axis=1) for scenario, data in ce_2050.items()}
        
        for scenario in scenarios:
            # print(f"Processing scenario {scenario} for variable {variable_name}.")
            # Create DataFrame for the scenario
            bau_df = pd.DataFrame(bau_mean_2050[scenario], index=region_list)
            ce_df = pd.DataFrame(ce_mean_2050[scenario], index=region_list)
            
            # Calculate percentage change
            percentage_change_df = calculate_percentage_change(bau_df, ce_df).mean(axis=1)
            # print(f"Percentage change for {scenario}:")
            # print(percentage_change_df.head())
            
            # Aggregate to countries
            country_data = create_country_data(region_iso_mapping, percentage_change_df)
            # print(f"Aggregated percentage change to countries for {scenario}:")
            # print({k: country_data[k] for k in list(country_data)[:5]})  # Print first 5 for brevity
            
            # Create region-country DataFrame
            region_country_df = create_region_country_dataframe(region_list, region_iso_mapping)
            
            # Plot choropleth map
            plot_choropleth(region_country_df, variable_name, f'{title} - {scenario}', output_dir, country_data)


# Example usage:
input_dir_bau = 'data/output/bau_depletion'
input_dir_ce = 'data/output/ce_depletion'
output_dir = 'figures/choropleths/2050/bau_vs_ce'
data_loader = DataLoader()
region_list= data_loader.REGION_LIST
region_mapping_path = 'data/input/rice50_regions_dict.json'
process_choropleth_variables(input_dir_bau, input_dir_ce, output_dir, region_list, region_mapping_path)
