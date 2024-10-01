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

def collect_variable_data(input_dir, variable_names):
    """
    Collect data for specified variables from all scenarios.
    """
    scenarios_data = {var: {} for var in variable_names}
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npz'):
            npz_file_path = os.path.join(input_dir, file_name)
            scenario_name = os.path.splitext(file_name)[0]
            data = load_npz_file(npz_file_path)
            for var in variable_names:
                scenarios_data[var][scenario_name] = data[var]
    return scenarios_data

def create_country_data(region_mapping, region_data):
    """
    Create country data from region data using the provided region mapping.
    """
    country_data = {}
    for region, countries in region_mapping.items():
        for country in countries:
            country_data[country] = region_data[region]
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
    return region_country_df

def sanitize_title(title):
    """
    Sanitize the title to create a valid file name.
    """
    return title.replace(" ", "_").replace("/", "_")

def plot_choropleth(region_country_df, variable_name, title, output_dir, country_data):
    """
    Plot a choropleth map for a specific variable.
    """
    # Add country names to the dataframe
    region_country_df["CountryName"] = region_country_df["CountryCode"].apply(
        lambda x: pycountry.countries.get(alpha_3=x).name if pycountry.countries.get(alpha_3=x) else None
    )
    
    # Add the variable data to the dataframe
    region_country_df["Value"] = region_country_df["CountryCode"].map(country_data)

    # Plotting the regions with different colors
    fig = px.choropleth(
        region_country_df,
        locations="CountryCode",
        color="Value",
        hover_name="CountryName",
        color_continuous_scale="amp",
    )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular')
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        sanitized_title = sanitize_title(variable_name)
        fig.write_image(os.path.join(output_dir, f'{sanitized_title}_choropleth.png'))
    except ValueError as e:
        print("Image export failed, please install 'kaleido' package to enable this feature.")
        print(e)
    
    pio.renderers.default = 'browser'
    fig.show()

def process_choropleth_ratios(input_dir, output_dir, region_list, region_mapping_path):
    """
    Process all .npz files for the selected case (BAU or CE) and generate choropleth maps for ratios across scenarios and regions.
    """
    variables_of_interest = [
        'emissions', 'emissions_avoided', 'recycling_cost', 'net_economic_output'
    ]
    titles = [
        'Emissions Avoided / Emissions in 2100', 'Recycling Cost / Net Economic Output in 2100'
    ]
    
    scenarios = [
     'SSP119', 'SSP245', 'SSP370', 'SSP460', 'SSP585'
    ]
    year_index = 2100 - 2015  # Calculate the index for the year 2100

    region_iso_mapping = load_region_iso_mapping(region_mapping_path)

    data = collect_variable_data(input_dir, variables_of_interest)
    
    for title in titles:
        for scenario in scenarios:
            #print(f"Processing scenario {scenario} for ratio {title}.")
            
            if 'Emissions Avoided / Emissions' in title:
                emissions = data['emissions'][scenario][:, year_index, :]
                emissions_avoided = data['emissions_avoided'][scenario][:, year_index, :]
                ratio = np.mean(emissions_avoided / emissions, axis=1)
            elif 'Recycling Cost / Net Economic Output' in title:
                recycling_cost = data['recycling_cost'][scenario][:, year_index, :]
                net_economic_output = data['net_economic_output'][scenario][:, year_index, :]
                ratio = np.mean(recycling_cost / net_economic_output, axis=1)
            
            df = pd.DataFrame(ratio, index=region_list)
            mean_df = df.mean(axis=1)
            
            country_data = create_country_data(region_iso_mapping, mean_df)
            
            region_country_df = create_region_country_dataframe(region_list, region_iso_mapping)
            
            plot_choropleth(region_country_df, title, f'{title} - {scenario}', output_dir, country_data)

def normalize_data(data):
    """
    Normalize the data to a range of [0, 1].
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

def process_choropleth_ratios_normalized(input_dir, output_dir, region_list, region_mapping_path):
    """
    Process all .npz files for the selected case (BAU or CE) and generate normalized choropleth maps for ratios across scenarios and regions.
    """
    variables_of_interest = [
        'emissions', 'emissions_avoided', 'recycling_cost', 'net_economic_output'
    ]
    titles = [
        'Emissions Avoided / Emissions in 2100', 'Recycling Cost / Net Economic Output in 2100'
    ]
    
    scenarios = [
     'SSP119', 'SSP245', 'SSP370', 'SSP460', 'SSP585'
    ]
    year_index = 2100 - 2015  # Calculate the index for the year 2100

    region_iso_mapping = load_region_iso_mapping(region_mapping_path)

    data = collect_variable_data(input_dir, variables_of_interest)
    
    for title in titles:
        all_ratios = []
        for scenario in scenarios:
            print(f"Processing scenario {scenario} for ratio {title}.")
            
            if 'Emissions Avoided / Emissions' in title:
                emissions = data['emissions'][scenario][:, year_index, :]
                emissions_avoided = data['emissions_avoided'][scenario][:, year_index, :]
                ratio = np.mean(emissions_avoided / emissions, axis=1)
            elif 'Recycling Cost / Net Economic Output' in title:
                recycling_cost = data['recycling_cost'][scenario][:, year_index, :]
                net_economic_output = data['net_economic_output'][scenario][:, year_index, :]
                ratio = np.mean(recycling_cost / net_economic_output, axis=1)
            
            all_ratios.append(ratio)
        
        # Normalize all ratios together
        all_ratios = np.concatenate(all_ratios)
        normalized_ratios = normalize_data(all_ratios)
        
        start = 0
        for scenario in scenarios:
            print(f"Plotting scenario {scenario} for normalized ratio {title}.")
            
            # Extract the normalized ratio for the current scenario
            end = start + len(region_list)
            ratio = normalized_ratios[start:end]
            start = end
            
            df = pd.DataFrame(ratio, index=region_list)
            mean_df = df.mean(axis=1)
            
            country_data = create_country_data(region_iso_mapping, mean_df)
            
            region_country_df = create_region_country_dataframe(region_list, region_iso_mapping)
            
            plot_choropleth(region_country_df, title, f'Normalized {title} - {scenario}', output_dir, country_data)

# Example usage:
data_loader = DataLoader()
region_list = data_loader.REGION_LIST
region_mapping_path = 'data/input/rice50_regions_dict.json'
input_dir = 'data/output/ce_depletion'  # Choose either 'data/output/ce_depletion' or 'data/output/bau_depletion'
output_dir = 'figures/choropleths/ce'  # Adjust directory as needed
#process_choropleth_ratios(input_dir, output_dir, region_list, region_mapping_path)
process_choropleth_ratios_normalized(input_dir, output_dir, region_list, region_mapping_path)
