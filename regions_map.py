import pandas as pd
import pycountry
import plotly.express as px
import plotly.io as pio
import json
import os
from src.util.data_loader import DataLoader


def load_region_iso_mapping(json_path):
    """
    Load the region to ISO code mapping from a JSON file.
    """
    with open(json_path, 'r') as f:
        region_iso_mapping = json.load(f)
    #print("Region ISO Mapping from JSON:", region_iso_mapping.keys())  # Debugging: Print region keys
    return region_iso_mapping

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
            print(f"Region '{region}' not found in JSON mapping.")  # Debugging: Print unmatched regions

    region_country_df = pd.DataFrame(region_country_list, columns=['Region', 'CountryCode'])
    return region_country_df

def plot_region_map(region_country_df, output_dir):
    """
    Plot a map with regions identified by different colors using Plotly.
    """
    # Add country names to the dataframe
    region_country_df["CountryName"] = region_country_df["CountryCode"].apply(
        lambda x: pycountry.countries.get(alpha_3=x).name if pycountry.countries.get(alpha_3=x) else None
    )

    # Debugging: Print the first few rows of the dataframe to check its structure
    #print(region_country_df.head()
    
    # Define a custom color scale with 57 different colors
    custom_color_scale = ['#A06077', '#B37788', '#8F4B66', '#C68799',
     '#9B5C74', '#D9AF6B', '#C7A060', '#EBC78E', '#BF994F',
     '#F2D1A0', '#735F4C', '#8F735B', '#5D4839', '#A3856D',
     '#4C3B2E', '#68855C', '#7A9C6F', '#556B49', '#8CAE82',
     '#3E4D31', '#625377', '#74618A', '#4F4460', '#85729C',
     '#3B3249', '#B18458', '#C49C6C', '#926E47', '#D6AF7E',
     '#704F33', '#B3B35E', '#C8C86D', '#99994B', '#DAD97C',
     '#80803D', '#7A6885', '#8F7998', '#5E5066', '#A28BAA',
     '#483E4E', '#526A83', '#637A94', '#3E5066', '#748BA5',
     '#2A3647', '#7C7C7C', '#8C8C8C', '#6C6C6C', '#9C9C9C', '#5C5C5C'
    ]
    # Plotting the regions with different colors
    fig = px.choropleth(
        region_country_df,
        locations="CountryCode",
        color="Region",
        hover_name="CountryName",
        color_discrete_sequence=custom_color_scale
    )

    fig.update_layout(
        title_text='Regions Identified by Different Colors',
        title_x=0.5,
        geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular')
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        fig.write_image(os.path.join(output_dir, 'regions_identified_map.png'))
    except ValueError as e:
        print("Image export failed, please install 'kaleido' package to enable this feature.")
        print(e)
    
    pio.renderers.default = 'browser'
    fig.show()


# Example usage:
region_iso_path = 'data/input/rice50_regions_dict.json'
output_dir = 'figures/maps'
data_loader = DataLoader()
region_list = data_loader.REGION_LIST
region_iso_mapping = load_region_iso_mapping(region_iso_path)
region_country_df = create_region_country_dataframe(region_list, region_iso_mapping)
plot_region_map(region_country_df, output_dir)
