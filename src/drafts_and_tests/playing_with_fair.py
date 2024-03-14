import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from fair import FAIR
from fair.interface import fill, initialise

#Instantiate FAIR
f = FAIR()

#Time horizon
f.define_time(0, 1000, 1)
f.timebounds

#Scenarios definition
f.define_scenarios(['abrupt-4xCO2'])

#Configuration definition (just reading some models names)
df = pd.read_csv("4xCO2_cummins_ebm3.csv")
models = df['model'].unique()
configs = []

for imodel, model in enumerate(models):
    for run in df.loc[df['model']==model, 'run']:
        configs.append(f"{model}_{run}")
f.define_configs(configs)

print(f.configs)

#Define species (only CO2)
species = ['CO2']
properties = {
    'CO2': {
        'type': 'co2',
        'input_mode': 'forcing',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False,
    },
}
f.define_species(species, properties)

#Run options (not used here)

#Create data structures
f.allocate()

#Filling datastructures
initialise(f.temperature, 0)

df = pd.read_csv("4xCO2_cummins_ebm3.csv")
models = df['model'].unique()

seed = 0

for config in configs:
    model, run = config.split('_')
    condition = (df['model']==model) & (df['run']==run)
    fill(f.climate_configs['ocean_heat_capacity'], df.loc[condition, 'C1':'C3'].values.squeeze(), config=config)
    fill(f.climate_configs['ocean_heat_transfer'], df.loc[condition, 'kappa1':'kappa3'].values.squeeze(), config=config)
    fill(f.climate_configs['deep_ocean_efficacy'], df.loc[condition, 'epsilon'].values[0], config=config)
    fill(f.climate_configs['gamma_autocorrelation'], df.loc[condition, 'gamma'].values[0], config=config)
    fill(f.climate_configs['sigma_eta'], df.loc[condition, 'sigma_eta'].values[0], config=config)
    fill(f.climate_configs['sigma_xi'], df.loc[condition, 'sigma_xi'].values[0], config=config)
    fill(f.climate_configs['stochastic_run'], True, config=config)
    fill(f.climate_configs['use_seed'], True, config=config)
    fill(f.climate_configs['seed'], seed, config=config)

    # We want to fill in a constant 4xCO2 forcing (for each model) across the run.
    fill(f.forcing, df.loc[condition, 'F_4xCO2'].values[0], config=config, specie='CO2')

    seed = seed + 10101
    
df

f.fill_species_configs()

fill(f.species_configs['tropospheric_adjustment'], 0, specie='CO2')

#Running Fair
f.run()

#Results
fig, ax = pl.subplots()
ax.plot(f.timebounds, f.temperature.loc[dict(layer=0, scenario='abrupt-4xCO2')]);
ax.set_xlim(0, 1000)
ax.set_ylim(0, 13)
ax.set_ylabel('Global mean warming above pre-industrial, °C')
ax.set_xlabel('Year')
ax.set_title('CMIP6 abrupt-4xCO$_2$ emulations, FaIR v2.1')
fig.tight_layout()

pl.plot(f.timebounds, f.toa_imbalance.loc[dict(scenario='abrupt-4xCO2')]);

pl.plot(f.timebounds, f.forcing_sum.loc[dict(scenario='abrupt-4xCO2')]);

pl.plot(f.timebounds[800:], f.toa_imbalance.loc[dict(scenario='abrupt-4xCO2')][800:,...])
pl.axhline(0, color='k')

fig, ax = pl.subplots(11, 6, figsize=(16, 30))

for i, config in enumerate(configs):
    ax[i//6,i%6].scatter(f.temperature.loc[dict(layer=0, scenario='abrupt-4xCO2', config=config)], f.toa_imbalance.loc[dict(scenario='abrupt-4xCO2', config=config)])
    ax[i//6,i%6].set_xlim(0,13)
    ax[i//6,i%6].set_ylim(-1, 10)
    ax[i//6,i%6].axhline(0, color='k')
    ax[i//6,i%6].set_title(config, fontsize=6)