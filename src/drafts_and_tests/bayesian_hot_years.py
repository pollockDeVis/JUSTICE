# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:42:26 2024

@author: apoujon
"""
import numpy as np
import matplotlib.pyplot as plt

distrib_resolution = 0.01;#In Celsius
distrib_x_axis = np.arange(-2,8,0.01)
time_max = 100
years = np.linspace(0,time_max,time_max)
temperature = np.linspace(1,2.5,time_max);


def gaussian_distrib(g_mean=0, g_std=1, min_val=-2, max_val=8, step=distrib_resolution):
    possible_values = np.arange(min_val,max_val,step);
    return np.exp(np.array([-1.*(x-g_mean)**2/(2.*(g_std**2)) for x in possible_values]));

def mean_distribution(distribution, min_val=-2, max_val=8, step=distrib_resolution):
    #Distribution is a matrix which rows are distributions
    return distribution @ np.arange(min_val,max_val,step);

def belief_to_projection(distrib_beliefs, belief_year_offset):
    temperature_projection = np.array([]);
    year = belief_year_offset[0];
    temp0 = mean_distribution(distrib_beliefs[0])
    for i in range(len(distrib_beliefs)):
        temp = mean_distribution(distrib_beliefs[i])
        step = (temp-temp0)/(belief_year_offset[i]-year+1);
        if step != 0:
            temperature_projection = np.append(temperature_projection, np.arange(temp0, temp, step) );
        year = belief_year_offset[i]+1
        temp0 = temp
    
    return temperature_projection

def belief_to_projection_uncertainty(distrib_beliefs, belief_year_offset):
    temperature_projection = np.array([]);
    year = belief_year_offset[0];
    temp0 = mean_distribution(distrib_beliefs[0])
    for i in range(len(distrib_beliefs)):
        temp = mean_distribution(distrib_beliefs[i])
        step = (temp-temp0)/(belief_year_offset[i]-year+1);
        lin_proj = []
        if step != 0:
            lin_proj = np.arange(temp0, temp, step);
            projection = lin_proj - temp + np.random.choice(np.arange(-2,8,0.01), len(lin_proj), p=distrib_beliefs[i])
        
        
        
            temperature_projection = np.append(temperature_projection, projection );
        year = belief_year_offset[i]+1
        temp0 = temp
    
    return temperature_projection
        

#Agent initial belief
# 1 --- Parameters of the agent
belief_year_offset = np.array([0,10,50,99])
initial_mean_beliefs =  np.array([1, 1.5, 1.7, 2.]);#Temperature now and in 10, 50, 100 years
initial_var_beliefs =  np.array([0.01, 0.05,0.15,0.3]);
nb_beliefs = len(initial_mean_beliefs)

# 2 --- Compute belief distribution over all possible values
distrib_beliefs = np.array([gaussian_distrib(g_mean=initial_mean_beliefs[i], g_std=initial_var_beliefs[i]) for i
                            in range(nb_beliefs)]); #Here range(nb_beliefs) depending on the number of beliefs to be modelled
norm_coeff = np.sum(distrib_beliefs, axis=1);
distrib_beliefs = np.array([distrib_beliefs[i,:] / norm_coeff[i] for i in range(nb_beliefs)]);

# 3 --- Defining information source
distrib_flsi = [[] for i in range(nb_beliefs)]
for i in range(nb_beliefs):
    year = belief_year_offset[i]
    temperature_info = temperature[year]
    distrib_flsi[i] = gaussian_distrib(g_mean=temperature_info, g_std=0.01)
    norm_coeff = np.sum(distrib_flsi[i], axis=0);
    distrib_flsi[i]=distrib_flsi[i]/norm_coeff

for i in range(nb_beliefs):
    plt.figure()
    plt.plot(distrib_x_axis,distrib_beliefs[i])
    print("Integral over distribution of belief ",i," : ",sum(distrib_beliefs[i]))
    plt.plot(distrib_x_axis,distrib_flsi[i])
    plt.legend(["Initial Belief", "Initial Information"])
    plt.title("For "+str(belief_year_offset[i])+" years in the future")

plt.figure()
plt.plot(belief_to_projection(distrib_beliefs, belief_year_offset),'o',markersize=0.1)
plt.plot(belief_to_projection_uncertainty(distrib_beliefs, belief_year_offset),'o',markersize=0.1)
plt.plot(temperature)
plt.legend(["Beliefs baseline", "Beliefs uncertainty", "Information"])
plt.title("Temperature projection based on beliefs (Before learning)")

# 4 --- Weighting of information
distrib_flfin = distrib_flsi; #No weighting

# 5 --- Updating agent belief (sometimes)
f1 = plt.figure()
for learning in range(10): #We are going to learn 10 times (here the info flfin doesn't change, but in JUSTICE it would/could!)
    p_encounter_info = 0.5;
    gamma = 0.3;
    p = np.random.rand();
    if p < p_encounter_info:
        #Updating with info
        distrib_beliefs = distrib_beliefs *  distrib_flfin;
        
    else:
        #Imperfect memory
        #Get the mean
        mean_beliefs = mean_distribution(distrib_beliefs)
        distrib_beliefs_save = distrib_beliefs
        #Compute distribution based on initial std for agent
        distrib_beliefs = np.array([gaussian_distrib(g_mean=mean_beliefs[i], g_std=initial_var_beliefs[i]) for i in range(nb_beliefs)]);
        norm_coeff = np.sum(distrib_beliefs, axis=1);
        distrib_beliefs=np.array([distrib_beliefs[i,:] / norm_coeff[i] for i in range(nb_beliefs)]);
        #Merging of learned belief and belief with initial std
        distrib_beliefs = gamma * distrib_beliefs_save + (1-gamma)*distrib_beliefs
        
    norm_coeff = np.sum(distrib_beliefs, axis=1);
    distrib_beliefs=np.array([distrib_beliefs[i,:] / norm_coeff[i] for i in range(nb_beliefs)]);
    
    for i in range(nb_beliefs):
        print("Integral over distribution of belief ",i," : ",sum(distrib_beliefs[i]))

    # SOME PLOTS
    plt.figure(f1)
    plt.plot(distrib_x_axis,distrib_beliefs[3])
    plt.plot(distrib_x_axis,distrib_flfin[3])
    plt.legend(["Belief", "Information"])
    plt.title("Evolution of belief for 100 years in the future")

plt.figure()
plt.plot(belief_to_projection(distrib_beliefs, belief_year_offset),'o',markersize=0.1)
plt.plot(belief_to_projection_uncertainty(distrib_beliefs, belief_year_offset),'o',markersize=0.1)
plt.plot(temperature)
plt.legend(["Beliefs baseline", "Beliefs uncertainty", "Information"])
plt.title("Temperature projection based on beliefs (After learning)")