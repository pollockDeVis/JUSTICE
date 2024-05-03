# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:42:26 2024

@author: apoujon
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

distrib_resolution = 0.01;#In Celsius
distrib_x_axis = np.arange(-2,8,0.01)
time_max = 100
years = np.linspace(0,time_max,time_max)
temperature = np.linspace(1,2.5,time_max);
rng = np.random.default_rng(seed=None)


def gaussian_distrib(g_mean=0, g_std=1, min_val=-2, max_val=8, step=distrib_resolution):
    possible_values = np.arange(min_val,max_val,step);
    return np.exp(np.array([-1.*(x-g_mean)**2/(2.*(g_std**2)) for x in possible_values]));

def mean_distribution(distribution, min_val=-2, max_val=8, step=distrib_resolution):
    #Distribution is a matrix which rows are distributions
    return distribution @ np.arange(min_val,max_val,step);




        

#Agent initial belief
# 1 --- Parameters of the agent
class Agent():
    def __init__(self, initial_mean_beliefs =  np.array([1, 1.5, 1.7, 2.]), initial_var_beliefs =  np.array([0.01, 0.01,0.05,0.01])):
        self.belief_year_offset = np.array([0,10,50,99])
        self.initial_mean_beliefs =  initial_mean_beliefs;#Temperature now and in 10, 50, 100 years
        self.initial_var_beliefs =  initial_var_beliefs;
        self.nb_beliefs = len(self.initial_mean_beliefs)
        
        self.distrib_beliefs = []
        self.norm_coeff = 1
        self.distribution()
        
    def distribution(self):
        self.distrib_beliefs = np.array([gaussian_distrib(g_mean=self.initial_mean_beliefs[i], g_std=self.initial_var_beliefs[i]) for i
                                    in range(self.nb_beliefs)]); #Here range(nb_beliefs) depending on the number of beliefs to be modelled
        self.norm_coeff = np.sum(self.distrib_beliefs, axis=1);
        self.distrib_beliefs = np.array([self.distrib_beliefs[i,:] / self.norm_coeff[i] for i in range(self.nb_beliefs)]);
        
    def update_from_info(self, information):
        # 4 --- Weighting of information
        distrib_flfin = information.distrib_flsi; #No weighting

        # 5 --- Updating agent belief (sometimes)
        p_encounter_info = 0.5;
        gamma = 0.3;
        p = rng.random();
        if p < p_encounter_info:
            #Updating with info
            self.distrib_beliefs = self.distrib_beliefs *  distrib_flfin;
            
        else:
            #Imperfect memory
            #Get the mean
            mean_beliefs = mean_distribution(self.distrib_beliefs)
            distrib_beliefs_save = self.distrib_beliefs
            #Compute distribution based on initial std for agent
            self.distrib_beliefs = np.array([gaussian_distrib(g_mean=mean_beliefs[i], g_std=self.initial_var_beliefs[i]) for i in range(self.nb_beliefs)]);
            self.norm_coeff = np.sum(self.distrib_beliefs, axis=1);
            distrib_beliefs=np.array([self.distrib_beliefs[i,:] / self.norm_coeff[i] for i in range(self.nb_beliefs)]);
            #Merging of learned belief and belief with initial std
            distrib_beliefs = gamma * distrib_beliefs_save + (1-gamma)*self.distrib_beliefs
            
        self.norm_coeff = np.sum(self.distrib_beliefs, axis=1);
        self.distrib_beliefs=np.array([self.distrib_beliefs[i,:] / self.norm_coeff[i] for i in range(self.nb_beliefs)]);
        
        for i in range(self.nb_beliefs):
            print("Integral over distribution of belief ",i," : ",sum(self.distrib_beliefs[i]))
            
    def belief_to_projection(self):
        temperature_projection = np.array([]);
        year = self.belief_year_offset[0];
        temp0 = mean_distribution(self.distrib_beliefs[0])
        for i in range(len(self.distrib_beliefs)):
            temp = mean_distribution(self.distrib_beliefs[i])
            step = (temp-temp0)/(self.belief_year_offset[i]-year+1);
            if step != 0:
                temperature_projection = np.append(temperature_projection, np.arange(temp0, temp, step) );
            year = self.belief_year_offset[i]+1
            temp0 = temp
        
        return temperature_projection
    
    def belief_to_projection_uncertainty(self):
        temperature_projection = np.array([]);
        year = self.belief_year_offset[0];
        temp0 = mean_distribution(self.distrib_beliefs[0])
        for i in range(len(self.distrib_beliefs)):
            temp = mean_distribution(self.distrib_beliefs[i])
            step = (temp-temp0)/(self.belief_year_offset[i]-year+1);
            lin_proj = []
            if step != 0:
                lin_proj = np.arange(temp0, temp, step);
                projection = lin_proj - temp + rng.choice(np.arange(-2,8,0.01), len(lin_proj), p=self.distrib_beliefs[i])
            
            
            
                temperature_projection = np.append(temperature_projection, projection );
            year = self.belief_year_offset[i]+1
            temp0 = temp
        
        return temperature_projection
        
        

# 2 --- Compute belief distribution over all possible values
class Information():
    def __init__(self, nb_beliefs):
        self.nb_beliefs = nb_beliefs
        self.belief_year_offset = np.array([0,10,50,99])
        self.distrib_flsi = [0 for i in range(nb_beliefs)]
        for i in range(nb_beliefs):
            year = self.belief_year_offset[i]
            temperature_info = temperature[year]
            self.distrib_flsi[i] = gaussian_distrib(g_mean=temperature_info, g_std=0.01)
            self.norm_coeff = np.sum(self.distrib_flsi[i], axis=0);
            self.distrib_flsi[i]=self.distrib_flsi[i]/self.norm_coeff




agents_list = [Agent(initial_mean_beliefs =  np.array([1+2*rng.random(), 1.5, 1.7, 2.])) for i in range(10)]
nb_agents = len(agents_list)
plt.figure()
mean_distrib = 0*agents_list[0].distrib_beliefs[0]
for a in agents_list:
    plt.plot(distrib_x_axis,a.distrib_beliefs[0])
    mean_distrib += a.distrib_beliefs[0]/nb_agents 
    
plt.figure()
plt.plot(distrib_x_axis,mean_distrib)
print(sum(mean_distrib))

social_information = Information(4);
social_information.distrib_flsi[0] = mean_distrib
information = social_information
    
    
    
agent = Agent();
#information = Information(4);



for i in range(information.nb_beliefs):
    plt.figure()
    plt.plot(distrib_x_axis,agent.distrib_beliefs[i])
    print("Integral over distribution of belief ",i," : ",sum(agent.distrib_beliefs[i]))
    plt.plot(distrib_x_axis,information.distrib_flsi[i])
    plt.legend(["Initial Belief", "Initial Information"])
    plt.title("For "+str(information.belief_year_offset[i])+" years in the future")

plt.figure()
plt.plot(agent.belief_to_projection(),'ok',markersize=0.1, alpha=1)
plt.plot(agent.belief_to_projection_uncertainty(),'or',markersize=0.1, alpha=1)
plt.plot(temperature)
plt.legend(["Beliefs baseline", "Beliefs uncertainty", "Information"])
plt.title("Temperature projection based on beliefs (Before learning)")

agent.update_from_info(information)

for i in range(information.nb_beliefs):
    plt.figure()
    plt.plot(distrib_x_axis,agent.distrib_beliefs[i])
    print("Integral over distribution of belief ",i," : ",sum(agent.distrib_beliefs[i]))
    plt.plot(distrib_x_axis,information.distrib_flsi[i])
    plt.legend(["Final Belief", "Initial Information"])
    plt.title("For "+str(information.belief_year_offset[i])+" years in the future")


plt.figure()
plt.plot(agent.belief_to_projection(),'ok',markersize=0.1, alpha=1)
plt.plot(agent.belief_to_projection_uncertainty(),'or',markersize=0.1, alpha=1)
plt.plot(temperature)
plt.legend(["Beliefs baseline", "Beliefs uncertainty", "Information"])
plt.title("Temperature projection based on beliefs (After learning)")

plt.show()