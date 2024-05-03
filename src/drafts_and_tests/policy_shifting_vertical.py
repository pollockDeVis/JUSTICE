# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:38:07 2024

@author: apoujon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as scint


start_year = 2000.
pol_at_start_year = 0;
end_year = 2300.
timestep_size = 1;




def delta_shift_range(policy, constrained_rate=0.04):
    return constrained_rate * (policy[0,1]-policy[0,0]) + policy[1,0] - policy[1,1]
    
def policy_shifting(policy, delta_target, constrained_rate=0.04):
    #Ensuring we are in the correct range of acceptable shift values
    print(delta_shift_range(policy, constrained_rate))
    if delta_shift_range(policy, constrained_rate) < delta_target:
        delta_target = delta_shift_range(policy, constrained_rate)
    elif delta_target+policy[1,1]-policy[1,0] < 0:
        delta_target = -policy[1,1]+policy[1,0]
        
    #Shifting policy target
    policy[1,1] =  min(delta_target+policy[1,1], 1)
    
    #Verifying new target compatibility with end goal
    gradient = (policy[1,2]-policy[1,1])/(policy[0,2]-policy[0,1])
    if gradient > constrained_rate:
        print("Pushing for later net zero at international negotiations!")
    
    
    return

def policy_step(policy, inner_target_period=5):
    if policy[0,1]+inner_target_period < policy[0,2]:
        #identify new target
        now = policy[:,1].copy()
        print(now)
        f = scint.interp1d(policy[0],
                           policy[1],
                           kind='linear')
        policy[:,1] = np.array([policy[0,1]+inner_target_period, f(policy[0,1]+inner_target_period)])
        policy[:,0] = now
        
        
        #get the shift
        delta_shift = -0.15
           
        
        #shift the target
        policy_shifting(policy, delta_target=delta_shift)
        
        #Updating current policy state
        print("-------------")
        
    
    


def keep_track(policy, ecr_projection, time_axes):
    f = scint.interp1d(policy[0],
                       policy[1],
                       kind='linear')
     
    ecr_projection = np.append(ecr_projection, f(np.linspace(policy[0,0], policy[0,1], int(policy[0,1]-policy[0,0]+1))))
    time_axes = np.append(time_axes,np.linspace(policy[0,0], policy[0,1], int(policy[0,1]-policy[0,0]+1)))
        
        

    
    
#In the simulation the policy consists: current target, next target, final target
#We only updat the next policy target every 5 years
ecr_projection = np.array([])
time_axes = np.array([])
policy=np.array([[start_year,start_year+5.,2100.],[0.1,0.15,1]]);

f = scint.interp1d(np.insert(np.append(policy[0], end_year), 0, start_year),
                   np.insert(np.append(policy[1], policy[1,-1]), 0, pol_at_start_year),
                   kind='linear')

ecr_projection_init = f(np.linspace(start_year, end_year, int(end_year-start_year+1)))
    
    
    
plt.plot(np.linspace(start_year, end_year, int(end_year-start_year+1)),ecr_projection_init,":")



for y in range(100):
    if y%5==0:
        print(policy)
        policy_step(policy, inner_target_period=5);
        f = scint.interp1d(policy[0],
                           policy[1],
                           kind='linear')
         
        ecr_projection = np.append(ecr_projection, f(np.linspace(policy[0,0], policy[0,1], int(policy[0,1]-policy[0,0]+1))))
        time_axes = np.append(time_axes,np.linspace(policy[0,0], policy[0,1], int(policy[0,1]-policy[0,0]+1)))
    
plt.plot(time_axes, ecr_projection)
plt.show()