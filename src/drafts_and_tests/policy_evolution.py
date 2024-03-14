# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:28:13 2024

@author: apoujon
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import time

"""
This script allows to modify an emission cutting rate (ecr) policy depending on the aggregated utility of constituency U.

It takes a policy defined as: 
    1) a starting year (fixed) for 0% ecr;
    2) a final year (fixed) for 100% ecr;
    3) an additionnal set of nb_inner_goals years and respective ecr targets
        -> For nb_inner_goals = 3, targets are 25%, 50% and 75% ecr respectively.

The utility U is considered translatable in a Shift_year:
    -> U>0 => Shift_year < 0, people are pushing for more climate policy, means the constituency would like to shift the policy to make it faster 
    -> U<0 => Shift_year > 0 people are pushing for less climate policy, means the constituency would like to shift the policy to make it slower

The Shift_year is a shift in number of years to be applied to one of the target years for 25%, 50% or 75% ecr goals. 
The starting and ending years are never affected. A current time (eg. current year) is also provided, such that any policy 
from the past can't be changed either.

The target on which to apply the shift is choosen randomly (uniform) amongst the three possibilities. 
After the shift is applied, the script checks for the gradient of ecr to ensure that it is within the authorized range (0-4%). 
If the condition is not respected, it tries modifying other targets (not the first and end years) to smoothen the curve.
Each time it tries a new combination it means the shift was too strong (Hence it might mean thart people wills are pushing for a different end year which will contribute to the constituency pressure during international negotiations).
The algorithm stops when a correct policy shape is obtained. This obtained policy shape might be the original policy unchanged if no possible configuration is found.


Another way to think about the international level is the carbon budget. Can come from IAM or can be similar to temperature expectations.
"""

### FUNCTION ###
def shifting_policy(policy, shift_year, max_policy_rate, current_time):    
    
    #Choosing a year to modify at random (The year must be in the future)
    p=policy[0,:-1]>current_time;
    if not p.any():
        print("Too far in the future, policy left unchanged")
        return policy
    year=np.random.choice(policy[0,:-1],1,p = p/sum(p))[0];
    year_ind = np.flatnonzero(year==policy[0,:])[0]
    min_p = max(0,np.flatnonzero(policy[0,:-1]>current_time)[0]-1);
    regional_pressure_later_ecr = 0;
    regional_pressure_earlier_ecr= 0;
    max_iter = 10;

    if shift_year > 0:
        reshape_successful = False
        while not reshape_successful and shift_year>0:
            #Reshape the policy curve until it fits with constrained emission rate gradient
            reshaped_policy = policy.copy()
            reshaped_policy[0,year_ind] = reshaped_policy[0,year_ind] + shift_year;
            grad = np.gradient(reshaped_policy[1,:], reshaped_policy[0,:])
            
            iter = 0
            while ((grad[:-1] > max_policy_rate).any() or (grad[:-1] < 0).any() or np.isnan(grad[:-1]).any()) and iter<max_iter:
                ind = np.flatnonzero((grad[:-1] > max_policy_rate) + (grad[:-1] < 0) + (np.isnan(grad[:-1])) )[0];
                reshaped_policy[0,ind+1] = 1/max_policy_rate * abs( reshaped_policy[1,ind]-reshaped_policy[1,ind+1])+reshaped_policy[0,ind];
                grad = np.gradient(reshaped_policy[1,:], reshaped_policy[0,:])
                iter += 1

            #If some points are projected after end year for policy, increase regional pressure for slower international negotiations
            #Retry with a smaller shift (Shift_year-1) if Shift_year-1 > 0
            if policy[0,-1] < reshaped_policy[0,-1] and iter<max_iter:
                # print("too bad U:",shift_year," -> ",shift_year-1)
                shift_year = shift_year-1
                regional_pressure_later_ecr +=1;
            elif iter==max_iter:
                regional_pressure_later_ecr +=1;
                # print("Impossible to find a right configuration")
                return policy
            else:
                # print("well done")
                policy = reshaped_policy;
                reshape_successful = True
                
        if shift_year==0:
            policy = policy
            # print("Policy left unchanged")
        
                
    else:
        reshape_successful = False
        while not reshape_successful and shift_year<0:
            #Reshape the policy curve until it fits with constrained emission rate gradient
            reshaped_policy = policy.copy()
            reshaped_policy[0,year_ind] = reshaped_policy[0,year_ind] + shift_year;
            grad = np.gradient(reshaped_policy[1,:], reshaped_policy[0,:])
            
            iter = 0
            while ((grad[:-1] > max_policy_rate).any() or (grad[:-1] < 0).any() or np.isnan(grad[:-1]).any()) and iter<max_iter:
                ind = np.flatnonzero((grad[:-1] > max_policy_rate) + (grad[:-1] < 0) + (np.isnan(grad[:-1])) )[0];
                # print("--------------------------------------")
                # print(grad)
                # print(reshaped_policy[0,ind])
                reshaped_policy[0,ind] = -1/max_policy_rate * abs( reshaped_policy[1,ind]-reshaped_policy[1,ind+1])+reshaped_policy[0,ind];
                grad = np.gradient(reshaped_policy[1,:], reshaped_policy[0,:])
                iter += 1
                # print(-1/max_policy_rate * ( reshaped_policy[1,ind+1]-reshaped_policy[1,ind]))
                # print(reshaped_policy[0])
                # time.sleep(3)
            
            #If some points are projected after end year for policy, increase regional pressure for slower international negotiations
            #Retry with a larger shift (Shift_year+1) if Shift_year+1 < 0
            if policy[0,min_p] > reshaped_policy[0,min_p] and iter<max_iter:
                # print("too bad Shift_year:",shift_year," -> ",shift_year+1)
                shift_year = shift_year+1
                regional_pressure_earlier_ecr +=1;
            elif iter==max_iter:
                regional_pressure_earlier_ecr +=1;
                # print("Impossible to find a right configuration")
                return policy
            else:
                # print("well done")
                policy = reshaped_policy;
                reshape_successful = True
        
        if shift_year==0:
            policy = policy
            # print("Policy left unchanged")
        
    return policy

def shifting_policy_spline_based(policy, shift_year, max_policy_rate, current_time):
    # Choosing a year to modify at random (The year must be in the future)
    p = policy[0, :-1] > current_time;
    if not p.any():
        print("Too far in the future, policy left unchanged")
        return policy
    year = np.random.choice(policy[0, :-1], 1, p=p / sum(p))[0];
    year_ind = np.flatnonzero(year == policy[0, :])[0]
    min_p = max(0, np.flatnonzero(policy[0, :-1] > current_time)[0] - 1);
    regional_pressure_later_ecr = 0;
    regional_pressure_earlier_ecr = 0;
    max_iter = 10;

    if shift_year > 0:
        reshape_successful = False
        while not reshape_successful and shift_year > 0:
            # Reshape the policy curve until it fits with constrained emission rate gradient
            reshaped_policy = policy.copy()
            reshaped_policy[0, year_ind] = reshaped_policy[0, year_ind] + shift_year;
            cx = sci.CubicSpline(reshaped_policy[0,:], reshaped_policy[1,:])
            dx = cx.derivative(reshaped_policy[0,:])


            iter = 0
            while ((grad[:-1] > max_policy_rate).any() or (grad[:-1] < 0).any() or np.isnan(
                    grad[:-1]).any()) and iter < max_iter:
                ind = np.flatnonzero((grad[:-1] > max_policy_rate) + (grad[:-1] < 0) + (np.isnan(grad[:-1])))[0];
                reshaped_policy[0, ind + 1] = 1 / max_policy_rate * abs(
                    reshaped_policy[1, ind] - reshaped_policy[1, ind + 1]) + reshaped_policy[0, ind];
                grad = np.gradient(reshaped_policy[1, :], reshaped_policy[0, :])
                iter += 1

            # If some points are projected after end year for policy, increase regional pressure for slower international negotiations
            # Retry with a smaller shift (Shift_year-1) if Shift_year-1 > 0
            if policy[0, -1] < reshaped_policy[0, -1] and iter < max_iter:
                # print("too bad U:",shift_year," -> ",shift_year-1)
                shift_year = shift_year - 1
                regional_pressure_later_ecr += 1;
            elif iter == max_iter:
                regional_pressure_later_ecr += 1;
                # print("Impossible to find a right configuration")
                return policy
            else:
                # print("well done")
                policy = reshaped_policy;
                reshape_successful = True

        if shift_year == 0:
            policy = policy
            # print("Policy left unchanged")


    else:
        reshape_successful = False
        while not reshape_successful and shift_year < 0:
            # Reshape the policy curve until it fits with constrained emission rate gradient
            reshaped_policy = policy.copy()
            reshaped_policy[0, year_ind] = reshaped_policy[0, year_ind] + shift_year;
            grad = np.gradient(reshaped_policy[1, :], reshaped_policy[0, :])

            iter = 0
            while ((grad[:-1] > max_policy_rate).any() or (grad[:-1] < 0).any() or np.isnan(
                    grad[:-1]).any()) and iter < max_iter:
                ind = np.flatnonzero((grad[:-1] > max_policy_rate) + (grad[:-1] < 0) + (np.isnan(grad[:-1])))[0];
                # print("--------------------------------------")
                # print(grad)
                # print(reshaped_policy[0,ind])
                reshaped_policy[0, ind] = -1 / max_policy_rate * abs(
                    reshaped_policy[1, ind] - reshaped_policy[1, ind + 1]) + reshaped_policy[0, ind];
                grad = np.gradient(reshaped_policy[1, :], reshaped_policy[0, :])
                iter += 1
                # print(-1/max_policy_rate * ( reshaped_policy[1,ind+1]-reshaped_policy[1,ind]))
                # print(reshaped_policy[0])
                # time.sleep(3)

            # If some points are projected after end year for policy, increase regional pressure for slower international negotiations
            # Retry with a larger shift (Shift_year+1) if Shift_year+1 < 0
            if policy[0, min_p] > reshaped_policy[0, min_p] and iter < max_iter:
                # print("too bad Shift_year:",shift_year," -> ",shift_year+1)
                shift_year = shift_year + 1
                regional_pressure_earlier_ecr += 1;
            elif iter == max_iter:
                regional_pressure_earlier_ecr += 1;
                # print("Impossible to find a right configuration")
                return policy
            else:
                # print("well done")
                policy = reshaped_policy;
                reshape_successful = True

        if shift_year == 0:
            policy = policy
            # print("Policy left unchanged")

    return policy


### SCRIPT ###
#Simulation parameters
current_time=1999.75;
max_policy_rate = 0.04

#Defining an initial Policy
nb_inner_goals = 20
start_year = 2000; #Defined as 2000
end_year = 2100 #Defined through international agreements
policy_initial = np.linspace([start_year,0], [end_year, 1], nb_inner_goals+2).T;
n_policy = len(policy_initial);

#Value of the aggregated opinion of people
shift_years =  np.arange(-10,10,1);

f1 = plt.figure()
plt.plot(policy_initial[0,:], policy_initial[1,:],'ok',markersize=3)

for s in shift_years:
    current_time = 2000
    policy = policy_initial.copy()
    for t in range(60):
        current_time += 1
        policy = shifting_policy(policy, s, max_policy_rate, current_time)
    

    print(policy[0,0]," to ", policy[0,-1])
    plt.plot(policy[0,:], policy[1,:],'o',markersize=3)
        
plt.legend(["Initial Policy"]+ ["Shifted Policy (shift = "+str(s) for s in shift_years],bbox_to_anchor=(1.1, 1.05))
plt.title("Effect of constituency pushing policy")
plt.show()
   
#Value of the aggregated opinion of people
shift_years =  np.linspace(0,10,1);

f2 = plt.figure()
plt.plot(policy_initial[0,:], policy_initial[1,:],'ok',markersize=3)

for s in shift_years:
    current_time = 2000
    policy = policy_initial.copy()
    for t in range(60):
        current_time += 1
        policy = shifting_policy(policy, s, max_policy_rate, current_time)
    

    print(policy[0,0]," to ", policy[0,-1])
    plt.plot(policy[0,:], policy[1,:],'o',markersize=3)
        
plt.legend(["Initial Policy"]+ ["Shifted Policy (shift = "+str(s) for s in shift_years],bbox_to_anchor=(1.1, 1.05))
plt.title("Effect of constituency pushing policy")
plt.show()