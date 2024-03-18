# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:22:14 2024

@author: apoujon
"""
import numpy as np

class Negotiator:
    def __init__(self, region, policy_start_year = 2015, policy_end_year = 2100, nb_inner_policy_goals=25):
        self.region_model = region;
        self.policy = np.linspace([policy_start_year,0], [policy_end_year, 1], nb_inner_policy_goals+2).T;
        
        #Some parameters
        self.opinion_influence = 10;#How much the aggregated opinion influences the policy 
        self.max_cutting_rate_gradient = 0.04;#Maximum emission cutting rate gradient per year
        


    def international_netzero_proposal(self):
        return self.policy[0,-1]
    
    def shifting_policy(self, timestep):



        policy = self.policy;
        shift_year = self.region_model.aggregate_households_opinions();
        max_policy_rate = self.max_cutting_rate_gradient;
        current_time = timestep+self.region_model.twolevelsgame_model.justice_model.time_horizon.start_year;

        (self.region_model.twolevelsgame_model.f_policy)[1].writerow([self.region_model.id,
                                                                      current_time,
                                                                      self.policy.shape[1]]
                                                                     + [p for p in self.policy[1]]
                                                                     + [y for y in self.policy[0]])
        
        #Choosing a year to modify at random (The year must be in the future)
        p=policy[0,:-1]>current_time;
        if not p.any():\
            return policy
        year=self.region_model.twolevelsgame_model.justice_model.rng.choice(policy[0,:-1],1,p = p/sum(p))[0];
        year_ind = np.flatnonzero(year==policy[0,:])[0]
        min_p = max(0,np.flatnonzero(policy[0,:-1]>current_time)[0]-1);
        regional_pressure_later_ecr = 0;
        regional_pressure_earlier_ecr= 0;
        max_iter = 10;

        #TODO APN: get rid of next line (used for debugging)
        temp = policy.T
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
                    #print("too bad U:",shift_year," -> ",shift_year-1)
                    shift_year = shift_year-1
                    regional_pressure_later_ecr +=1;
                elif iter==max_iter:
                    regional_pressure_later_ecr +=1;
                    #print("Impossible to find a right configuration")
                    return policy
                else:
                    #print("well done")
                    policy = reshaped_policy;
                    reshape_successful = True

            if shift_year==0:
                return
                #print("Policy left unchanged")


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
                    #print("too bad Shift_year:",shift_year," -> ",shift_year+1)
                    shift_year = shift_year+1
                    regional_pressure_earlier_ecr +=1;
                elif iter==max_iter:
                    regional_pressure_earlier_ecr +=1;
                    #print("Impossible to find a right configuration")
                    return policy
                else:
                    #print("well done")
                    policy = reshaped_policy;
                    reshape_successful = True

            if shift_year==0:
                return
                #print("Policy left unchanged")

        self.policy = policy;  
        return 
