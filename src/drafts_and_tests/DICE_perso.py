# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:18:02 2024

@author: apoujon
"""
import numpy as np
import matplotlib.pyplot as plt

def DICE():
    def __init__(self):
        self.t = 2015;
        self.step = 5;
        self.t_max = 2315;
        
        #preferences
        self.elasticity_marginal_utility_cons = 1.45;
        self.pure_rate_social_time_preference = 0.015;
        
        self.delta_L = 0;
        self.delta_A = 0.006;
        self.delta_sigma = -0.001;
        self.delta_K = 0;
        
        self.theta_1 = 0;
        self.theta_2 = 0;
        self.emissions_reduction_rate = 0;
        
        #population and technology
        self.gamma = 0.300;#Capital elasticity in production 
        
        #External
        self.E_land = lambda t: 3.3*(1-0.2)**(t-1);
        
        
        self.g_L = 0.134;
        self.g_A = 0.079
        self.g_sigma = -0.01;
        self.L = 0;
        self.A = 0
        self.sigma = 0;
        
        self.abatement = 0;
        self.damages = 0;
        self.K = 0;
        self.I = 0;
        self.C = 0;
        self.C_per_capita = 0;
        
        self.E_ind  = 0;
        
        self.E = 0;
        
        phi11=0
        phi21=0
        phi12=0
        phi22=0
        phi32=0
        phi23=0
        phi33=0
        self.PHI = np.array([[phi11, phi21, 0],\
                             [phi12, phi22, phi32],\
                             [0, phi23, phi33]])
        m_at=0
        self.m_at_1750
        m_lo=0
        m_up=0
        self.M = np.matix([m_at,m_up,m_lo]).T; 
        self.F = 0;
        self.eta = 0;
        self.F_ex = 0;#Other forcings, exogeneous
        
        f=0
        t_at = 0
        t_lo = 0
        self.F = np.matix([f,t_at,t_lo]).T;
        xi1=0
        xi2=0
        xi3=0
        xi4=0
        self.XI = np.array([[0, 0, 0],\
                             [xi1, 1-xi1(xi2-xi3), xi1*xi3],\
                             [0, xi4, 1-xi1]])
            
        
        
        
        
        
        
    def step(self):
        #Economy
        self.g_L = self.g_L / (1 + self.delta_L);
        self.L = self.L * (1 + self.g_L);
        
        self.g_A = self.g_A / (1 + self.delta_A);
        self.A = self.A * [1 + self.g_A]
        
        self.abatement = self.theta_1 * self.emissions_reduction_rate ** self.theta_2;
        self.I
        self.K = self.I - self.delta_K * self.K;
        
        self.Q = (1 - self.abatement)*self.A*self.K**self.gamma*self.L**(1-self.gamma)*1/(1+self.damages);
        self.I = self.Q - self.C
        self.C_per_capita = self.C / self.L
        
        #Emissions
        self.g_sigma = self.g_sigma / (1 + self.delta_sigma)
        self.sigma = self.sigma * (1 + self.g_sigma)
        self.E_ind = self.sigma * (1 - self.emissions_reduction_rate) * self.A * self.K**self.gamma * self.L**(1-self.gamma)
        
        
        self.E = self.E_ind + self.E_land
        
        #Carbon cycle
        self.M = self.PHI @ self.M +  np.matrix([self.E, 0, 0]).T;
        self.F = self.XI @ self.F + np.matrix([self.eta * np.log2(self.M[0]/self.m_at_1750) + self.F_ex, 0 ,0]).T;
        
    def run(self):
        while self.t < self.t_max:
            self.step();
            self.W = self.U * self.R;
            self.t += self.step;
    
    
        