# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:39:39 2024

@author: apoujon
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as scint


start_year = 2000.
pol_at_start_year = 0;
end_year = 2300.
timestep_size = 1;
policy=np.array([[2005,2010.,2060.,2100.],[0.1,0.5,0.9,0.2]]);


f = scint.interp1d(np.insert(np.append(policy[0], end_year), 0, start_year),
                   np.insert(np.append(policy[1], policy[1,-1]), 0, pol_at_start_year),
                   kind='linear')

ecr_projection = f(np.linspace(start_year, end_year, int(end_year-start_year+1)))
    
    
    
plt.plot(ecr_projection)
plt.show()


