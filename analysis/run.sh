#!/bin/bash -l                                                                                                       

#SBATCH --job-name="MEAN_RUN_1000"                                                                                     
#SBATCH --time=48:00:00
#SBATCH --partition=compute


#SBATCH --nodes=1                                                                                                    
#SBATCH --exclusive                                                                                                  
#SBATCH --mem=0     

                                                                                                                                                                          

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load python
module load py-mpi4py


srun python hpc_run.py
