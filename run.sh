#!/bin/bash -l                                                                                                       

#SBATCH --job-name="Test10_95"                                                                                     
#SBATCH --time=10:00:00
#SBATCH --partition=compute

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0                                                                                               
                                                                                     

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load python
module load py-mpi4py


srun python hpc_run.py
