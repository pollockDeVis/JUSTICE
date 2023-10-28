#!/bin/bash -l                                                                                                       

#SBATCH --job-name="RUN_10000_MEAN"                                                                                     
#SBATCH --time=48:00:00
#SBATCH --partition=compute

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0                                                                                               
                                                                                     

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load python
module load py-mpi4py


srun python hpc_run.py
