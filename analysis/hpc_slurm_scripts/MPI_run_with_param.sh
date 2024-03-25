#!/bin/bash -l                                                                                                       

#SBATCH --job-name="MPI_DPS"                                                                                      
#SBATCH --time=60:00:00
#SBATCH --partition=compute-p2
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G                                                                                                                                                                           
#SBATCH --account=research-tpm-mas                                                                                   

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip

pip install --user ema_workbench


# My nfe value
nfe=1000  

# Call python script with nfe_value argument
mpiexec -n 1 python3 hpc_run.py $nfe