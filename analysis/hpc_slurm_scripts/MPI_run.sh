#!/bin/bash -l                                                                                                       

#SBATCH --job-name="MPI_DPS"                                                                                      
#SBATCH --time=120:00:00
#SBATCH --partition=compute-p2
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G                                                                                                                                                                           
#SBATCH --account=research-tpm-mas                                                                                   

module load 2023r1
module load openmpi
module load py-mpi4py
module load miniconda3


# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate justice-env
# Call python script with stat argument
mpiexec -n 1 python3 hpc_run.py
conda deactivate