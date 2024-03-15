#!/bin/bash -l                                                                                                       

#SBATCH --job-name="CPU_Seq_DPS"                                                                                      
#SBATCH --time=120:00:00
#SBATCH --partition=compute-p2
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G  
                                                                                                                                                                          

#SBATCH --account=research-tpm-mas                                                                                   

module load 2023r1
module load openmpi
module load miniconda3


# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate justice-env
# Call python script with stat argument
srun python hpc_run.py
conda deactivate