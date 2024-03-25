#!/bin/bash -l                                                                                                       

#SBATCH --job-name="100k_DPS"                                                                                      
#SBATCH --time=120:00:00
#SBATCH --partition=compute-p2


#SBATCH --nodes=1                                                                                                    
#SBATCH --exclusive                                                                                                  
#SBATCH --mem=0     
                 

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