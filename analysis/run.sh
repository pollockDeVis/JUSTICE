#!/bin/bash -l                                                                                                       

#SBATCH --job-name="20k_%a"                                                                                      
#SBATCH --time=48:00:00
#SBATCH --partition=compute


#SBATCH --nodes=1                                                                                                    
#SBATCH --exclusive                                                                                                  
#SBATCH --mem=0     
#SBATCH --array=1-4    
                                                                                                                                                                          

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load python
module load py-mpi4py


# Array to specify stat value
stats=( "mean" "median" "95th" "5th" )

# Determine which stat to use
stat=${stats[$(( $SLURM_ARRAY_TASK_ID - 1 ))]}

# Call python script with stat argument
srun python hpc_run.py $stat

