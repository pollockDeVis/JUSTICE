#!/bin/bash -l                                                                                                       

#SBATCH --job-name="MPI_DPS"                                                                                      
#SBATCH --time=24:00:00
#SBATCH --partition=compute-p2
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-3                                                                                                                                                                           
#SBATCH --account=research-tpm-mas                                                                                   

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip


# My nfe value [Range: 1 - inf]
nfe=10  

# My swf value [Range: 0 - 3]
swf=(0 1 2 3) 
myswf=${swf[$SLURM_ARRAY_TASK_ID]}

# Seed values
seeds=(9845531 1644652 3569126 6075612 521475)
if ((SLURM_ARRAY_TASK_ID < ${#seeds[@]})); then
    seed=${seeds[$SLURM_ARRAY_TASK_ID]}
else
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds the number of seeds (${#seeds[@]})."
    exit 1
fi

# Call python script with nfe_value argument
mpiexec -n 1 python3 hpc_run.py $nfe $myswf $seed