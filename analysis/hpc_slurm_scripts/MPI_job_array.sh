#!/bin/bash -l                                                                                                       

#SBATCH --job-name="MPI_DPS"                                                                                      
#SBATCH --time=24:00:00
#SBATCH --partition=compute-p2
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-19                                                                                                                                                                           
#SBATCH --account=research-tpm-mas                                                                                   

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip


# My nfe value [Range: 1 - inf]
nfe=100000  

# My swf value [Range: 0 - 3]
swf=(0 1 2 3) 
myswf=${swf[$SLURM_ARRAY_TASK_ID]}

# Seed values
seeds=(9845531 1644652 3569126 6075612 521475)

# Total number of swf values and seeds
num_swf=${#swf[@]}
num_seeds=${#seeds[@]}

# Calculate swf and seed indices
swf_index=$(( SLURM_ARRAY_TASK_ID / num_seeds ))
seed_index=$(( SLURM_ARRAY_TASK_ID % num_seeds ))

# Ensure indices are within bounds
if (( swf_index < num_swf )) && (( seed_index < num_seeds )) ; then
    myswf=${swf[$swf_index]}
    myseed=${seeds[$seed_index]}
else
    echo "Error: Calculated index exceeds array bounds. swf_index: $swf_index, seed_index: $seed_index."
    exit 1
fi

# Display task information for debugging
echo "Running task with:"
echo "nfe: $nfe"
echo "swf: $myswf (index $swf_index)"
echo "seed: $myseed (index $seed_index)"

# Call python script with nfe_value argument
mpiexec -n 1 python3 hpc_run.py $nfe $myswf $myseed