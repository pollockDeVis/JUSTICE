#!/bin/bash -l

#SBATCH --job-name="50k"
#SBATCH --partition=memory
#SBATCH --time=30:00:00

#SBATCH --exclusive
#SBATCH --mem=0     

#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1

#SBATCH --account=research-tpm-mas
#SBATCH --array=0-9

# Set job parameters
nfe=50000    # Number of function evaluations

# Define swf and seed arrays
swf=(0 1)
seeds=(9845531 1644652 3569126 6075612 521475)

# Determine array sizes
num_swf=${#swf[@]}
num_seeds=${#seeds[@]}

# Calculate indices for swf and seed based on SLURM_ARRAY_TASK_ID
swf_index=$(( SLURM_ARRAY_TASK_ID / num_seeds ))
seed_index=$(( SLURM_ARRAY_TASK_ID % num_seeds ))

# Ensure that the calculated index for swf is within bounds
if (( swf_index < num_swf )) && (( seed_index < num_seeds )); then
    myswf=${swf[$swf_index]}
    myseed=${seeds[$seed_index]}
else
    echo "Error: Calculated index exceeds array bounds. swf_index: $swf_index, seed_index: $seed_index."
    exit 1
fi

# Display task information for debugging
echo "Running task with:"
echo "nfe: $nfe"
echo "swf: $myswf (swf index: $swf_index)"
echo "seed: $myseed (seed index: $seed_index)"

# Launch the job via the container
apptainer exec justice_MP.sif python3 hpc_run.py $nfe $myswf $myseed