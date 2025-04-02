#!/bin/bash -l

#SBATCH --job-name="50kU2"
#SBATCH --partition=memory
#SBATCH --time=30:00:00

#SBATCH --exclusive
                                                                                              
#SBATCH --mem=0     

#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1



#SBATCH --account=research-tpm-mas



# Set job parameters
nfe=50000       # Number of function evaluations
myswf=4         # Use swf value 4 only
myseed=9845531  # Fixed seed value

echo "Running task with:"
echo "nfe: $nfe"
echo "swf: $myswf"
echo "seed: $myseed"

apptainer exec justice_MP.sif python3 hpc_run.py $nfe $myswf $myseed
