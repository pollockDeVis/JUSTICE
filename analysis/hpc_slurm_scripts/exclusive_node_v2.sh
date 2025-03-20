#!/bin/bash -l
#SBATCH --job-name="Ex10kP"
#SBATCH --time=24:00:00
#SBATCH --partition=compute-p1

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks=1


#SBATCH --account=research-tpm-mas



# Set job parameters
nfe=10000       # Number of function evaluations
myswf=5         # Use swf value 4 only
myseed=1644652  # Fixed seed value

echo "Running task with:"
echo "nfe: $nfe"
echo "swf: $myswf"
echo "seed: $myseed"

apptainer exec justice_MultiProcessingEval.sif python3 hpc_run.py $nfe $myswf $myseed
