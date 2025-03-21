#!/bin/bash -l
#SBATCH --job-name="Ut2_50k"
#SBATCH --time=100:00:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=research-tpm-mas

export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer/cache


nfe=50000       # Number of function evaluations
myswf=5         # Use swf value 4 only
myseed=1644652  # Fixed seed value


echo "Running task with:"
echo "nfe: $nfe"
echo "swf: $myswf"
echo "seed: $myseed"


apptainer exec justice_MultiProcessingEval.sif python3 hpc_run.py $nfe $myswf $myseed