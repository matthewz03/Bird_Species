#!/bin/bash
# number of compute nodes
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -p p100_normal_q
#SBATCH --gres=gpu:2
#SBATCH -A cmda3634_rjh
#SBATCH -o run.out

# Submit this file as a job request with
# sbatch run.sh

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Unload all except default modules
module reset

# Load the modules you need
module load Anaconda3/2020.11
module load cuda11.6/toolkit

# Compile (this may not be necessary if the program is already built)
jupyter nbconvert bird_species_classification.ipynb --to python

# Print the number of threads for future reference
echo "Running Bird Classification"

# Run the program. Don't forget arguments!
torchrun --standalone --nproc_per_node=gpu bird_species_classification.py

# The script will exit whether we give the "exit" command or not.
exit

