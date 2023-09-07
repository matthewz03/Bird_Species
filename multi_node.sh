#!/bin/bash
# number of compute nodes
#SBATCH -N 4
#SBATCH -t 04:00:00
#SBATCH -p p100_normal_q
#SBATCH --gres=gpu:1
#SBATCH -A cmda3634_rjh
#SBATCH -o multi_node.out

# Submit this file as a job request with
# sbatch run.sh

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Get head node ip
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export MASTER_ADDR=$head_node_ip
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=INFO

# Unload all except default modules
module reset

# Load the modules you need
module load Anaconda3/2020.11
module load cuda11.6/toolkit
# module load apps site/infer-skylake/easybuild/arc.arcadm
# module load apps site/infer-skylake/easybuild/setup
# module load apps site/infer/easybuild/arc.arcadm
# module load apps site/infer/easybuild/setup
# module load NCCL/2.10.3-GCCcore-11.2.0-CUDA-11.4.1

# Convert Jupyter Notebook to Python (this may not be necessary if the program is already built)
jupyter nbconvert bird_species_classification.ipynb --to python

# Activate conda environment called 'pytorch'
source activate pytorch

# Print the number of threads for future reference
echo "Running Bird Classification"

# Run the program. Don't forget arguments!
# torchrun --standalone --nproc_per_node=gpu bird_species_classification.py
srun torchrun --nproc_per_node=1 --nnodes=4 --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$head_node_ip:42203 bird_species_classification.py
# srun torchrun \
# --nnodes 4 \
# --nproc_per_node=1 \
# --rdzv_id $RANDOM \
# --rdzv_backend c10d \
# --rdzv_endpoint $head_node_ip:12345 \
# bird_species_classification.py

# The script will exit whether we give the "exit" command or not.
exit

