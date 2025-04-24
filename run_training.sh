#!/bin/bash
#SBATCH --job-name=texture-diffusion-training

#SBATCH --output=/home/logan.boehm/diffusion/NN-ICP/logs/diffusion-train.%J.txt
#SBATCH --error=/home/logan.boehm/diffusion/NN-ICP/logs/diffusion-train.%J.err
#ASBATCH --mail-type=END
#ASBATCH --mail-user=lboehm2020@my.fit.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=60:00:00
#SBATCH --mem=40GB

#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

echo "Started at $(date)"
echo "Nodelist: $SLURM_NODELIST"
echo "NumNodes: $SLURM_NNODES"
echo "NumProcs: $SLURM_NPROCS"
echo "Working Dir: $pwd"

echo "Loading modules..."

module load cuda/11.6

cd ~/diffusion/NN-ICP/

echo "Starting GPU monitor"
# Start GPU monitoring in the background
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 5 > gpu_usage.log &

# Save the monitoring process ID to kill it later
MONITOR_PID=$!

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pose-rec

python3 -u train.py

echo "Done!"

# Stop GPU monitoring after your job is done
kill $MONITOR_PID
