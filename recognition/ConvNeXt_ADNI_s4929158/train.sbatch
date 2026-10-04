#!/bin/bash
# Slurm batch script for Rangpur. Usage (from this folder):
#     mkdir -p logs
#     sbatch train.sbatch convnext        # or: cnn | mlp
#
# Fill in the PARTITION below. Find valid partition names with:  sinfo -s
# and the GPU request syntax in the Rangpur documentation linked from the course.
#SBATCH --job-name=adni_train
#SBATCH --partition=PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

MODEL=${1:-convnext}

source ~/venvs/comp3710/bin/activate
python train.py --model "$MODEL" --num-workers "$SLURM_CPUS_PER_TASK" --amp
